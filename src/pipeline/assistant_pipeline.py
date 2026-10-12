"""
assistant_pipeline.py — threaded, fully local voice assistant.

Threads
  AudioThread      mic → (WebRTC gate → wake word) | capture (Silero endpointing)
                   | barge-in detection while the assistant is busy
  ProcessingThread STT → intent/LLM(stream) → TTS(stream), one turn at a time
  SpeakerThread    (audio/speaker.py) the only thing that makes sound
  ProgressNarrator per-turn helper that keeps the user informed while waiting

Conversation model
  IDLE   → wake word → AWAKE (conversation open: no wake word needed for follow-ups)
  AWAKE  → CONVERSATION_TIMEOUT_S of quiet → warning → IDLE
  A "turn" = capture → STT → answer. At most one turn is active; speaking over
  the assistant (barge-in) cancels the active turn and starts the next one.

Cancellation is epoch based: Speaker.epoch is bumped by Speaker.stop(); every
stage of a turn remembers the epoch it started in and abandons work as soon as
the epoch moves on (LLM stream is hung up, queued TTS is dropped).

Every transition is recorded in the structured trace (common/log.py) which the
automated evaluator in tests/e2e consumes.
"""

from __future__ import annotations

import json
import queue
import re
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from wave import open as wave_open

import numpy as np

from audio.aec import EchoCanceller, ReferenceBuffer
from audio.mic import MicBase, PyAudioMic
from audio.sounds import build_earcons, make_calibration
from audio.speaker import AudioItem, PyAudioBackend, Speaker, SpeakerBackend
from common import stamp
from common.config import load_config
from common.event_logger import logger
from common.log import get_logger, now_s, trace
from common.servers import SERVERS, _ping, _start, port_in_use, wait_for_servers
from llm.gemma import GemmaLLM, LLMError, LLMTimeout
from pipeline import textutils as tu
from pipeline.speech import SpeechStreamer
from stt.whisper_cpp import STTResult, WhisperCppSTT
from tts.kokoro import KokoroTTS
from tts.phrases import PhrasePlayer
from vad.silero import SileroGate
from vad.webrtc import WebRTCGate
from wakeword.listen import WakeWordListener

log = get_logger("pipeline")

BLANK_TOKENS = tu.BLANK_TOKENS   # re-exported for older tools


# ──────────────────────────────────────────────────────────────────────────────
# Data carried between threads
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Utterance:
    samples: List[np.ndarray]          # int16 frames
    source: str                        # "wake" | "followup" | "barge"
    speech_ms: float
    rms: float
    t_eos: float                       # monotonic: when the user (approximately) stopped talking
    t_queued: float
    wake_score: float = 0.0
    text: Optional[str] = None         # pre-transcribed (interjections are transcribed before they cancel a turn)
    stt: Optional[STTResult] = None    # confidence of the transcript when it is known
    stt_future: Optional[Future] = None  # speculative transcript started while the user was still in the silence hold
    wakeish: bool = False              # the capture contained the wake phrase (only then is it stripped from the text)
    paused: bool = False               # our answer is paused (not stopped) until this speech is judged


@dataclass
class Capture:
    """State of one in-progress utterance capture (audio thread only)."""
    source: str
    t_start: float
    frames: List[np.ndarray] = field(default_factory=list)
    sbuf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    ms: float = 0.0                    # audio time captured
    speech_ms: float = 0.0
    last_speech_ms: float = 0.0
    heard: bool = False
    wake_score: float = 0.0
    wakeish: bool = False              # the wake phrase was heard in this capture
    paused: bool = False               # started by speech over our answer, which is paused meanwhile
    speech_sq: float = 0.0             # for the user's typical speaking level
    speech_frames: int = 0
    spec: Optional["Spec"] = None      # speculative STT in flight / done


@dataclass
class Spec:
    fut: Future
    last_speech_ms: float              # speech position the snapshot covered; newer speech invalidates it


# ──────────────────────────────────────────────────────────────────────────────
# Slow-inference narration
# ──────────────────────────────────────────────────────────────────────────────
class ProgressNarrator(threading.Thread):
    """
    Keeps the user informed while the answer is not ready yet. Started as soon
    as the user finishes speaking, stopped as soon as answer audio is queued.

      0 s            soft "ack" tick (played by the audio thread)
      first_phrase_s spoken filler ("Let me think about that.")
      then           a gentle pulse every pulse_every_s
      every phrase_every_s  a spoken reassurance ("Still working on it.")
    """

    def __init__(self, pipe: "Pipeline", epoch: int, turn: int, t_eos: float):
        super().__init__(name=f"Narrator{turn}", daemon=True)
        self.pipe, self.epoch, self.turn, self.t_eos = pipe, epoch, turn, t_eos
        self._halt = threading.Event()
        self.events = 0

    def stop(self) -> None:
        self._halt.set()

    def _alive(self) -> bool:
        return not self._halt.is_set() and self.pipe.speaker.epoch == self.epoch and self.pipe.running.is_set()

    def run(self) -> None:
        cfg = self.pipe.cfg
        waited_phrases = 0
        next_at = cfg.progress_first_phrase_s
        last_phrase_at = None
        while self._alive():
            timeout = max(0.0, self.t_eos + next_at - time.monotonic())
            if self._halt.wait(min(timeout, 0.25)):
                return
            if time.monotonic() < self.t_eos + next_at:
                continue
            if not self._alive():
                return
            elapsed = time.monotonic() - self.t_eos
            need_phrase = last_phrase_at is None or (elapsed - last_phrase_at) >= cfg.progress_phrase_every_s
            if need_phrase and self.pipe.llm_streaming_recently():
                # Tokens are flowing, the first sentence is imminent: a spoken filler
                # would only delay it (the speaker plays one sound at a time).
                next_at = elapsed + 0.4
                continue
            if need_phrase:
                group = "progress_first" if waited_phrases == 0 else (
                    "progress_next" if waited_phrases == 1 else "progress_long")
                picked = self.pipe.phrases.pick(group)
                if picked:
                    key, pcm, sr = picked
                    trace.emit("progress", turn=self.turn, kind="phrase", key=key, waited_s=round(elapsed, 2))
                    log.info("still waiting (%.1fs) → saying '%s'", elapsed, key)
                    self.pipe.speaker.enqueue(AudioItem("phrase", pcm, sr, label=key, epoch=self.epoch))
                    self.events += 1
                waited_phrases += 1
                last_phrase_at = elapsed
            else:
                trace.emit("progress", turn=self.turn, kind="pulse", waited_s=round(elapsed, 2))
                self.pipe._earcon("pulse", epoch=self.epoch)
                self.events += 1
            next_at = elapsed + cfg.progress_pulse_every_s


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self, cfg=None, mic: Optional[MicBase] = None,
                 speaker_backend: Optional[SpeakerBackend] = None):
        self.cfg = cfg or load_config()
        self._injected_mic = mic

        # VAD / wake word components.
        self.webrtc = WebRTCGate()
        self.wake = WakeWordListener()
        self.silero = SileroGate()

        # AI stage clients (HTTP only — no model loading here).
        self.stt = WhisperCppSTT(self.cfg)
        self.llm = GemmaLLM(self.cfg)
        self.tts = KokoroTTS(self.cfg)
        self.phrases = PhrasePlayer(self.cfg)

        # Audio out: one serialised speaker for everything the assistant says.
        self._backend = speaker_backend
        self.speaker: Optional[Speaker] = None
        self.earcons = build_earcons(self.cfg.sample_rate)

        # Conversation memory (alternating user/assistant messages).
        self.history: deque = deque(maxlen=self.cfg.memory_turns * 2)
        self._last_response = ""

        # Thread coordination.
        self.utterance_queue: "queue.Queue[Utterance]" = queue.Queue(maxsize=1)
        self.turn_active = threading.Event()       # an utterance is queued or being processed
        self.conversation_mode = threading.Event() # AWAKE
        self.running = threading.Event()
        self.last_activity = time.monotonic()
        self._threads: List[threading.Thread] = []
        self._mic: Optional[MicBase] = None

        # Counters.
        self._m_turns = 0
        self._m_dropped = 0
        self._m_false_wakes = 0
        self._m_blank_stt = 0
        self._m_interrupt_detected = 0
        self._m_interrupt_success = 0
        self._m_server_restarts = 0
        self._last_debug_wav = None
        self._echo_ratio = 0.0                     # learned speaker→mic coupling
        self._llm_first_token_t: Optional[float] = None
        self._user_level: Optional[float] = None   # typical RMS of the user's voice (speech frames)
        self.aec: Optional[EchoCanceller] = None
        if self.cfg.aec_enabled:
            self.aec = EchoCanceller(block=self.webrtc.frame_samples, partitions=self.cfg.aec_partitions,
                                     mu=self.cfg.aec_mu, sr=self.cfg.sample_rate,
                                     trust_db=self.cfg.aec_trusted_erle_db,
                                     ref=ReferenceBuffer(self.cfg.sample_rate))
        self._stt_pool: Optional[ThreadPoolExecutor] = None
        self._server_procs: dict = {}

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def llm_streaming_recently(self, within_s: float = 2.5) -> bool:
        t = self._llm_first_token_t
        return t is not None and (time.monotonic() - t) < within_s

    def _earcon(self, name: str, epoch: Optional[int] = None) -> None:
        pcm = self.earcons[name]
        self.speaker.enqueue(AudioItem("earcon", pcm, self.cfg.sample_rate, label=name, epoch=epoch))

    def _say(self, key: str, epoch: Optional[int] = None) -> Optional[AudioItem]:
        """Queue a cached system phrase. Never blocks."""
        got = self.phrases.get(key)
        if not got:
            log.warning("phrase '%s' unavailable", key)
            return None
        pcm, sr = got
        return self.speaker.enqueue(AudioItem("phrase", pcm, sr, label=key, epoch=epoch))

    def _pii(self, text: str, limit: Optional[int] = None) -> str:
        """What the user said / we answered, for logs and traces: redacted unless LOG_TRANSCRIPTS=1."""
        text = text if limit is None else text[:limit]
        if self.cfg.log_transcripts:
            return text
        return f"<{len(text)} chars>" if text else ""

    @staticmethod
    def _rms(frames: list) -> float:
        if not frames:
            return 0.0
        combined = np.concatenate(frames).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(combined ** 2)))

    @staticmethod
    def _is_breath_or_hiss(frames: list, sample_rate: int) -> bool:
        """True if audio looks like a breath or hiss (high centroid AND high ZCR)."""
        if not frames:
            return False
        pcm = np.concatenate(frames).astype(np.float32) / 32768.0
        if len(pcm) < 64:
            return False
        zcr = float(np.mean(np.abs(np.diff(np.sign(pcm)))) / 2)
        fft_mag = np.abs(np.fft.rfft(pcm))
        freqs = np.fft.rfftfreq(len(pcm), d=1.0 / sample_rate)
        centroid = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-9))
        return centroid > 4000 and zcr > 0.35

    def save_debug_wav(self, samples, prefix="utterance") -> Optional[Path]:
        if not self.cfg.debug_save_wav or not samples:
            return None
        out_dir = Path(self.cfg.debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{prefix}_{int(time.time() * 1000)}.wav"
        with wave_open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.cfg.sample_rate)
            wf.writeframes(np.concatenate(samples).astype(np.int16).tobytes())
        log.debug("debug wav saved: %s", path)
        self._last_debug_wav = path
        try:                                   # keep only the newest DEBUG_MAX_WAVS recordings
            old = sorted(out_dir.glob(f"{prefix}_*.wav"), key=lambda q: q.stat().st_mtime)[:-max(1, self.cfg.debug_max_wavs)]
            for q in old:
                q.unlink()
        except OSError as e:
            log.debug("debug wav pruning failed: %s", e)
        return path

    def _log_turn(self, turn_data: dict) -> None:
        """One JSON line per turn in a daily JSONL file."""
        try:
            log_dir = Path(self.cfg.turnlogdir)
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / f"turns_{time.strftime('%Y-%m-%d')}.jsonl", "a") as f:
                f.write(json.dumps(turn_data) + "\n")
        except OSError as e:               # telemetry must never break (or mask the outcome of) a turn
            log.warning("turn log not written: %s", e)

    def _enqueue_utterance(self, utt: Utterance) -> None:
        """Newest utterance wins: if one is still waiting unprocessed, replace it."""
        self.turn_active.set()
        while True:
            try:
                self.utterance_queue.put_nowait(utt)
                return
            except queue.Full:
                try:
                    old = self.utterance_queue.get_nowait()
                    self._m_dropped += 1
                    trace.emit("utterance_replaced", old_source=old.source)
                    log.warning("replacing unprocessed utterance with newer one")
                except queue.Empty:
                    pass

    # ------------------------------------------------------------------
    # Endpointing helpers
    # ------------------------------------------------------------------
    # Short follow-ups that really are questions when said over the assistant.
    _SHORT_QUESTIONS = {"why", "what", "how", "who", "when", "where", "which", "really", "huh", "pardon",
                        "again", "more", "sorry", "continue", "go on", "and", "so", "meaning", "explain",
                        "warum", "was", "wie", "wer", "wann", "wo", "wieso", "bitte"}

    def _spec_transcribe(self, frames) -> Optional[STTResult]:
        try:
            return self.stt.transcribe_ex(frames, self.cfg.sample_rate)
        except Exception as e:
            log.debug("speculative STT failed: %s", e)
            return None

    def _endpoint_hold(self, cap: "Capture", silence_ms: float) -> float:
        """
        How long to wait in silence before declaring the utterance finished.
        The transcript of what was said so far (speculative STT) tells us how patient to be:
          bare wake phrase     → wait long for the actual request
          ends like a sentence → answer sooner
          trails off ("and…")  → they are probably still thinking, wait longer
        """
        cfg = self.cfg
        hold = float(cfg.utterance_silence_hold_ms)
        spec = cap.spec
        if spec is None:
            return hold
        if not spec.fut.done():
            # Result pending: a wake-only capture must not end before we know it is wake-only.
            return hold + 600.0 if cap.wakeish and silence_ms < hold + 600.0 else hold
        try:
            res = spec.fut.result()
        except Exception:
            return hold
        raw = res.text if res is not None else None
        kind = tu.endpoint_class(raw, cfg.wakeword, cap.wakeish)
        if kind == "wake_wait":
            return float(cfg.utterance_wake_hold_ms)
        if kind == "question":
            return float(cfg.utterance_early_hold_ms)
        if kind == "hesitation":
            return float(cfg.utterance_hesitation_hold_ms)
        return hold

    # ------------------------------------------------------------------
    # Thread 1: Audio
    # ------------------------------------------------------------------
    def _audio_thread(self) -> None:
        """Supervisor: an unexpected exception in the audio loop must never leave the assistant deaf."""
        failures = 0
        while self.running.is_set():
            t_start = time.monotonic()
            try:
                self._audio_loop()
                return
            except Exception as e:
                if time.monotonic() - t_start > 60:
                    failures = 0
                failures += 1
                log.exception("audio loop crashed (%d/5): %s", failures, e)
                trace.emit("error", where="audio_loop", msg=str(e))
                self.speaker.stop("audio_crash")
                if failures >= 5:
                    log.error("audio loop keeps failing — giving up")
                    self._earcon("error")
                    self._say("fallback_generic")
                    self.running.clear()
                    return
                time.sleep(0.5)

    def _audio_loop(self) -> None:
        cfg = self.cfg
        mic = self._mic
        fs = self.webrtc.frame_samples
        fms = fs / cfg.sample_rate * 1000.0
        chunk = self.silero.chunk_size
        chunk_ms = chunk / cfg.sample_rate * 1000.0

        pre_roll = deque(maxlen=max(1, int(max(cfg.utterance_pre_roll_ms, cfg.utterance_wake_pre_roll_ms) / fms)))
        n_pre = max(1, int(cfg.utterance_pre_roll_ms / fms))
        cap: Optional[Capture] = None

        # wake word gating
        ww_hang = 0
        ww_prev_speech = False
        # follow-up trigger
        reentry_hits = 0
        reentry_buf = np.zeros(0, dtype=np.int16)
        reentry_probs: deque = deque(maxlen=5)
        # barge-in detector
        bi_frames: List[np.ndarray] = []
        bi_speech_ms = 0.0
        bi_buf = np.zeros(0, dtype=np.int16)
        bi_session_start = None          # when the assistant became busy
        playing_prev = False
        play_start = 0.0
        play_gap_t = 0.0
        warned = False
        webrtc_prev = None
        noise_floor = 0.003            # slow estimate of the room's background RMS (quiet frames only)

        def silero_reset():
            self.silero.reset()

        def start_capture(source: str, seed: List[np.ndarray], wake_score: float = 0.0, paused: bool = False) -> None:
            nonlocal cap
            silero_reset()
            cap = Capture(source=source, t_start=time.monotonic(), wake_score=wake_score, wakeish=(source == "wake"),
                          paused=paused)
            self.wake.hits = 0
            trace.emit("capture_start", source=source, seed_ms=round(len(seed) * fms))
            for fr in seed:
                feed_capture(fr)

        def feed_capture(fr: np.ndarray) -> None:
            """Advance the capture by one 30 ms frame (Silero → speech accounting)."""
            c = cap
            c.frames.append(fr)
            c.ms += fms
            c.sbuf = np.concatenate([c.sbuf, fr])
            if c.source in ("followup", "interject") and not c.wakeish:
                # The wake phrase also works while the conversation is open: it changes how
                # patient we are afterwards ("Hey Jarvis … [pause] … question").
                sc = self.wake.model.predict(fr).get(cfg.wakeword, 0.0)
                self.wake.hits = (self.wake.hits + 1) if sc >= self.wake.threshold else 0
                if self.wake.hits >= self.wake.trigger_level:
                    self.wake.hits = 0
                    c.wakeish = True
                    trace.emit("wake", score=round(float(sc), 3), during="followup")
                    log.info("wake phrase heard during follow-up capture (score=%.2f)", sc)
            while len(c.sbuf) >= chunk:
                part, c.sbuf = c.sbuf[:chunk], c.sbuf[chunk:]
                prob = self.silero.predict(part.tobytes())
                changed, new_state, old_state, avg, started, t2 = self.silero.update(prob)
                if new_state == "speech":
                    c.speech_ms += chunk_ms
                    c.last_speech_ms = c.ms
                    c.heard = True
                    c.speech_sq += float(np.mean((part.astype(np.float32) / 32768.0) ** 2))
                    c.speech_frames += 1
                if changed:
                    log.debug("silero %s→%s avg=%.2f t=%.0fms speech=%.0fms", old_state, new_state, avg, c.ms, c.speech_ms)

        def finish_capture(reason: str) -> None:
            nonlocal cap, warned
            c = cap
            cap = None
            silence_ms = c.ms - c.last_speech_ms if c.heard else c.ms
            # Trim trailing silence down to post-roll.
            trim = max(0, int((silence_ms - cfg.utterance_post_roll_ms) / fms))
            frames = c.frames[: len(c.frames) - trim] if trim < len(c.frames) else c.frames
            rms = self._rms(frames)
            accepted = (reason in ("silence", "max")
                        and c.speech_ms >= cfg.utterance_min_speech_ms
                        and rms >= cfg.utterance_reject_rms)
            t_eos = time.monotonic() - silence_ms / 1000.0
            trace.emit("capture_end", source=c.source, reason=reason, accepted=accepted,
                       dur_ms=round(c.ms), speech_ms=round(c.speech_ms), rms=round(rms, 4),
                       silence_ms=round(silence_ms))
            logger.utterance_accepted(rms) if accepted else logger.utterance_rejected(rms)
            if not accepted:
                self._m_false_wakes += 1
                log.info("capture rejected (%s, %s): %.0fms audio, %.0fms speech, rms=%.4f",
                         c.source, reason, c.ms, c.speech_ms, rms)
                if c.paused:
                    # Whatever interrupted our answer was not speech: carry on where we left off.
                    self.speaker.resume("capture_rejected")
                    trace.emit("barge_in_noise", why=reason)
                if c.source == "wake":
                    # Woke up but nobody spoke: say nothing more, go back to sleep quietly.
                    self.conversation_mode.clear()
                    self._user_level = None
                    self._earcon("interrupt")
                    trace.emit("sleep", why="wake_without_speech")
                else:
                    self.last_activity = time.monotonic()
                return
            self.save_debug_wav(frames)
            warned = False
            self.last_activity = time.monotonic()
            log.info("utterance accepted (%s): %.0fms, speech %.0fms, rms=%.4f", c.source, c.ms, c.speech_ms, rms)
            if c.speech_frames:
                lvl = (c.speech_sq / c.speech_frames) ** 0.5
                self._user_level = lvl if self._user_level is None else 0.5 * self._user_level + 0.5 * lvl
            utt = Utterance(frames, c.source, c.speech_ms, rms, t_eos, time.monotonic(), c.wake_score,
                            stt_future=c.spec.fut if c.spec else None, wakeish=c.wakeish or c.source == "wake",
                            paused=c.paused)
            if c.source == "interject":
                threading.Thread(target=self._confirm_interject, args=(utt,), name="Interject", daemon=True).start()
                return
            if cfg.ack_earcon_enabled:
                self._earcon("ack")
            self._enqueue_utterance(utt)

        def reset_barge():
            nonlocal bi_frames, bi_speech_ms, bi_buf
            bi_frames, bi_speech_ms, bi_buf = [], 0.0, np.zeros(0, dtype=np.int16)

        frames_read = 0
        # Mic sample clock for the echo canceller: block k ended at mic_t0 + (k+1)*fms. Using the
        # wall-clock time at which read() returned would jitter by tens of ms (a late loop returns
        # buffered data) and wreck the delay alignment between mic and loudspeaker reference.
        mic_t0 = None
        lag_hist: deque = deque(maxlen=150)
        while self.running.is_set():
            pcm = mic.read(fs)
            if not self.running.is_set():
                break
            t = time.monotonic()
            frame = np.frombuffer(pcm, dtype=np.int16)
            if len(frame) < fs:
                continue
            raw_rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) / 32768.0
            if self.aec is not None:
                if mic_t0 is None:
                    mic_t0 = t - fms / 1000.0
                t_nominal = mic_t0 + (frames_read + 1) * fms / 1000.0
                lag_hist.append(t - t_nominal)
                if frames_read % 100 == 99:               # follow slow clock drift / start-up latency
                    m = min(lag_hist)
                    if abs(m) > 0.004:
                        mic_t0 += m
                frame = self.aec.process(frame, t_nominal)  # subtract our own loudspeaker output
                pcm = frame.tobytes()
                if self.aec.active and frames_read % 33 == 0:
                    trace.emit("aec", erle_db=round(self.aec.erle_db, 1), converged=self.aec.converged,
                               resets=self.aec.resets, blocks=self.aec.trained_blocks,
                               mic_rms=round(raw_rms, 4),
                               out_rms=round(float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) / 32768.0, 4))
            frames_read += 1
            rms_now = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) / 32768.0
            logger.amplitude(rms_now)
            pre_roll.append(frame)

            awake = self.conversation_mode.is_set()
            sp = self.speaker
            sp_kind = sp.active_kind()
            sp_active = sp.is_active()
            busy = self.turn_active.is_set() or sp_active

            # ── 1. capturing an utterance ─────────────────────────────────
            if cap is not None:
                feed_capture(frame)
                silence_ms = cap.ms - cap.last_speech_ms if cap.heard else cap.ms
                if cap.ms >= cfg.utterance_max_ms:
                    finish_capture("max")
                    continue
                if cap.speech_ms < cfg.utterance_min_speech_ms:
                    if cap.ms >= cfg.utterance_noise_abort_ms + (1500 if cap.source == "wake" else 0):
                        finish_capture("no_speech")
                    continue
                # Speculative STT: transcribe while the silence hold is still running.
                if cap.spec is not None and cap.spec.last_speech_ms != cap.last_speech_ms:
                    cap.spec = None                 # user resumed talking → stale
                if (cap.spec is None and silence_ms >= cfg.speculative_stt_ms and (cap.source != "interject" or cap.paused)
                        and self._stt_pool is not None):
                    # Exactly the audio finish_capture() would send (speech + post-roll of silence),
                    # so the speculative transcript is the transcript of the final utterance.
                    trim = max(0, int((silence_ms - cfg.utterance_post_roll_ms) / fms))
                    snap = list(cap.frames[: len(cap.frames) - trim]) if trim < len(cap.frames) else list(cap.frames)
                    cap.spec = Spec(self._stt_pool.submit(self._spec_transcribe, snap), cap.last_speech_ms)
                hold = self._endpoint_hold(cap, silence_ms)
                if silence_ms >= hold:
                    finish_capture("silence")
                continue

            # ── 2. assistant is busy: watch for barge-in ──────────────────
            if awake and busy:
                if sp_kind == "earcon":
                    continue                      # ignore the beeps themselves
                if bi_session_start is None:
                    bi_session_start = t
                    reset_barge()
                    silero_reset()      # the model is streaming: do not carry state over from unrelated audio
                level = sp.recent_level()
                # Calibration window: the first ~0.45 s after our audio (re)starts being audible.
                # We learn the speaker→mic coupling there and never barge-in on it.
                if level > 0.02:
                    if not playing_prev:
                        play_start = t
                    playing_prev = True
                    play_gap_t = t
                elif playing_prev and t - play_gap_t > 0.4:
                    playing_prev = False
                in_calibration = playing_prev and (t - play_start) < 0.45
                # Learn how loudly our own playback leaks into the mic.
                if level > 0.02 and bi_speech_ms == 0.0:
                    sample = rms_now / level
                    if in_calibration or sample < self._echo_ratio:
                        self._echo_ratio = 0.7 * self._echo_ratio + 0.3 * sample
                    else:
                        self._echo_ratio = 0.995 * self._echo_ratio + 0.005 * sample
                if level > 0.02 and in_calibration:
                    continue            # still learning how loud our own voice is at the mic

                # Until the echo canceller has converged, demand a larger margin over the learned echo.
                cautious = (self.aec is not None and not self.aec.trusted and self._echo_ratio > 0.03)
                margin = cfg.interrupt_echo_margin * (1.6 if cautious else 1.0)
                echo_floor = self._echo_ratio * level * margin
                loud_enough = rms_now >= max(cfg.interrupt_energy_threshold, 2.5 * noise_floor, echo_floor)

                bi_buf = np.concatenate([bi_buf, frame])
                bi_frames.append(frame)
                if len(bi_frames) > int(1500 / fms):
                    bi_frames = bi_frames[-int(1500 / fms):]
                # Raw per-chunk Silero probability (not the 5-chunk smoothed state): a one-word
                # command like "stop" is only ~8 chunks long and must be recognised in time.
                while len(bi_buf) >= chunk:
                    part, bi_buf = bi_buf[:chunk], bi_buf[chunk:]
                    prob = self.silero.predict(part.tobytes())
                    self.silero.history.append(prob)            # keep Silero's smoothing state coherent
                    if len(self.silero.history) > self.silero.history_max:
                        self.silero.history.pop(0)
                    if prob >= cfg.silero_threshold and loud_enough:
                        bi_speech_ms += chunk_ms
                    else:
                        bi_speech_ms = max(0.0, bi_speech_ms - chunk_ms * 1.5)
                        if bi_speech_ms == 0.0 and len(bi_frames) > int(600 / fms):
                            bi_frames = bi_frames[-int(300 / fms):]

                # Immediate feedback: as soon as someone seems to be talking, turn our own voice down
                # (also cuts the echo we feed back into the mic). Restored if it was a false alarm.
                if cfg.barge_in_duck_gain < 1.0:
                    sp.set_duck(cfg.barge_in_duck_gain if bi_speech_ms >= cfg.barge_in_duck_after_ms else 1.0)

                if bi_speech_ms >= cfg.interrupt_min_speech_ms * (2 if cautious and level > 0.02 else 1):
                    rms = self._rms(bi_frames[-int(bi_speech_ms / fms):])
                    if cfg.interrupt_hiss_filter and self._is_breath_or_hiss(list(bi_frames), cfg.sample_rate):
                        reset_barge()
                        continue
                    self._m_interrupt_detected += 1
                    logger.interrupt(rms)
                    trace.emit("barge_in", rms=round(rms, 4), speech_ms=round(bi_speech_ms),
                               echo_floor=round(echo_floor, 4), speaker_kind=sp_kind,
                               turn_active=self.turn_active.is_set(),
                               erle_db=round(self.aec.erle_db, 1) if self.aec else None,
                               mode="pause" if sp_kind == "tts" else "interject")
                    log.info("BARGE-IN (rms=%.4f, speaker=%s, mode=%s)", rms, sp_kind,
                             "pause" if sp_kind == "tts" else "interject")
                    seed = list(bi_frames)
                    # Talking over answer audio → silence it *now* but only pause it: whether this was a
                    # request, "stop" or a cough is decided by STT afterwards (resume / stop). Over silence or
                    # filler ("thinking" phase) there is nothing to pause; the pending answer is cancelled
                    # only once STT has confirmed real speech.
                    paused = sp_kind == "tts"
                    if paused:
                        sp.pause("barge_in")
                    else:
                        sp.set_duck(1.0)
                    reset_barge()
                    bi_session_start = None
                    warned = False
                    start_capture("interject", seed, paused=paused)
                continue
            bi_session_start = None
            playing_prev = False
            if bi_frames or bi_speech_ms:
                reset_barge()
                self.speaker.set_duck(1.0)

            # ── 3. post-playback guard (speaker tail / room reverb) ───────
            if sp_kind == "calib":
                lvl = sp.recent_level()
                if lvl > 0.01:
                    self._echo_ratio = 0.9 * self._echo_ratio + 0.1 * (rms_now / lvl)
            if sp_active or t < sp.last_end + cfg.assistant_audio_cooldown_s:
                reentry_hits = 0
                ww_hang = 0
                continue

            # ── 4. AWAKE and quiet: follow-up speech or inactivity timeout ─
            if awake:
                is_speech = self.webrtc.vad.is_speech(pcm, self.webrtc.sample_rate)
                if not is_speech:
                    noise_floor = 0.995 * noise_floor + 0.005 * min(rms_now, 0.05)
                trigger_rms = max(cfg.conversation_reentry_rms, 2.5 * noise_floor)
                if self._user_level:
                    # Ignore anything much quieter than the user's own voice (TV, other rooms).
                    trigger_rms = max(trigger_rms, cfg.followup_level_ratio * self._user_level)
                if is_speech and rms_now >= trigger_rms:
                    if reentry_hits == 0:
                        silero_reset()
                    reentry_hits += 1
                    # Confirm with Silero so door slams / claps / chair noises don't open a capture.
                    reentry_buf = np.concatenate([reentry_buf, frame])
                    while len(reentry_buf) >= chunk:
                        part, reentry_buf = reentry_buf[:chunk], reentry_buf[chunk:]
                        reentry_probs.append(self.silero.predict(part.tobytes()))
                else:
                    reentry_hits = max(0, reentry_hits - 1) if is_speech else 0
                    if reentry_hits == 0:
                        reentry_probs.clear()
                        reentry_buf = np.zeros(0, dtype=np.int16)
                if (reentry_hits >= cfg.conversation_reentry_start_hits and len(reentry_probs) >= 3
                        and sum(reentry_probs) / len(reentry_probs) >= cfg.silero_threshold):
                    reentry_hits = 0
                    reentry_probs.clear()
                    reentry_buf = np.zeros(0, dtype=np.int16)
                    self.last_activity = t
                    warned = False
                    start_capture("followup", list(pre_roll)[-n_pre:])
                    continue
                elif reentry_hits >= cfg.conversation_reentry_start_hits + 8:
                    reentry_hits = 0                      # loud but not speech-like: keep ignoring

                idle_s = t - self.last_activity
                if not warned and idle_s >= cfg.conversation_timeout_s - cfg.conversation_warning_s:
                    warned = True
                    trace.emit("sleep_warning", idle_s=round(idle_s, 1))
                    self._say("going_to_sleep")
                if idle_s >= cfg.conversation_timeout_s:
                    self.conversation_mode.clear()
                    warned = False
                    self._user_level = None
                    trace.emit("sleep", why="timeout", idle_s=round(idle_s, 1))
                    log.info("conversation timeout after %.0fs → idle", idle_s)
                    self._earcon("sleep")
                    self._say("goodbye")
                continue

            # ── 5. IDLE: WebRTC gate → wake word ──────────────────────────
            is_speech = self.webrtc.vad.is_speech(pcm, self.webrtc.sample_rate)
            if not is_speech:
                noise_floor = 0.995 * noise_floor + 0.005 * min(rms_now, 0.05)
            state = "speech" if is_speech else "silence"
            if state != webrtc_prev:
                logger.webrtc(is_speech)
                webrtc_prev = state
                self.webrtc.state = state
            if is_speech:
                ww_hang = 12
            elif ww_hang > 0:
                ww_hang -= 1
            else:
                ww_prev_speech = False
                continue

            frames_to_score = [frame]
            if not ww_prev_speech:
                # Speech just began: also give the model the lead-in it would otherwise miss.
                frames_to_score = list(pre_roll)[-6:]
                ww_prev_speech = True
            for wf in frames_to_score:
                score = self.wake.model.predict(wf).get(cfg.wakeword, 0.0)
                self.wake.hits = (self.wake.hits + 1) if score >= self.wake.threshold else 0
                if self.wake.hits >= self.wake.trigger_level:
                    break
            if self.wake.hits >= self.wake.trigger_level:
                self.wake.hits = 0
                try:
                    self.wake.model.reset()
                except Exception:
                    pass
                ww_prev_speech = False
                ww_hang = 0
                self.conversation_mode.set()
                self.last_activity = t
                warned = False
                logger.wakeword(score)
                trace.emit("wake", score=round(float(score), 3))
                log.info("WAKE WORD '%s' score=%.2f → AWAKE", cfg.wakeword, score)
                self._earcon("wake")
                start_capture("wake", list(pre_roll), wake_score=float(score))

    # ------------------------------------------------------------------
    # Thread 2: Processing
    # ------------------------------------------------------------------
    def _processing_thread(self) -> None:
        while self.running.is_set():
            try:
                utt = self.utterance_queue.get(timeout=0.4)
            except queue.Empty:
                continue
            self.turn_active.set()
            try:
                self._run_turn(utt)
            except Exception as e:  # never let one bad turn kill the assistant
                log.exception("turn crashed: %s", e)
                trace.emit("error", where="turn", msg=str(e))
                logger.error(str(e))
                self.speaker.stop("turn_crash")
                self._earcon("error")
                self._say("fallback_generic")
            finally:
                self.last_activity = time.monotonic()
                if self.utterance_queue.empty():
                    self.turn_active.clear()

    def _awaiting_answer(self) -> bool:
        """The assistant's last reply ended in a question, so "yes"/"okay"/"no" over it are answers."""
        return tu.ends_with_question(self._last_response)

    def _is_noise(self, raw: Optional[str], res: STTResult, cut_in: bool) -> bool:
        """
        True when the transcript carries no real speech. A confident stop/sleep intent always counts as
        speech, even for words whisper also hallucinates on silence ("Bye.").
        """
        text = tu.clean_transcript(raw or "")
        if not text:
            return True
        if tu.is_hallucination(raw):
            core = tu.strip_wake_phrase(text, self.cfg.wakeword, strict=True)
            intent = tu.classify_intent(core, barge=cut_in, awaiting_answer=self._awaiting_answer())
            return not (intent in ("stop", "sleep") and res.confident(self.cfg.stt_min_logprob))
        return False

    def _confirm_interject(self, utt: Utterance) -> None:
        """
        The user spoke while the assistant was thinking, or while its answer was paused. Decide, from STT,
        whether it was real speech:
          noise        → resume the paused answer / keep the pending one
          muffled word → (only when we paused an answer) it almost always means "stop": stop, say nothing
          speech       → stop the current answer and treat this as the next request
        """
        cfg = self.cfg
        sp = self.speaker
        try:
            res = self._get_transcript(utt)
        except Exception as e:
            log.warning("interjection STT failed: %s", e)
            if utt.paused:
                sp.resume("interject_stt_failed")
            return
        raw = res.text
        core = tu.strip_wake_phrase(tu.clean_transcript(raw or ""), cfg.wakeword, strict=not utt.wakeish)
        intent = tu.classify_intent(core, barge=True, awaiting_answer=self._awaiting_answer())
        if self._is_noise(raw, res, cut_in=True):
            trace.emit("interject_rejected", raw=self._pii(raw or "", 60))
            log.info("interjection was noise (%s) — keeping the answer", self._pii(raw or "", 60))
            if utt.paused:
                sp.resume("interject_noise")
            return
        if intent is None and not res.confident(cfg.stt_min_logprob):
            if utt.paused and utt.speech_ms < 1400:
                # A muffled short word over our answer ("Snobber." for "stop"): obey it, do not guess a request.
                trace.emit("interject_confirmed", text=self._pii(raw or "", 80), as_="stop_unclear")
                log.info("unclear short cut-in → stop")
                sp.stop("interject_unclear")
                self._m_interrupt_detected += 1
                self._m_interrupt_success += 1
                return
            if not utt.paused:
                trace.emit("interject_rejected", raw=self._pii(raw or "", 60))
                log.info("unclear interjection while thinking — keeping the pending answer")
                return
        utt.text, utt.stt = raw, res
        trace.emit("interject_confirmed", text=self._pii(raw or "", 80))
        log.info("interjection confirmed: %s → cancelling current turn", self._pii(raw or "", 80))
        sp.stop("interject")
        self._enqueue_utterance(utt)

    def _get_transcript(self, utt: Utterance) -> STTResult:
        if utt.text is not None:
            return utt.stt or STTResult(utt.text)
        if utt.stt_future is not None:
            try:
                res = utt.stt_future.result(timeout=30)
                if res is not None:
                    trace.emit("stt_speculative_used")
                    return res
            except Exception:
                pass
        return self._transcribe(utt)

    # -- STT -------------------------------------------------------------
    def _transcribe(self, utt: Utterance) -> STTResult:
        """Fresh transcription; raises on server failure."""
        logger.stt_start()
        res = self.stt.transcribe_ex(utt.samples, self.cfg.sample_rate)
        logger.stt_result(res.text or "")
        return res

    # -- one turn --------------------------------------------------------
    def _run_turn(self, utt: Utterance) -> None:
        cfg = self.cfg
        sp = self.speaker
        epoch = sp.epoch
        self._m_turns += 1
        turn = self._m_turns
        t_turn0 = time.monotonic()
        trace.emit("turn_start", turn=turn, source=utt.source,
                   queue_wait_ms=round((t_turn0 - utt.t_queued) * 1000),
                   since_eos_ms=round((t_turn0 - utt.t_eos) * 1000))
        self._llm_first_token_t = None
        outcome, response, spoken_text = "completed", "", ""
        stt_ms = llm_ms = 0.0
        transcript = ""
        narrator = ProgressNarrator(self, epoch, turn, utt.t_eos)
        narrator.start()
        speech: Optional[SpeechStreamer] = None

        def cancelled() -> bool:
            return sp.epoch != epoch or not self.running.is_set()

        try:
            # ── STT ──────────────────────────────────────────────────────
            t0 = time.monotonic()
            try:
                stt_res = self._get_transcript(utt)
                raw = stt_res.text
            except Exception as e:
                log.error("STT failed: %s", e)
                trace.emit("error", where="stt", msg=str(e), turn=turn)
                logger.error(f"stt: {e}")
                narrator.stop()
                outcome = "stt_error"
                self._earcon("error", epoch)
                self._say("fallback_stt", epoch)
                self._recover_servers()
                return
            stt_ms = (time.monotonic() - t0) * 1000
            transcript = tu.clean_transcript(raw or "")
            trace.emit("stt_done", turn=turn, ms=round(stt_ms), raw=self._pii(raw or "", 600),
                       text=self._pii(transcript, 600),
                       logprob=None if stt_res.avg_logprob is None else round(stt_res.avg_logprob, 2),
                       min_word=None if stt_res.min_word_prob is None else round(stt_res.min_word_prob, 2))
            log.info("STT (%.0fms): %s", stt_ms, self._pii(raw or ""))

            if cancelled():
                outcome = "cancelled"
                return

            cut_in = utt.source in ("barge", "interject")
            # Only a capture that contained the wake phrase has it stripped: "Service animals in Germany…"
            # said as a follow-up must reach the model intact.
            core = tu.strip_wake_phrase(transcript, cfg.wakeword, strict=not utt.wakeish)
            if self._is_noise(raw, stt_res, cut_in):
                self._m_blank_stt += 1
                outcome = "blank"
                narrator.stop()
                if utt.source == "wake":
                    self._say("yes", epoch)
                elif utt.speech_ms >= 700:
                    self._say("didnt_catch", epoch)
                return
            if not core:
                # Only the wake phrase: invite the real request.
                outcome = "wake_only"
                narrator.stop()
                self._say("yes", epoch)
                return

            # ── Intents ──────────────────────────────────────────────────
            intent = tu.classify_intent(core, barge=cut_in, awaiting_answer=self._awaiting_answer())
            if intent is None and not stt_res.confident(cfg.stt_min_logprob):
                # whisper itself does not believe this transcript ("Snobber."): don't feed garbage to the LLM.
                narrator.stop()
                trace.emit("stt_unclear", turn=turn, text=self._pii(core, 80), logprob=stt_res.avg_logprob)
                log.info("unclear transcript %s (logprob=%.2f) → not answering it", self._pii(core), stt_res.avg_logprob or 0.0)
                if cut_in and utt.speech_ms < 1400:
                    outcome = "stop"             # a muffled word over the assistant almost always means "stop"
                elif utt.source == "wake" and utt.speech_ms < 1300:
                    outcome = "wake_only"        # a mangled "Hey Jarvis": invite the real request
                    self._say("yes", epoch)
                else:
                    outcome = "unclear"
                    self._say("didnt_catch", epoch)
                return
            if (intent is None and cut_in and utt.speech_ms < 700 and len(core.split()) <= 2
                    and core.lower().strip(" .?!,") not in self._SHORT_QUESTIONS):
                # A brief unrecognised word over the assistant ("Snava." for "stop") is a stop, not a request.
                intent = "stop"
                trace.emit("stt_short_cut_in", turn=turn, text=self._pii(core, 40), logprob=stt_res.avg_logprob)
            if intent:
                trace.emit("intent", turn=turn, name=intent)
                log.info("intent: %s", intent)
            if intent == "stop":
                narrator.stop()
                was_busy = utt.source in ("barge", "interject")
                sp.stop("voice_stop")
                outcome = "stop"
                if not was_busy:
                    self._say("okay", sp.epoch)
                return
            if intent == "sleep":
                narrator.stop()
                outcome = "sleep"
                self.conversation_mode.clear()
                trace.emit("sleep", why="voice")
                self._earcon("sleep", epoch)
                self._say("goodbye", epoch)
                return
            if intent == "repeat":
                response = self._last_response
                if not response:
                    narrator.stop()
                    self._say("nothing_to_repeat", epoch)
                    outcome = "repeat_empty"
                    return
                outcome = "repeat"
                self.history.append({"role": "user", "content": core})
                speech = SpeechStreamer(self.tts, sp, epoch, turn)
                narrator.stop()
                chunker = tu.SentenceChunker()
                for ch in chunker.feed(response + " ") + chunker.flush():
                    speech.submit(ch)
                speech.finish()
                ok = speech.wait()
                spoken_text = " ".join(speech.spoken)
                self.history.append({"role": "assistant", "content": tu.trim_for_memory(response, cfg.memory_assistant_max_chars)})
                if not ok:
                    outcome = "interrupted" if cancelled() else "tts_error"
                return
            fast = tu.answer_intent(intent, None) if intent else None

            # ── Answer: instant or LLM-streamed ─────────────────────────
            speech = SpeechStreamer(self.tts, sp, epoch, turn)
            chunker = tu.SentenceChunker()
            history = list(self.history)
            parts: List[str] = []
            llm_error: Optional[Exception] = None
            first_chunk_seen = False

            def emit_chunks(chunks):
                nonlocal first_chunk_seen
                for ch in chunks:
                    if not first_chunk_seen:
                        first_chunk_seen = True
                        narrator.stop()
                        trace.emit("first_chunk", turn=turn, since_eos_ms=round((time.monotonic() - utt.t_eos) * 1000),
                                   text=self._pii(ch, 80))
                    speech.submit(ch)

            if fast:
                outcome = "fast_" + intent
                response = fast
                emit_chunks([fast])
            else:
                logger.llm_start()
                t_llm = time.monotonic()
                trace.emit("llm_start", turn=turn, history=len(history))
                try:
                    for delta in self.llm.stream(core, history, cancel=cancelled):
                        if not parts:
                            self._llm_first_token_t = time.monotonic()
                            trace.emit("llm_first_token", turn=turn, ms=round((time.monotonic() - t_llm) * 1000))
                        parts.append(delta)
                        emit_chunks(chunker.feed(delta))
                    llm_ms = (time.monotonic() - t_llm) * 1000
                    if not cancelled():
                        rest = chunker.flush()
                        if self.llm.last_finish_reason == "length" and rest:
                            rest = [tu.trim_to_sentence(r) for r in rest]
                        emit_chunks(rest)
                except LLMError as e:
                    llm_error = e
                    llm_ms = (time.monotonic() - t_llm) * 1000
                logger.llm_result()
                response = "".join(parts).strip()
                trace.emit("llm_done", turn=turn, ms=round(llm_ms), tokens=self.llm.last_tokens,
                           finish=self.llm.last_finish_reason, chars=len(response),
                           error=str(llm_error) if llm_error else None)
                log.info("LLM (%.0fms, %d tok, %s): %s", llm_ms, self.llm.last_tokens,
                         self.llm.last_finish_reason, self._pii(response, 120))

            narrator.stop() if (first_chunk_seen or llm_error) else None
            speech.finish()
            ok = speech.wait()
            spoken_text = " ".join(speech.spoken)

            if llm_error is not None:
                outcome = "llm_timeout" if isinstance(llm_error, LLMTimeout) else "llm_error"
                log.error("LLM failed: %s", llm_error)
                trace.emit("error", where="llm", msg=str(llm_error), turn=turn)
                logger.error(f"llm: {llm_error}")
                if not cancelled():
                    self._earcon("error", epoch)
                    self._say("llm_timeout" if isinstance(llm_error, LLMTimeout) else "fallback_llm", epoch)
                    self._recover_servers()
            elif speech.failed and not cancelled():
                outcome = "tts_error"
                self._earcon("error", epoch)
                self._say("fallback_tts", epoch)
                self._recover_servers()
            elif cancelled():
                outcome = "interrupted"
            elif not response:
                outcome = "empty_response"
                self._say("fallback_generic", epoch)
            elif not ok:
                outcome = "interrupted"

            # Memory: only remember what the user actually heard.
            remembered = response if outcome in ("completed",) or outcome.startswith("fast_") else spoken_text
            if outcome == "interrupted":
                # Only what was actually heard, with a plain ellipsis (a bracketed marker got fed back to the model).
                remembered = (tu.trim_for_memory(spoken_text, cfg.memory_assistant_max_chars).rstrip(" .,;:") + " …").strip()
            if outcome in ("completed", "interrupted") or outcome.startswith("fast_"):
                self.history.append({"role": "user", "content": core})
                self.history.append({"role": "assistant",
                                     "content": tu.trim_for_memory(remembered, cfg.memory_assistant_max_chars) or "(no answer)"})
                self._last_response = response or self._last_response
            if outcome == "interrupted":
                self._m_interrupt_success += 1
        finally:
            narrator.stop()
            if speech is not None:
                speech.finish()
            total_ms = (time.monotonic() - t_turn0) * 1000
            since_eos = (time.monotonic() - utt.t_eos) * 1000
            first_audio_ms = None
            if speech is not None and speech.first_audio_t is not None:
                first_audio_ms = round((speech.first_audio_t - utt.t_eos) * 1000)
            trace.emit("turn_end", turn=turn, outcome=outcome, total_ms=round(total_ms),
                       since_eos_ms=round(since_eos), first_audio_ms=first_audio_ms,
                       stt_ms=round(stt_ms), llm_ms=round(llm_ms), transcript=self._pii(transcript, 600),
                       response=self._pii(response, 600), spoken=self._pii(spoken_text, 600), narration=narrator.events)
            if cfg.metrics_enabled:
                log.info("METRICS turn=%d outcome=%s stt=%.0fms llm=%.0fms first_audio=%sms total=%.0fms narration=%d",
                         turn, outcome, stt_ms, llm_ms, first_audio_ms, total_ms, narrator.events)
                self._log_turn({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "turn": turn,
                    "transcript": self._pii(transcript), "response_chars": len(response),
                    "stt_ms": round(stt_ms, 1), "llm_ms": round(llm_ms, 1),
                    "total_ms": round(total_ms, 1), "first_audio_ms": first_audio_ms,
                    "interrupted": outcome == "interrupted", "failed": outcome.endswith("_error") or outcome == "llm_timeout",
                    "outcome": outcome, "debug_wav": str(self._last_debug_wav) if self._last_debug_wav else None,
                })
            self.last_activity = time.monotonic()

    # ------------------------------------------------------------------
    # Server error recovery
    # ------------------------------------------------------------------
    def _recover_servers(self) -> None:
        """
        Restart any server that stopped answering its health check. A server that was just
        (re)started and is still loading its model is waited for, never started twice
        (the LLM can take a minute to load).
        """
        for srv in SERVERS:
            if _ping(srv["url"], timeout=3):
                continue
            name = srv["name"]
            log.error("%s server down at %s → restarting", name, srv["url"])
            trace.emit("server_down", name=name)
            spawns, deadline = 0, time.monotonic() + (120 if name == "LLM" else 60)
            proc = self._server_procs.get(name)
            while time.monotonic() < deadline and self.running.is_set():
                if proc is None or proc.poll() is not None:
                    # Something we did not start (or a hung copy) may still own the port: never fight it.
                    if port_in_use(srv):
                        log.warning("%s port %d is bound but unhealthy — waiting, not spawning a second copy",
                                    name, srv["port"])
                        time.sleep(1.5)
                        if _ping(srv["url"], timeout=3):
                            trace.emit("server_restarted", name=name, spawns=spawns)
                            break
                        continue
                    if spawns >= 3:
                        break
                    proc = self._server_procs[name] = _start(srv)
                    spawns += 1
                    self._m_server_restarts += 1
                if _ping(srv["url"], timeout=3):
                    log.info("%s healthy again (after %d spawn(s))", name, spawns)
                    trace.emit("server_restarted", name=name, spawns=spawns)
                    break
                time.sleep(1.5)
            else:
                log.error("%s still down after waiting", name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        seen = set()
        for srv in SERVERS:
            url, name = srv.get("url", ""), srv.get("name", "?")
            try:
                port = int(url.split("://", 1)[1].split("/", 1)[0].rsplit(":", 1)[1])
            except Exception:
                print(f"[{stamp()}] Config error: cannot parse port from {name} URL: {url}")
                sys.exit(1)
            if port in seen:
                print(f"[{stamp()}] Config error: duplicate port {port} in SERVERS")
                sys.exit(1)
            seen.add(port)
        self._validate_files()
        self._check_language_stack()

    def _on_mic_status(self, event: str) -> None:
        """Called from the audio thread when the microphone disappears / comes back."""
        trace.emit("mic_" + event)
        if self.speaker is not None:
            self._say("mic_lost" if event == "lost" else "mic_back")

    def _validate_files(self) -> None:
        """A wrong model path otherwise shows up as a server that silently never comes up."""
        cfg = self.cfg
        checks = [("STT", cfg.whisper_model, "WHISPER_MODEL"),
                  ("TTS", cfg.kokoro_model_path, "KOKORO_MODEL_PATH"),
                  ("TTS", cfg.kokoro_voices_path, "KOKORO_VOICES_PATH")]
        by_name = {s["name"]: s for s in SERVERS}
        missing = []
        for srv_name, path, var in checks:
            srv = by_name.get(srv_name)
            if srv is not None and (_ping(srv["url"], timeout=1) or port_in_use(srv)):
                continue                      # that server is already running: its files are its business
            if path and not Path(path).exists():
                missing.append((srv_name, var, path))
        if missing:
            for srv_name, var, path in missing:
                print(f"[{stamp()}] Missing file for the {srv_name} server: {path}  (set {var})")
            sys.exit(1)

    def _check_language_stack(self) -> None:
        """
        The assistant's language is decided in four places that must agree: prompt, STT model/language,
        TTS voice, cached phrases. Say loudly when they visibly do not (an English-only Whisper model
        cannot transcribe German, whatever the prompt asks for).
        """
        cfg = self.cfg
        model = Path(cfg.whisper_model).name.lower()
        german = "german" in cfg.llm_system_prompt.lower() or "deutsch" in cfg.llm_system_prompt.lower()
        problems = []
        if model.endswith(".en.bin") and (cfg.whisper_language != "en" or german):
            problems.append(f"STT model '{model}' is English-only but the assistant is configured for German; "
                            f"use a multilingual model (WHISPER_MODEL=…/ggml-base.bin or small.bin, WHISPER_LANGUAGE=de|auto)")
        if german and cfg.whisper_language == "en":
            problems.append("WHISPER_LANGUAGE=en: German speech will be transcribed as English-sounding text")
        if german:
            problems.append(f"TTS voice '{cfg.tts_voice}' and the cached phrases are English; Kokoro v1.0 ships no "
                            f"German voice, so German answers will be read with an English accent")
        for p in problems:
            log.warning("LANGUAGE STACK: %s", p)
        if problems:
            trace.emit("language_stack_warning", problems=problems)

    def start(self, wait_servers: bool = True) -> None:
        """Open devices, warm everything up and start the worker threads (non-blocking)."""
        trace.open() if trace.path is None else None
        log.info("Pipeline: WebRTC → openWakeWord(%s) → Silero → Whisper → Gemma(stream) → Kokoro(stream)",
                 self.cfg.wakeword_framework)
        log.info("wakeword=%s sr=%d memory_turns=%d silence_hold=%dms barge_in=%dms timeout=%ds",
                 self.cfg.wakeword, self.cfg.sample_rate, self.cfg.memory_turns,
                 self.cfg.utterance_silence_hold_ms, self.cfg.interrupt_min_speech_ms,
                 self.cfg.conversation_timeout_s)
        self._validate_config()
        if wait_servers:
            try:
                wait_for_servers()
            except RuntimeError as e:
                print(f"[{stamp()}] {e}")
                sys.exit(1)
        self.phrases.warm_up()
        threading.Thread(target=self.llm.warm_up, name="LLMWarmup", daemon=True).start()

        self._backend = self._backend or PyAudioBackend()
        self.speaker = Speaker(self._backend, drain_extra_s=self.cfg.tts_drain_extra_s,
                               pause_timeout_s=self.cfg.barge_in_pause_timeout_s)
        if self.aec is not None:
            self.speaker.taps.append(
                lambda t_item, off, seg, sr, item, gains: self.aec.ref.push_queued(item, off, len(seg), sr, gains))
        self._mic = self._injected_mic or PyAudioMic(self.cfg, self.webrtc.frame_samples)
        self._mic.open()
        self._mic.on_status = self._on_mic_status

        self._stt_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="SpecSTT")
        self.running.set()
        self.last_activity = time.monotonic()
        for target, name in ((self._audio_thread, "AudioThread"), (self._processing_thread, "ProcessingThread")):
            th = threading.Thread(target=target, name=name, daemon=True)
            th.start()
            self._threads.append(th)
        if self.aec is not None and self.cfg.aec_calibrate_on_start:
            self.speaker.enqueue(AudioItem("calib", make_calibration(self.cfg.sample_rate), self.cfg.sample_rate,
                                           label="aec_calibration"))
            self.speaker.wait_idle(5.0)
            time.sleep(0.3)
        self._earcon("wake")
        trace.emit("ready")
        log.info("ready — say '%s'", self.cfg.wakeword.replace("_", " "))

    def stop(self) -> None:
        self.running.clear()
        if self.speaker:
            self.speaker.shutdown()
        for th in self._threads:
            th.join(timeout=3.0)
        if self._mic:
            self._mic.close()
        if self._stt_pool:
            self._stt_pool.shutdown(wait=False, cancel_futures=True)
        trace.emit("stopped")

    def print_summary(self) -> None:
        print("\n── Metrics Summary ──────────────────────────────────────")
        print(f"  Turns completed        : {self._m_turns}")
        print(f"  Replaced utterances    : {self._m_dropped}")
        print(f"  Blank STT results      : {self._m_blank_stt}")
        print(f"  Rejected captures      : {self._m_false_wakes}")
        print(f"  Interrupts detected    : {self._m_interrupt_detected}")
        print(f"  Interrupted turns      : {self._m_interrupt_success}")
        print(f"  Server restart attempts: {self._m_server_restarts}")
        print("─────────────────────────────────────────────────────────\n")

    def run(self) -> None:
        self._validate_microphone_early()
        self.start()
        signal.signal(signal.SIGTERM, lambda *_: self.running.clear())
        try:
            while self.running.is_set() and self._threads[0].is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping pipeline...")
        finally:
            self.stop()
            if self.cfg.metrics_exit_summary:
                self.print_summary()
            print("Stopped.")

    def _validate_microphone_early(self) -> None:
        if self._injected_mic is not None:
            return
        import pyaudio
        from audio.devices import MicError, resolve_from_cfg
        pa = pyaudio.PyAudio()
        try:
            try:
                dev = resolve_from_cfg(pa, self.cfg)
                s = pa.open(format=pyaudio.paInt16, channels=1, rate=self.cfg.sample_rate, input=True,
                            input_device_index=dev.index, frames_per_buffer=256)
                s.close()
            except (MicError, OSError) as e:
                print(f"[{stamp()}] Audio error: {e}\n  List devices: python -m audio.devices   "
                      f"Select: MIC_NAME=<substring> or MIC_INDEX=<n>")
                sys.exit(1)
        finally:
            pa.terminate()


if __name__ == "__main__":
    Pipeline().run()
