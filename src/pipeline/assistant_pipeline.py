"""
assistant_pipeline.py — threaded voice assistant pipeline.

Architecture:
  Thread 1 (_audio_thread)      — mic read, wake word, Silero, interrupt detection
  Thread 2 (_processing_thread) — STT → LLM → TTS, cancellable between sentences

Shared state:
  utterance_queue   Queue(1)  — T1 puts utterance samples, T2 gets them
  interrupt_queue   Queue(1)  — T1 puts interrupt samples, T2 gets them
  cancel_event      Event     — T1 sets when interrupt confirmed; T2 checks between sentences
  is_speaking       Event     — T2 sets during TTS; T1 enables interrupt detection
  processing_busy   Event     — T2 sets during STT/LLM/TTS for a normal turn
  conversation_mode Event     — set after wake word; cleared after 45s silence
"""

import io
import re
import threading
import queue
import time
import numpy as np
from collections import deque
from pathlib import Path
from wave import open as wave_open

import pyaudio
import soundfile as sf

from common import stamp
from common.config import load_config
from common.servers import SERVERS, _ping, _start, wait_for_servers
from llm.gemma import GemmaLLM
from stt.whisper_cpp import WhisperCppSTT
from tts.kokoro import KokoroTTS
from tts.phrases import PhrasePlayer
from vad.silero import SileroGate
from vad.webrtc import WebRTCGate
from wakeword.listen import WakeWordListener

# Whisper tokens that mean "nothing was said" — never send these to the LLM.
BLANK_TOKENS = {"[BLANK_AUDIO]", "[SILENCE]", "(silence)", "(ambient noise)"}


class Pipeline:
    def __init__(self):
        # Single shared config — every stage reads from this.
        self.cfg = load_config()

        # VAD / wake word components.
        self.webrtc = WebRTCGate()
        self.wake   = WakeWordListener()
        self.silero = SileroGate()

        # AI stage clients (HTTP only — no model loading here).
        self.stt = WhisperCppSTT(self.cfg)
        self.llm = GemmaLLM(self.cfg)
        self.tts = KokoroTTS(self.cfg)

        # System phrase player — synthesized once at startup, reused from disk.
        self.phrases = PhrasePlayer(self.cfg)

        # Rolling conversation history.
        # maxlen * 2 because each turn = 2 messages (user + assistant).
        self.history: deque = deque(maxlen=self.cfg.memory_turns * 2)

        # ── Shared threading primitives ───────────────────────────────────────

        # T1 → T2: audio samples for a complete utterance.
        # maxsize=1: if T2 is busy, T1 drops and plays "please wait".
        self.utterance_queue: queue.Queue = queue.Queue(maxsize=1)

        # T1 → T2: audio samples for an interrupt utterance.
        self.interrupt_queue: queue.Queue = queue.Queue(maxsize=1)

        # T1 → T2: set when interrupt is confirmed. T2 checks between sentences.
        self.cancel_event = threading.Event()

        # T2 → T1: set while TTS is playing. Enables interrupt detection in T1.
        self.is_speaking = threading.Event()

        # T2 → T1: set while processing a normal turn.
        # Prevents T1 from starting a fresh capture too early.
        self.processing_busy = threading.Event()

        # Shared: set after wake word fires; cleared after conversation timeout.
        self.conversation_mode = threading.Event()

        # Hard cooldown after assistant/system audio so the assistant does not
        # hear itself and immediately start a fresh turn.
        self.cooldown_until = 0.0

        # Shared timestamp used by conversation timeout.
        self._last_tts_end_time = None

        # Clean shutdown signal for both threads.
        self.running = threading.Event()
        self.running.set()

        # ── Day 1 Metrics counters ────────────────────────────────────────────
        # All counters are for the lifetime of the pipeline process.
        # Thread-safety: only _audio_thread writes wake/drop counters;
        # only _processing_thread writes stt/llm/tts/interrupt counters.
        # No lock needed because each counter has a single writer.
        self._m_turns              = 0   # completed turn attempts (samples dequeued)
        self._m_dropped            = 0   # utterances dropped — queue full
        self._m_false_wakes        = 0   # wakes that produced no valid utterance
        self._m_blank_stt          = 0   # STT returned blank / noise token
        self._m_interrupt_detected = 0   # interrupt threshold crossed (T1 side)
        self._m_interrupt_success  = 0   # interrupt -> non-blank transcript used
        self._m_server_restarts    = 0   # server restart attempts

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def save_debug_wav(self, samples, prefix="utterance") -> None:
        if not self.cfg.debug_save_wav or not samples:
            return

        out_dir = Path(self.cfg.debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{prefix}_{int(time.time())}.wav"

        with wave_open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.cfg.sample_rate)
            wf.writeframes(np.concatenate(samples).astype(np.int16).tobytes())

        print(f"[{stamp()}] Debug WAV saved: {path}")

    # ── RMS energy helper ─────────────────────────────────────────────────────

    @staticmethod
    def _rms(frames: list) -> float:
        """
        Compute normalised RMS energy of a list of int16 numpy frames.
        Returns a value in [0.0, 1.0]. Used to filter low-energy noise.
        """
        if not frames:
            return 0.0
        combined = np.concatenate(frames).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(combined ** 2)))

    def _set_cooldown(self, seconds: float | None = None) -> None:
        """Ignore mic input for a short period after assistant/system audio."""
        self.cooldown_until = time.monotonic() + (seconds or self.cfg.assistant_audio_cooldown_s)

    # ------------------------------------------------------------------
    # Thread 1: Audio
    # ------------------------------------------------------------------
    def _audio_thread(self, stream) -> None:
        """
        Continuously reads mic frames.

        Pre-wake: WebRTC gate → openWakeWord detection.
        Post-wake: Silero utterance capture → enqueue for processing.
        Interrupt: While T2 is speaking, monitor for sustained speech with
        energy above threshold → signal T2 to cancel.
        Conversation timeout: silence after TTS → back to idle.
        """

        # Per-utterance state.
        after_wake = False
        silence_frames = 0
        command_start = None
        silero_buf = np.zeros(0, dtype=np.int16)
        recording = []
        current_capture_from_conversation = False

        # Pre-roll: capture audio just before wake word so the first syllable
        # of the command is not clipped.
        pre_roll_frames = max(
            1,
            int((self.cfg.utterance_pre_roll_ms / 1000) * self.cfg.sample_rate / self.webrtc.frame_samples),
        )
        pre_roll = deque(maxlen=pre_roll_frames)

        # Post-roll: keep a little tail after final silence.
        post_roll_frames = max(
            1,
            int((self.cfg.utterance_post_roll_ms / 1000) * self.cfg.sample_rate / self.webrtc.frame_samples),
        )
        post_roll_queue = deque(maxlen=post_roll_frames)

        # Conversation-mode re-entry gating.
        conversation_reentry_hits = 0

        # Interrupt detection state.
        interrupt_speech_ms = 0.0
        interrupt_recording = []
        interrupt_silero = SileroGate()
        interrupt_silero_buf = np.zeros(0, dtype=np.int16)
        interrupt_active = False

        # Conversation timeout state.
        warning_played = False

        def reset_utterance_state() -> None:
            nonlocal after_wake, silence_frames, command_start
            nonlocal silero_buf, recording, current_capture_from_conversation
            after_wake = False
            silence_frames = 0
            command_start = None
            current_capture_from_conversation = False
            self.silero.state = None
            self.silero.started_at = None
            self.silero.history = []
            silero_buf = np.zeros(0, dtype=np.int16)
            recording = []
            post_roll_queue.clear()

        def reset_interrupt_state() -> None:
            nonlocal interrupt_speech_ms, interrupt_recording
            nonlocal interrupt_silero_buf, interrupt_active
            interrupt_speech_ms = 0.0
            interrupt_recording = []
            interrupt_silero_buf = np.zeros(0, dtype=np.int16)
            interrupt_active = False
            interrupt_silero.state = None
            interrupt_silero.started_at = None
            interrupt_silero.history = []

        frame_duration_ms = (self.webrtc.frame_samples / self.cfg.sample_rate) * 1000

        while self.running.is_set():
            pcm   = stream.read(self.webrtc.frame_samples, exception_on_overflow=False)
            t     = time.monotonic()
            frame = np.frombuffer(pcm, dtype=np.int16)

            # Keep rolling pre-roll while idle.
            if not after_wake:
                pre_roll.append(frame)

            # Hard cooldown after assistant/system speech.
            if t < self.cooldown_until:
                continue

            # Conversation timeout only when truly idle.
            if (
                self.conversation_mode.is_set()
                and not self.is_speaking.is_set()
                and not self.processing_busy.is_set()
                and not after_wake
                and self._last_tts_end_time is not None
            ):
                silent_for = t - self._last_tts_end_time
                timeout = self.cfg.conversation_timeout_s
                warning_at = timeout - self.cfg.conversation_warning_s

                if not warning_played and silent_for >= warning_at:
                    self.phrases.play("going_to_sleep")
                    self._set_cooldown()
                    warning_played = True

                if silent_for >= timeout:
                    self.phrases.play("goodbye")
                    self._set_cooldown()
                    self.conversation_mode.clear()
                    self._last_tts_end_time = None
                    warning_played = False
                    reset_utterance_state()
                    print(f"[{stamp()}] Conversation timeout — returning to idle")
                    continue

            # Interrupt detection while assistant is speaking.
            if self.is_speaking.is_set() and self.conversation_mode.is_set():
                # append frame for interrupt detection
                interrupt_recording.append(frame)
                # Silero gate on the interrupt buffer to detect sustained speech.
                interrupt_silero_buf = np.concatenate([interrupt_silero_buf, frame])

                # while we have enough audio for Silero, check for sustained speech in the interrupt buffer
                while len(interrupt_silero_buf) >= interrupt_silero.chunk_size:
                    # use Silero to detect if there's sustained speech in the interrupt buffer
                    chunk = interrupt_silero_buf[:interrupt_silero.chunk_size]
                    # remove the chunk from the buffer so the next iteration checks the next chunk
                    interrupt_silero_buf = interrupt_silero_buf[interrupt_silero.chunk_size:]
                    # predict speech probability with Silero on the chunk
                    prob = interrupt_silero.predict(chunk.tobytes())
                    # update Silero state and get the new state
                    _, new_state, _, _, _, _ = interrupt_silero.update(prob)

                    # if Silero detects speech
                    if new_state == "speech":
                        # increment the interrupt speech duration with the chunk duration
                        interrupt_speech_ms += frame_duration_ms
                    # if Silero detects silence
                    else:
                        # Speech must be sustained, not sporadic.
                        # Reset the interrupt speech duration and recording
                        interrupt_speech_ms = 0.0
                        interrupt_recording = []

                    # Fire interrupt only if:
                    #   1. Sustained speech for interrupt_min_speech_ms
                    #   2. RMS energy above threshold (filters hiss/breath noise)
                    if interrupt_speech_ms >= self.cfg.interrupt_min_speech_ms:
                        rms = self._rms(interrupt_recording)
                        if rms >= self.cfg.interrupt_energy_threshold and not interrupt_active:
                            interrupt_active = True
                            print(f"[{stamp()}] Interrupt detected (rms={rms:.4f})")
                            try:
                                self.interrupt_queue.put_nowait(list(interrupt_recording))
                                print(f"[{stamp()}] Interrupt audio enqueued ({len(interrupt_recording) / self.webrtc.sample_rate:.2f}) seconds)")
                            except queue.Full:
                                pass
                            # Count detection here (T1 side — single writer for this counter).
                            self._m_interrupt_detected += 1
                            self.cancel_event.set()
                            self._set_cooldown()
                            reset_interrupt_state()
                        elif rms < self.cfg.interrupt_energy_threshold:
                            reset_interrupt_state()

                continue

            # Reset interrupt state when not speaking.
            reset_interrupt_state()

            # Pre-wake path.
            if not after_wake and not self.conversation_mode.is_set():
                is_speech = self.webrtc.vad.is_speech(pcm, self.webrtc.sample_rate)
                new_webrtc_state = "speech" if is_speech else "silence"

                if self.webrtc.state is None:
                    self.webrtc.state = new_webrtc_state
                    self.webrtc.started_at = t
                    print(f"[{stamp()}] WebRTC {new_webrtc_state} (start)")
                elif new_webrtc_state != self.webrtc.state:
                    old = self.webrtc.state
                    print(
                        f"[{stamp()}] WebRTC {old} -> {new_webrtc_state} "
                        f"after {t - self.webrtc.started_at:.2f}s"
                    )
                    self.webrtc.state = new_webrtc_state
                    self.webrtc.started_at = t

                if self.webrtc.state != "speech":
                    continue

                score = self.wake.model.predict(frame).get(self.cfg.wakeword, 0.0)
                self.wake.hits = (self.wake.hits + 1) if score >= self.wake.threshold else 0

                if self.wake.hits >= self.wake.trigger_level:
                    self.wake.hits = 0
                    after_wake = True
                    current_capture_from_conversation = False
                    command_start = t
                    silero_buf = np.zeros(0, dtype=np.int16)
                    silence_frames = 0
                    self.silero.state = None
                    self.silero.started_at = None
                    self.silero.history = []
                    recording = list(pre_roll)
                    post_roll_queue = deque(maxlen=post_roll_frames)
                    self.conversation_mode.set()
                    warning_played = False
                    print(f"[{stamp()}] WakeWord detected: {self.cfg.wakeword} score={score:.3f}")
                    self.phrases.play("listening")
                    self._set_cooldown()
                    continue

            # Conversation-mode re-entry.
            if not after_wake and self.conversation_mode.is_set():
                # Prevent immediate re-trigger if we're still processing or speaking the previous turn.
                if self.processing_busy.is_set() or self.is_speaking.is_set():
                    continue

                # Require sustained speech with WebRTC to re-enter, to avoid noise-triggered false wakes.
                is_speech = self.webrtc.vad.is_speech(pcm, self.webrtc.sample_rate)
                if not is_speech:
                    conversation_reentry_hits = 0
                    continue

                # Note: we intentionally do not use Silero for conversation re-entry gating, to keep it 
                # more responsive and less likely to get stuck in silence if Silero misses the start of speech.
                conversation_reentry_hits += 1
                if conversation_reentry_hits < self.cfg.conversation_reentry_start_hits:
                    continue

                after_wake = True
                current_capture_from_conversation = True
                command_start = t
                silero_buf = np.zeros(0, dtype=np.int16)
                silence_frames = 0
                self.silero.state = None
                self.silero.started_at = None
                self.silero.history = []
                recording = list(pre_roll)
                post_roll_queue = deque(maxlen=post_roll_frames)
                conversation_reentry_hits = 0

            if not after_wake:
                continue

            recording.append(frame)
            post_roll_queue.append(frame)
            silero_buf = np.concatenate([silero_buf, frame])

            while len(silero_buf) >= self.silero.chunk_size:
                chunk      = silero_buf[:self.silero.chunk_size]
                silero_buf = silero_buf[self.silero.chunk_size:]
                prob       = self.silero.predict(chunk.tobytes())
                changed, new_state, old_state, avg, started, t2 = self.silero.update(prob)

                if changed:
                    if old_state is None:
                        print(f"[{stamp()}] Silero {new_state} (start) avg={avg:.3f}")
                    else:
                        print(
                            f"[{stamp()}] Silero {old_state} -> {new_state} "
                            f"after {t2 - started:.2f}s avg={avg:.3f}"
                        )

                silence_frames = (
                    silence_frames + 1 if self.silero.state == "silence" else 0
                )

                command_age_ms = (t - command_start) * 1000 if command_start else 0
                enough_silence = silence_frames >= self.cfg.silero_stop_silence_frames
                enough_time    = command_age_ms >= self.cfg.utterance_min_ms
                past_floor     = command_age_ms >= self.cfg.utterance_floor_ms
                very_silent    = avg < self.cfg.silero_early_exit_threshold

                # Early exit is allowed only for the first wake-triggered utterance,
                # not for conversation-mode re-entry captures.
                early_exit = (
                    past_floor
                    and very_silent
                    and enough_silence
                    and not current_capture_from_conversation
                )

                if (enough_time and enough_silence) or early_exit:
                    if early_exit and not enough_time:
                        print(f"[{stamp()}] Early exit at {command_age_ms:.0f}ms (avg={avg:.3f})")
                    else:
                        print(f"[{stamp()}] Utterance ended")

                    samples = recording + list(post_roll_queue)
                    self.save_debug_wav(samples)

                    # Tiny, low-energy captures are almost always noise.
                    # Drop them quietly instead of saying "please wait".
                    capture_rms = self._rms(samples)
                    if command_age_ms < self.cfg.utterance_reject_ms or capture_rms < self.cfg.utterance_reject_rms:
                        print(
                            f"[{stamp()}] REJECTED Utterance"
                            f"(age={command_age_ms:.0f}ms rms={capture_rms:.4f})"
                        )
                        # A wake fired but the capture was too short/quiet — count as false wake.
                        self._m_false_wakes += 1
                        reset_utterance_state()
                        break
                    else:
                        print(
                            f"[{stamp()}] ACCEPTED Utterance"
                            f"(age={command_age_ms:.0f}ms rms={capture_rms:.4f})"
                        )

                    try:
                        self.utterance_queue.put_nowait(samples)
                    except queue.Full:
                        # T2 still processing previous turn — this utterance is lost.
                        self._m_dropped += 1
                        print(f"[{stamp()}] Utterance dropped — T2 still busy")
                        self.phrases.play("please_wait")
                        self._set_cooldown()

                    reset_utterance_state()
                    break

    # ------------------------------------------------------------------
    # Thread 2: Processing
    # ------------------------------------------------------------------
    def _processing_thread(self) -> None:
        """
        Waits for utterances from T1, runs STT → LLM → TTS.
        Checks cancel_event between TTS sentences.
        On cancel: stops TTS, picks up interrupt from interrupt_queue,
        builds combined transcript, reruns LLM.
        """

        def _transcribe_samples(samples) -> str | None:
            """
            Run STT. Returns clean transcript string, or None if blank/noise.
            Blank audio is ignored quietly to avoid a speech loop.
            """
            try:
                transcript = self.stt.transcribe(samples, self.cfg.sample_rate)
            except Exception as e:
                print(f"[{stamp()}] STT error: {e}")
                return None

            print(f"[{stamp()}] Transcript: {transcript}")

            if not transcript or not transcript.strip() or transcript.strip() in BLANK_TOKENS:
                return None

            return transcript.strip()

        def _run_llm(user_text: str) -> str | None:
            """
            Play 'thinking', call LLM with history.
            Returns response or None."""
            self.phrases.play("thinking")
            self._set_cooldown()
            try:
                response = self.llm.generate(
                    user_text,
                    history=list(self.history)[:-1],
                )
                print(f"[{stamp()}] LLM: {response}")
                return response
            except Exception as e:
                print(f"[{stamp()}] LLM error: {e}")
                return None

        def _speak_with_cancel(response: str) -> bool:
            """
            Play response sentence by sentence.
            Checks cancel_event before each sentence and between playback chunks.
            Returns True if playback completed, False if cancelled.
            """
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response) if s.strip()]
            if not sentences:
                sentences = [response]

            pa = pyaudio.PyAudio()
            stream = None

            try:
                for i, sentence in enumerate(sentences):
                    # Stop before synthesis if interrupt already happened.
                    if self.cancel_event.is_set():
                        print(f"[{stamp()}] TTS cancelled before sentence: {sentence[:40]}")
                        return False

                    wav = self.tts._synthesize(sentence)
                    if wav is None:
                        return False

                    # Stop if interrupt happened during synth HTTP call.
                    if self.cancel_event.is_set():
                        print(f"[{stamp()}] TTS cancelled after synthesis: {sentence[:40]}")
                        return False

                    buf = io.BytesIO(wav)
                    audio_data, sample_rate = sf.read(buf, dtype="float32")

                    # Handle mono/stereo safely.
                    if getattr(audio_data, "ndim", 1) > 1:
                        channels = audio_data.shape[1]
                    else:
                        channels = 1

                    pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)

                    # Open stream once, reuse for all sentences.
                    if stream is None:
                        stream = pa.open(
                            format=pyaudio.paInt16,
                            channels=channels,
                            rate=sample_rate,
                            output=True,
                            frames_per_buffer=1024,
                        )

                    chunk_size = 1024
                    offset = 0

                    while offset < len(pcm):
                        if self.cancel_event.is_set():
                            print(f"[{stamp()}] TTS cut mid-sentence")
                            stream.stop_stream()
                            return False

                        end = min(offset + chunk_size, len(pcm))
                        chunk = pcm[offset:end]
                        stream.write(chunk.tobytes())
                        offset = end

                    print(f"[{stamp()}] TTS streamed: {sentence[:60]}...")

                    # Small natural pause between sentences.
                    if i < len(sentences) - 1:
                        time.sleep(0.05)

                return True

            finally:
                if stream is not None:
                    try:
                        # On normal completion, do not call stop_stream().
                        # Close lets the playback drain cleanly.
                        stream.close()
                    except Exception as e:
                        print(f"[{stamp()}] TTS stream close error: {e}")

                try:
                    pa.terminate()
                except Exception as e:
                    print(f"[{stamp()}] TTS terminate error: {e}")

        while self.running.is_set():
            try:
                samples = self.utterance_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self.processing_busy.set()

            # ── Turn start timestamp ──────────────────────────────────────────
            # Monotonic clock: unaffected by system clock adjustments.
            _turn_start = time.monotonic()
            self._m_turns += 1

            # ── STT ───────────────────────────────────────────────────────────
            _stt_t0 = time.monotonic()
            transcript = _transcribe_samples(samples)
            _stt_ms = (time.monotonic() - _stt_t0) * 1000.0

            if transcript is None:
                # Blank STT result — count it and skip this turn.
                self._m_blank_stt += 1
                self.processing_busy.clear()
                continue

            self.history.append({"role": "user", "content": transcript})

            # ── LLM ───────────────────────────────────────────────────────────
            _llm_t0 = time.monotonic()
            response = _run_llm(transcript)
            _llm_ms = (time.monotonic() - _llm_t0) * 1000.0

            if response is None:
                self.processing_busy.clear()
                self._handle_server_error()
                continue

            self.history.append(
                {
                    "role": "assistant",
                    "content": response[: self.cfg.memory_assistant_max_chars],
                }
            )

            self.cancel_event.clear()
            self.processing_busy.clear()
            self.is_speaking.set()

            # ── TTS ───────────────────────────────────────────────────────────
            _tts_t0 = time.monotonic()
            completed = _speak_with_cancel(response)
            _tts_ms = (time.monotonic() - _tts_t0) * 1000.0

            self.is_speaking.clear()

            # ── Per-turn metrics log line ─────────────────────────────────────
            # Printed after every turn so the terminal becomes the baseline log.
            # Format matches docs/metrics.md so manual test recordings stay consistent.
            _total_ms = (time.monotonic() - _turn_start) * 1000.0
            if self.cfg.metrics_enabled:
                print(
                    f"[{stamp()}] METRICS "
                    f"turn={self._m_turns} "
                    f"stt={_stt_ms:.0f}ms "
                    f"llm={_llm_ms:.0f}ms "
                    f"tts={_tts_ms:.0f}ms "
                    f"total={_total_ms:.0f}ms "
                    f"dropped={self._m_dropped} "
                    f"blank_stt={self._m_blank_stt} "
                    f"false_wakes={self._m_false_wakes} "
                    f"interrupts_ok={self._m_interrupt_success}"
                )

            if not completed:
                # ── Interrupt path ────────────────────────────────────────────
                self.cancel_event.clear()

                # Drain any stale mic frames that built up during TTS.
                # (Handled by T1 discarding frames while is_speaking was set —
                # but clear interrupt_queue of any double-fire just in case.)
                try:
                    interrupt_samples = self.interrupt_queue.get_nowait()
                except queue.Empty:
                    # Interrupt signal arrived but no audio — treat as fresh
                    # listen cycle.
                    print(f"[{stamp()}] Interrupt with no audio — resuming listen")
                    self._set_cooldown()
                    self._last_tts_end_time = time.monotonic()
                    continue

                self.processing_busy.set()
                interrupt_transcript = _transcribe_samples(interrupt_samples)
                if interrupt_transcript is None:
                    # Interrupt was noise — continue conversation normally.
                    print(f"[{stamp()}] Interrupt transcribed as blank — continuing")
                    self.processing_busy.clear()
                    self._set_cooldown()
                    self._last_tts_end_time = time.monotonic()
                    continue

                # Interrupt produced a usable transcript — count as success.
                self._m_interrupt_success += 1

                # Build combined transcript: original + spoken correction.
                # "Actually, ..." is a strong correction signal for small LLMs.
                combined = f"{transcript}. Actually, {interrupt_transcript}"
                print(f"[{stamp()}] Combined transcript: {combined}")

                # Update history: replace the last user entry with combined.
                # Pop assistant turn (last) and user turn (second to last),
                # then re-add combined user turn. LLM gets the full context.
                if len(self.history) >= 2:
                    self.history.pop()  # remove assistant response
                    self.history.pop()  # remove original user message
                    self.history.append({"role": "user", "content": combined})

                response = _run_llm(combined)
                if response is None:
                    self.processing_busy.clear()
                    self._handle_server_error()
                    continue

                self.history.append(
                    {
                        "role": "assistant",
                        "content": response[: self.cfg.memory_assistant_max_chars],
                    }
                )

                self.cancel_event.clear()
                self.processing_busy.clear()
                self.is_speaking.set()
                _speak_with_cancel(response)
                self.is_speaking.clear()
                self.cancel_event.clear()

            # Record when TTS last finished — drives conversation timeout.
            self._last_tts_end_time = time.monotonic()
            self._set_cooldown()

    # ------------------------------------------------------------------
    # Server error recovery
    # ------------------------------------------------------------------
    def _handle_server_error(self) -> None:
        """Check all servers, restart any that are down, play fallback phrase."""
        for srv in SERVERS:
            if not _ping(srv["url"]):
                print(f"[{stamp()}] {srv['name']} down — restarting...")
                # Count every restart attempt for the metrics exit summary.
                self._m_server_restarts += 1
                _start(srv)
                time.sleep(3)

        try:
            self.tts.speak(self.cfg.fallback_phrase)
            self._set_cooldown()
        except Exception as e:
            print(f"[{stamp()}] Fallback TTS failed: {e}")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        print(
            "Pipeline: WebRTC -> openWakeWord -> Silero -> "
            "Whisper[8081] -> Gemma[8080] -> Kokoro[8082]"
        )
        print(f"wakeword={self.cfg.wakeword} sample_rate={self.cfg.sample_rate}")
        print(
            f"memory_turns={self.cfg.memory_turns} "
            f"utterance_floor_ms={self.cfg.utterance_floor_ms}"
        )
        print(
            f"conversation_timeout={self.cfg.conversation_timeout_s}s "
            f"interrupt_min_speech={self.cfg.interrupt_min_speech_ms}ms"
        )

        wait_for_servers()
        self.phrases.warm_up()

        print("Press Ctrl+C to stop.")

        audio, stream = self.webrtc.open()

        t1 = threading.Thread(
            target=self._audio_thread,
            args=(stream,),
            name="AudioThread",
            daemon=True,
        )
        t2 = threading.Thread(
            target=self._processing_thread,
            name="ProcessingThread",
            daemon=True,
        )

        t1.start()
        t2.start()

        try:
            # Main thread just keeps the process alive and handles Ctrl+C.
            while t1.is_alive():
                t1.join(timeout=1.0)
        except KeyboardInterrupt:
            print("\nStopping pipeline...")
            self.running.clear()
        finally:
            self.running.clear()
            t1.join(timeout=3.0)
            t2.join(timeout=3.0)
            stream.stop_stream()
            stream.close()
            audio.terminate()

            # ── Exit summary ──────────────────────────────────────────────────
            # Printed once on graceful shutdown (Ctrl+C or clean exit).
            # Record these numbers as your Week 1 baseline in docs/metrics.md.
            if self.cfg.metrics_exit_summary:
                print("\n── Metrics Summary ──────────────────────────────────────")
                print(f"  Turns completed        : {self._m_turns}")
                print(f"  Dropped utterances     : {self._m_dropped}")
                print(f"  Blank STT results      : {self._m_blank_stt}")
                print(f"  False wakes            : {self._m_false_wakes}")
                print(f"  Interrupts detected    : {self._m_interrupt_detected}")
                print(f"  Interrupts succeeded   : {self._m_interrupt_success}")
                print(f"  Server restart attempts: {self._m_server_restarts}")
                print("─────────────────────────────────────────────────────────\n")

            print("Stopped.")


if __name__ == "__main__":
    Pipeline().run()
