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

import sys
import io
import re
import threading
import queue
import time
import datetime
import numpy as np
from collections import deque
import json
from pathlib import Path
from wave import open as wave_open

import pyaudio
import soundfile as sf

from common import stamp, _LatencyTracer
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
BLANK_TOKENS = {"[BLANK_AUDIO]", "[SILENCE]", "[MUSIC]", "(silence)", "(ambient noise)",
                "(birds chirping)", "(chuckles)", "(speaking in foreign language)",
                "(upbeat music)", "(crowd chattering)", "(humming)"}


class Pipeline:
    def __init__(self):
        # Single shared config — every stage reads from this.
        self.cfg = load_config()

        self._tracer = _LatencyTracer()
        self._last_response: str = ""   # for __REPEAT__ fast path

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
        self._m_turns              = 0     # completed turn attempts (samples dequeued)
        self._m_dropped            = 0     # utterances dropped — queue full
        self._m_false_wakes        = 0     # wakes that produced no valid utterance
        self._m_blank_stt          = 0     # STT returned blank / noise token
        self._m_interrupt_detected = 0     # interrupt threshold crossed (T1 side)
        self._m_interrupt_success  = 0     # interrupt -> non-blank transcript used
        self._m_server_restarts    = 0     # server restart attempts
        self._last_debug_wav       = None  # reset each turn; set by save_debug_wav

        self._FP = [
            (re.compile(r'\b(what(?:\'s| is) the time|current time|what time is it)\b', re.I),
            lambda: f"It's {datetime.now().strftime('%I:%M %p').lstrip('0')}"),
            (re.compile(r'\b(today(?:\'s)? date|what(?:\'s| is) the date)\b', re.I),
            lambda: f"Today is {datetime.now().strftime('%A, %B %d')}"),
            (re.compile(r'\b(stop|cancel|nevermind|quit|exit|shut ?up)\b', re.I),
            lambda: "__STOP__"),
            (re.compile(r'\b(hello|hi there|hey there)\b', re.I),
            lambda: "Hello! How can I help?"),
            (re.compile(r'\b(thank(?:s| you))\b', re.I),
            lambda: "You're welcome!"),
            (re.compile(r'\b(repeat(?: that)?|say that again)\b', re.I),
            lambda: "__REPEAT__"),
        ]

        ## Pre-synthesise earcons at startup for instant playback when needed.
        sr = self.cfg.sample_rate

        # WAKE DETECTED — ascending two-tone (mirror of interrupt)
        # 660 Hz → 880 Hz, 90 ms each: "something woke up, going up"
        self.wake_earcon_pcm = self._make_earcon(
            [(660, 0.09), (880, 0.09)], sr
        )

        # LISTENING STARTED — single short high pip
        # 1046 Hz (C6), 70 ms: crisp, attention-grabbing "ready now"
        self.listening_earcon_pcm = self._make_earcon(
            [(1046, 0.07)], sr, amplitude=10000
        )

        # THINKING — two soft rising pulses, slower
        # 523 Hz → 659 Hz → 784 Hz, 100 ms each: "working on it"
        self.thinking_earcon_pcm = self._make_earcon(
            [(523, 0.10), (659, 0.10), (784, 0.10)], sr, amplitude=9000
        )

        # INTERRUPT — descending two-tone, quick and attention-grabbing but not too harsh.
        # 880 Hz → 660 Hz, 90 ms each: "something interrupted me, going down"
        self.interrupt_earcon_pcm = self._make_earcon(
            [(880, 0.09), (660, 0.09)], sr   # descending — barge-in registered
        )

        # ERROR — descending tritone, slightly longer
        # 440 Hz → 311 Hz → 220 Hz, 100 ms each: "something went wrong"
        self.error_earcon_pcm = self._make_earcon(
            [(440, 0.10), (311, 0.10), (220, 0.10)], sr, amplitude=11000
        )

        # SLEEP / GOODBYE — slow falling three-tone
        # 880 Hz → 523 Hz → 330 Hz, 120 ms each: "going away"
        self.sleep_earcon_pcm = self._make_earcon(
            [(880, 0.12), (523, 0.12), (330, 0.12)], sr, amplitude=11000
        )

    # ------------------------------------------------------------------
    # Startup validation helpers
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        """
        Validate critical config: model paths, server URLs, ports.
        Exit fast on misconfiguration so the user gets a clear error.
        """

        # Validate server URLs / ports from SERVERS table.
        # This catches obvious port conflicts / bad URLs at startup.
        seen_ports = set()
        for srv in SERVERS:
            url = srv.get("url", "")
            name = srv.get("name", "unknown")
            if "://" not in url:
                print(f"[{stamp()}] Config error: invalid URL for {name}: {url}")
                sys.exit(1)

            # Very small parser: assume http://host:port
            try:
                host_port = url.split("://", 1)[1].split("/", 1)[0]
                host, port_str = host_port.rsplit(":", 1)
                port = int(port_str)
            except Exception:
                print(f"[{stamp()}] Config error: cannot parse port from {name} URL: {url}")
                sys.exit(1)

            if port in seen_ports:
                print(f"[{stamp()}] Config error: duplicate port {port} in SERVERS")
                sys.exit(1)
            seen_ports.add(port)

    def _validate_microphone(self) -> None:
        """
        Ensure the configured input device and sample rate are available
        before starting audio / processing threads.
        """
        pa = pyaudio.PyAudio()
        try:
            device_index = getattr(self.cfg, "input_device_index", None)
            sample_rate = int(self.cfg.sample_rate)

            # If user pinned a device index, ensure it exists.
            if device_index is not None:
                try:
                    info = pa.get_device_info_by_index(device_index)
                except Exception as e:
                    print(f"[{stamp()}] Audio error: input_device_index {device_index} invalid: {e}")
                    sys.exit(1)
                if info.get("maxInputChannels", 0) <= 0:
                    print(f"[{stamp()}] Audio error: device {device_index} has no input channels")
                    sys.exit(1)

            # Try opening a tiny test stream to confirm sample-rate assumptions.
            try:
                test_stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=256,
                )
                test_stream.close()
            except Exception as e:
                print(
                    f"[{stamp()}] Audio error: cannot open input stream at "
                    f"{sample_rate} Hz (device={device_index}): {e}"
                )
                sys.exit(1)

            print(
                f"[{stamp()}] Audio OK: device={device_index} sample_rate={sample_rate}"
            )
        finally:
            pa.terminate()

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

        # Store path so _log_turn can reference it for this turn
        self._last_debug_wav = path
    
    def _log_turn(self, turn_data: dict) -> None:
        """
        Append one JSON line per turn to a daily JSONL file.
        Each line is a self-contained record — easy to grep, stream, or replay.
        Fields: ts, turn, transcript, response_chars, stt_ms, llm_ms, tts_ms,
                total_ms, interrupted, failed, debug_wav
        """
        log_dir = Path(self.cfg.turnlogdir)
        log_dir.mkdir(parents=True, exist_ok=True)
        # One file per day so logs don't grow unbounded
        day = time.strftime("%Y-%m-%d")
        log_path = log_dir / f"turns_{day}.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(turn_data) + "\n")

    def _viz_log(
        self,
        msg: str,
        *,
        # These are optional — only passed from _audio_thread where values exist
        rms: float | None = None,
        silero_prob: float | None = None,
        webrtc_state: str | None = None,
        phase: str | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        """
        Print msg unconditionally (preserves all existing log output).
        When debug_mode is on, also print a structured ASCII viz line
        BEFORE the message so the event is clearly anchored to live stats.
        Thread-safe: print() holds the GIL per call, so lines don't interleave.
        """
        if self.cfg.debug_mode and any(
            v is not None for v in (rms, silero_prob, webrtc_state, phase, elapsed_ms)
        ):
            # --- RMS bar (20 chars wide, scaled 0→0.05 for voice range) ---
            bar_max = 0.05          # RMS above 0.05 is very loud speech
            rms_val = rms or 0.0
            filled = int(min(rms_val / bar_max, 1.0) * 20)
            rms_bar = "█" * filled + "░" * (20 - filled)

            # --- Silero bar (10 chars, 0→1 probability) ---
            sil_val = silero_prob or 0.0
            sil_filled = int(min(sil_val, 1.0) * 10)
            sil_bar = "▓" * sil_filled + "·" * (10 - sil_filled)

            # --- Compose the viz prefix line ---
            parts = []
            if elapsed_ms is not None:
                parts.append(f"T+{elapsed_ms:6.0f}ms")
            if rms is not None:
                parts.append(f"RMS[{rms_bar}]{rms_val:.4f}")
            if webrtc_state is not None:
                tag = "SPK" if webrtc_state == "speech" else "SIL"
                parts.append(f"WebRTC:{tag}")
            if silero_prob is not None:
                parts.append(f"Silero[{sil_bar}]{sil_val:.2f}")
            if phase is not None:
                parts.append(f"Phase:{phase}")

            print("  VIZ | " + " | ".join(parts))

        print(msg)

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

    @staticmethod
    def _is_breath_or_hiss(frames: list, sample_rate: int) -> bool:
        """True if audio looks like a breath or hiss rather than voiced speech.
        Uses spectral centroid > 4 kHz AND zero-crossing rate > 0.35.
        Both must fire to reject — avoids dropping sibilant speech like 'stop'."""
        if not frames:
            return False
        pcm = np.concatenate(frames).astype(np.float32) / 32768.0
        if len(pcm) < 64:
            return False
        # Zero-crossing rate
        zcr = float(np.mean(np.abs(np.diff(np.sign(pcm)))) / 2)
        # Spectral centroid
        fft_mag = np.abs(np.fft.rfft(pcm))
        freqs = np.fft.rfftfreq(len(pcm), d=1.0 / sample_rate)
        centroid = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-9))
        return centroid > 4000 and zcr > 0.35

    def _set_cooldown(self, seconds: float | None = None) -> None:
        """Ignore mic input for a short period after assistant/system audio."""
        self.cooldown_until = time.monotonic() + (seconds or self.cfg.assistant_audio_cooldown_s)

    @staticmethod
    def _make_earcon(
        tones: list[tuple[float, float]],  # [(freq_hz, duration_s), ...]
        samplerate: int,
        amplitude: int = 13000,            # out of 32767; 13000 ≈ 40%
        fade_ms: float = 5.0,              # ms for fade-in on first / fade-out on last tone
    ) -> np.ndarray:
        """
        Synthesise a multi-tone earcon from a list of (freq_hz, duration_s) pairs.
        Each tone is a pure sine wave. The first tone gets a fade-in and the last
        tone gets a fade-out (fade_ms each), so the earcon never clicks.
        Returns an int16 numpy array ready for PyAudio output.
        """
        segments = []
        fade_samples = int(samplerate * fade_ms / 1000)

        for i, (freq, dur) in enumerate(tones):
            n = int(samplerate * dur)
            t = np.linspace(0, dur, n, endpoint=False)
            wave = np.sin(2 * np.pi * freq * t)

            if i == 0 and fade_samples > 0:                         # fade-in on first
                wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
            if i == len(tones) - 1 and fade_samples > 0:            # fade-out on last
                wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)

            segments.append(wave)

        combined = np.concatenate(segments)
        return (combined * amplitude).astype(np.int16)
    
    def _play_earcon(self, pcm: np.ndarray) -> None:
        """Play a pre-baked earcon PCM array in a daemon thread (non-blocking)."""
        def _play():
            pa = pyaudio.PyAudio()
            st = pa.open(format=pyaudio.paInt16, channels=1,
                        rate=self.cfg.sample_rate, output=True)
            st.write(pcm.tobytes())
            st.stop_stream(); st.close(); pa.terminate()
        threading.Thread(target=_play, daemon=True).start()


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
                    self._play_earcon(self.sleep_earcon_pcm)
                    self.phrases.play("going_to_sleep")
                    self._set_cooldown()
                    warning_played = True

                if silent_for >= timeout:
                    self._play_earcon(self.sleep_earcon_pcm)
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
                        # interrupt_recording = []

                    # Fire interrupt only if:
                    #   1. Sustained speech for interrupt_min_speech_ms
                    #   2. RMS energy above threshold (filters hiss/breath noise)
                    if interrupt_speech_ms >= self.cfg.interrupt_min_speech_ms:
                        rms = self._rms(interrupt_recording)
                        if rms >= self.cfg.interrupt_energy_threshold and not interrupt_active \
                                and not (self.cfg.interrupt_hiss_filter \
                                    and self._is_breath_or_hiss(list(interrupt_recording), self.cfg.sample_rate) \
                                ):
                            interrupt_active = True
                            self._viz_log(
                                f"[{stamp()}] Interrupt detected (rms={rms:.4f})",
                                rms=rms,
                                phase="INTERRUPT",
                            )
                            try:
                                self.interrupt_queue.put_nowait(list(interrupt_recording))
                                self._viz_log(
                                    f"[{stamp()}] Interrupt audio enqueued ({len(interrupt_recording) / self.webrtc.sample_rate:.2f}) seconds)",
                                    rms=rms,
                                    phase="INTERRUPT",
                                )
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
                    self._viz_log(
                        f"[{stamp()}] WebRTC {new_webrtc_state} (start)",
                        webrtc_state=new_webrtc_state,
                        phase="PRE-WAKE",
                    )
                elif new_webrtc_state != self.webrtc.state:
                    old = self.webrtc.state
                    self._viz_log(
                        f"[{stamp()}] WebRTC {old} -> {new_webrtc_state} "
                        f"after {t - self.webrtc.started_at:.2f}s",
                        webrtc_state=new_webrtc_state,
                        phase="PRE-WAKE",
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
                    self._viz_log(
                        f"[{stamp()}] WakeWord detected: {self.cfg.wakeword} score={score:.3f}",
                        rms=self._rms([frame]),
                        phase="WAKE",
                    )
                    threading.Thread(
                        target=lambda: (
                            self._play_earcon(self.wake_earcon_pcm),
                            self.phrases.play("listening"),
                            self._set_cooldown()
                        ),
                        daemon=True
                    ).start()
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
                self._play_earcon(self.listening_earcon_pcm)

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
                    elapsed = (t - command_start) * 1000 if command_start else 0.0
                    if old_state is None:
                        self._viz_log(
                            f"[{stamp()}] Silero {new_state} (start) avg={avg:.3f}",
                            rms=self._rms([recording[-1]]) if recording else None,
                            silero_prob=prob,
                            webrtc_state=self.webrtc.state,
                            phase="LISTENING",
                            elapsed_ms=elapsed,
                        )
                    else:
                        self._viz_log(
                            f"[{stamp()}] Silero {old_state} -> {new_state} "
                            f"after {t2 - started:.2f}s avg={avg:.3f}",
                            rms=self._rms([recording[-1]]) if recording else None,
                            silero_prob=prob,
                            webrtc_state=self.webrtc.state,
                            phase="LISTENING",
                            elapsed_ms=elapsed,
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
                        self._viz_log(
                            f"[{stamp()}] Early exit at {command_age_ms:.0f}ms (avg={avg:.3f})",
                            rms=self._rms(recording) if recording else None,
                            silero_prob=avg,
                            phase="LISTENING",
                            elapsed_ms=command_age_ms,
                        )
                    else:
                        self._viz_log(
                            f"[{stamp()}] Utterance ended",
                            rms=self._rms(recording) if recording else None,
                            silero_prob=avg,
                            phase="LISTENING",
                            elapsed_ms=command_age_ms,
                        )

                    samples = recording + list(post_roll_queue)
                    self.save_debug_wav(samples)

                    # Tiny, low-energy captures are almost always noise.
                    # Drop them quietly instead of saying "please wait".
                    capture_rms = self._rms(samples)
                    if command_age_ms < self.cfg.utterance_reject_ms or capture_rms < self.cfg.utterance_reject_rms:
                        self._viz_log(
                            f"[{stamp()}] REJECTED Utterance"
                            f"(age={command_age_ms:.0f}ms rms={capture_rms:.4f})",
                            rms=capture_rms,
                            silero_prob=avg,
                            phase="LISTENING",
                            elapsed_ms=command_age_ms,
                        )
                        # A wake fired but the capture was too short/quiet — count as false wake.
                        self._m_false_wakes += 1
                        # Log a summary every N false wakes to help tune thresholds.
                        interval = self.cfg.false_wake_log_interval
                        if interval > 0 and self._m_false_wakes % interval == 0:
                            print(
                                f"[false-wake summary] total={self._m_false_wakes} "
                                f"| cooldown={self.cfg.assistant_audio_cooldown_s}s "
                                f"| reentry_hits={self.cfg.conversation_reentry_start_hits} "
                                f"| interrupt_min_speech={self.cfg.interrupt_min_speech_ms}ms"
                            )
                        reset_utterance_state()
                        break
                    else:
                        self._viz_log(
                            f"[{stamp()}] ACCEPTED Utterance"
                            f"(age={command_age_ms:.0f}ms rms={capture_rms:.4f})",
                            rms=capture_rms,
                            silero_prob=avg,
                            phase="LISTENING",
                            elapsed_ms=command_age_ms,
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
            threading.Thread(
                target=lambda: (
                    self._play_earcon(self.thinking_earcon_pcm),
                    self.phrases.play("thinking"),
                    self._set_cooldown()
                ),
                daemon=True
            ).start()
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
                        # TTS failed to synthesize this sentence.
                        try:
                            self._play_earcon(self.error_earcon_pcm)
                            self.phrases.play("fallback_tts")
                            self._set_cooldown()
                        except Exception as e:
                            print(f"[{stamp()}] TTS fallback phrase failed: {e}")
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
                    _FADE_CHUNKS = 4   # ~100 ms of fade at 1024 frames/16 kHz
                    _fade_count  = 0

                    while offset < len(pcm):
                        end   = min(offset + chunk_size, len(pcm))
                        chunk = pcm[offset:end].copy()

                        if self.cancel_event.is_set():
                            if _fade_count >= _FADE_CHUNKS:
                                stream.stop_stream()
                                print(f"[{stamp()}] TTS faded out mid-sentence")
                                return False
                            # Linear fade-out so the cut isn't a hard click.
                            fade = np.linspace(1.0, 0.0, len(chunk),
                                               dtype=np.float32)
                            chunk = (chunk.astype(np.float32) * fade
                                     ).astype(np.int16)
                            _fade_count += 1

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
                self._tracer.reset(); self._tracer.mark("dequeue")
            except queue.Empty:
                continue

            self.processing_busy.set()

            self._last_debug_wav = None   # reset at start of each new turn

            # ── Turn start timestamp ──────────────────────────────────────────
            # Monotonic clock: unaffected by system clock adjustments.
            _turn_start = time.monotonic()
            self._m_turns += 1

            # ── STT ───────────────────────────────────────────────────────────
            _stt_t0 = time.monotonic()
            transcript = _transcribe_samples(samples)
            self._tracer.mark("stt_done")
            _stt_ms = (time.monotonic() - _stt_t0) * 1000.0

            if transcript is None:
                # Blank or failed STT — count it, speak specific fallback.
                self._m_blank_stt += 1
                self.processing_busy.clear()
                try:
                    self._play_earcon(self.error_earcon_pcm)
                    self.phrases.play("fallback_stt")
                    self._set_cooldown()
                except Exception as e:
                    print(f"[{stamp()}] STT fallback phrase failed: {e}")
                continue

            fast_result = self._try_fast_path(transcript)
            self._tracer.mark("response_ready")
            _llm_ms = 0.0

            if fast_result == "__STOP__":
                self.conversation_mode.clear()
                self._last_tts_end_time = None
                self.processing_busy.clear()
                self.phrases.play("goodbye")
                self._set_cooldown()
                continue

            elif fast_result == "__REPEAT__":
                response = self._last_response or "I don't have anything to repeat."
                self.history.append({"role": "user", "content": transcript})

            elif fast_result is not None:
                response = fast_result
                self._last_response = response
                self.history.append({"role": "user", "content": transcript})
                self.history.append({"role": "assistant", "content": response})

            else:                                # Normal LLM path
                self._maybe_summarize_oldest()
                self.history.append({"role": "user", "content": transcript})
                _llm_t0 = time.monotonic()
                response = _run_llm(transcript)
                self._tracer.mark("response_ready")
                _llm_ms = (time.monotonic() - _llm_t0) * 1000.0
                if response is None:
                    self.processing_busy.clear()
                    self._play_earcon(self.error_earcon_pcm)
                    self._handle_server_error()
                    continue
                self.history.append({"role": "assistant", "content": self._trim_for_memory(response)})
                self._last_response = response

            self.cancel_event.clear()
            self.processing_busy.clear()
            self.is_speaking.set()

            # ── TTS ───────────────────────────────────────────────────────────
            _tts_t0 = time.monotonic()
            completed = _speak_with_cancel(response)
            self._tracer.mark("tts_done")
            if self.cfg.metrics_enabled:
                print(f"[LATENCY] {self._tracer.report()}")
            _tts_ms = (time.monotonic() - _tts_t0) * 1000.0

            self.is_speaking.clear()
            self.cancel_event.clear()
            # Apply full post-TTS cooldown so mic ignores speaker hardware drain.
            self._set_cooldown(self.cfg.assistant_audio_cooldown_s)
            self.last_tts_end_time = time.monotonic()

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
                # Collect all timing + outcome data for this turn into one record.
                # debug_wav is the path saved by save_debug_wav (may be None if disabled).
                self._log_turn({
                    "ts":            time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "turn":          self._m_turns,
                    "transcript":    transcript or "",
                    "response_chars": len(response) if response else 0,
                    "stt_ms":        round(_stt_ms, 1),
                    "llm_ms":        round(_llm_ms, 1),
                    "tts_ms":        round(_tts_ms, 1),
                    "total_ms":      round(_total_ms, 1),
                    "interrupted":   not completed,
                    "failed":        response is None,
                    "debug_wav":     str(self._last_debug_wav) if self._last_debug_wav else None,
                })

            if not completed:
                # ── Interrupt path ────────────────────────────────────────────
                self.cancel_event.clear()

                # Drain any stale mic frames that built up during TTS.
                # (Handled by T1 discarding frames while is_speaking was set —
                # but clear interrupt_queue of any double-fire just in case.)
                try:
                    interrupt_samples = self.interrupt_queue.get_nowait()
                    print(f"[{stamp()}] Interrupt audio dequeued ({len(interrupt_samples) / self.webrtc.sample_rate:.2f} seconds)")
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

                # Build combined transcript with context-aware joining.
                _STOP_RE       = re.compile(
                    r'^\s*(stop|cancel|nevermind|never mind|quit|shut up)\s*[.!?]?\s*$',
                    re.I)
                _CORRECTION_RE = re.compile(
                    r'\b(actually|no\b|wait|that\'?s wrong|incorrect|never mind)\b',
                    re.I)

                if _STOP_RE.match(interrupt_transcript):
                    # Pure stop: clear conversation, don't call LLM.
                    print(f"[{stamp()}] Interrupt is STOP command — clearing conversation")
                    self.conversation_mode.clear()
                    self._last_tts_end_time = None
                    self.processing_busy.clear()
                    self._play_earcon(self.sleep_earcon_pcm)
                    self.phrases.play("goodbye")
                    self._set_cooldown()
                    continue                        # ← back to top of while loop

                elif _CORRECTION_RE.search(interrupt_transcript):
                    # Correction: give LLM explicit signal, keep prior context.
                    combined = (f"{transcript.rstrip('.')}. "
                                f"[Correction] {interrupt_transcript}")
                else:
                    # New topic mid-answer: just use interrupt as the new query.
                    combined = interrupt_transcript

                print(f"""[{stamp()}] Combined transcript ({
                      'correction' if _CORRECTION_RE.search(interrupt_transcript) else 'new'
                      }): {combined!r}""")

                # Update history: replace the last user entry with combined.
                # Pop assistant turn (last) and user turn (second to last),
                # then re-add combined user turn. LLM gets the full context.
                if len(self.history) >= 2:
                    self.history.pop()  # remove assistant response
                    self.history.pop()  # remove original user message
                    self.history.append({"role": "user", "content": combined})

                response = _run_llm(combined)
                self._tracer.mark("response_ready")
                if response is None:
                    self.processing_busy.clear()
                    # _handle_server_error will now try restarts and play a
                    # specific LLM / generic fallback phrase.
                    self._handle_server_error()
                    continue

                self.history.append(
                    {
                        "role": "assistant",
                        "content": self._trim_for_memory(response),
                    }
                )

                self.cancel_event.clear()
                self.processing_busy.clear()
                self.is_speaking.set()
                # Guard window at the START of TTS playback so the opening phrase
                # doesn't self-trigger the wake word listener.
                self._set_cooldown(self.cfg.tts_playback_guard_s)
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
        """
        Check all servers, restart any that are down, with simple
        retry + backoff. Then play a specific spoken fallback so the
        user hears what went wrong.
        """
        failed_services = []

        for srv in SERVERS:
            name = srv.get("name", "unknown")
            url = srv.get("url", "")
            if _ping(url):
                continue

            print(f"[{stamp()}] {name} appears down at {url} — attempting restart...")
            failed_services.append(name)

            # Simple retry with exponential backoff.
            max_attempts = 3
            base_delay_s = 1.0
            for attempt in range(1, max_attempts + 1):
                self._m_server_restarts += 1
                print(
                    f"[{stamp()}] Restart attempt {attempt}/{max_attempts} for {name}"
                )
                _start(srv)

                # Backoff: 1s, 2s, 4s.
                delay = base_delay_s * (2 ** (attempt - 1))
                time.sleep(delay)

                if _ping(url):
                    print(f"[{stamp()}] {name} is healthy again after restart")
                    break
            else:
                print(f"[{stamp()}] {name} still unhealthy after {max_attempts} attempts")

        # Choose a spoken fallback phrase based on what failed.
        try:
            if not failed_services:
                # Generic failure – e.g., STT returned None but all services pinged OK.
                self.phrases.play("fallback_generic")
            else:
                # Map service names to more specific fallbacks.
                down = set(s.lower() for s in failed_services)
                if any("whisper" in s or "stt" in s for s in down):
                    self.phrases.play("fallback_stt")
                elif any("gemma" in s or "llm" in s for s in down):
                    self.phrases.play("fallback_llm")
                elif any("kokoro" in s or "tts" in s for s in down):
                    self.phrases.play("fallback_tts")
                else:
                    self.phrases.play("fallback_generic")

            self._set_cooldown()
        except Exception as e:
            print(f"[{stamp()}] Fallback phrase playback failed: {e}")

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------
    def _maybe_summarize_oldest(self) -> None:
        """If history is at capacity, shorten the oldest user message to a short
        keyword stub so it still gives topic context without wasting token budget.
        Only acts when the deque is full (all turns occupied)."""
        max_msgs = self.cfg.memory_turns * 2
        if len(self.history) < max_msgs:
            return  # Still have room, nothing to compress
        # The oldest entry is a user message (index 0 in the deque)
        oldest = self.history[0]
        if oldest["role"] == "user" and len(oldest["content"]) > 60:
            # Compress to first 60 chars, trim to last word boundary
            stub = oldest["content"][:60]
            last_space = stub.rfind(' ')
            if last_space > 20:
                stub = stub[:last_space]
            self.history[0] = {"role": "user", "content": f"[earlier: {stub}…]"}
    
    def _trim_for_memory(self, text: str) -> str:
        """Keep only the first N chars of a reply, but end on a sentence boundary.
        This avoids storing half-sentences that confuse the LLM on next turn."""
        limit = self.cfg.memory_assistant_max_chars
        if len(text) <= limit:
            return text
        # Find the last sentence-ending punctuation before the limit
        cut = text.rfind('.', 0, limit)
        if cut == -1:
            cut = text.rfind('?', 0, limit)
        if cut == -1:
            cut = text.rfind('!', 0, limit)
        # Fall back to hard cut if no punctuation found
        return text[:cut + 1] if cut > 0 else text[:limit]

    

    def _try_fast_path(self, transcript: str) -> str | None:
        if not getattr(self.cfg, 'fast_path_enabled', True):
            return None
        for pattern, handler in self._FP:
            if pattern.search(transcript):
                result = handler()
                print(f"[FastPath] {result[:60]}")
                return result
        return None

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

        # Validate configuration and audio hardware before touching servers.
        self._validate_config()
        self._validate_microphone()

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
