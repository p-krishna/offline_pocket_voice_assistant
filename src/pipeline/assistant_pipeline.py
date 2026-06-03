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
  conversation_mode Event     — set after wake word; cleared after 45s silence
"""

import threading
import queue
import time
import numpy as np
from collections import deque
from pathlib import Path
from wave import open as wave_open

from common import stamp
from common.config import load_config
from common.servers import wait_for_servers, SERVERS, _ping, _start
from vad.webrtc import WebRTCGate
from wakeword.listen import WakeWordListener
from vad.silero import SileroGate
from stt.whisper_cpp import WhisperCppSTT
from llm.gemma import GemmaLLM
from tts.kokoro import KokoroTTS
from tts.phrases import PhrasePlayer


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

        # Shared: set after wake word fires; cleared after conversation timeout.
        self.conversation_mode = threading.Event()

        # Clean shutdown signal for both threads.
        self.running = threading.Event()
        self.running.set()

    # ── Debug helpers ─────────────────────────────────────────────────────────

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
        Returns a value in [0.0, 1.0]. Used to filter low-energy noise
        that Silero might classify as speech.
        """
        if not frames:
            return 0.0
        combined = np.concatenate(frames).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(combined ** 2)))

    # ── Thread 1: Audio ───────────────────────────────────────────────────────

    def _audio_thread(self, stream) -> None:
        """
        Continuously reads mic frames.

        Pre-wake:  WebRTC gate → openWakeWord detection.
        Post-wake: Silero utterance capture → enqueue for processing.
        Interrupt: While T2 is speaking, monitor for 1000ms sustained speech
                   with energy above threshold → signal T2 to cancel.
        Conversation timeout: 45s silence after TTS → back to idle.
        """

        # ── Per-utterance state (reset on each cycle) ─────────────────────────
        after_wake       = False
        silence_frames   = 0
        command_start    = None
        silero_buf       = np.zeros(0, dtype=np.int16)
        recording        = []

        # Pre-roll: capture audio just before wake word so first syllable
        # of the command is not clipped.
        pre_roll_frames = max(1, int(
            (self.cfg.utterance_pre_roll_ms / 1000)
            * self.cfg.sample_rate / self.webrtc.frame_samples
        ))
        pre_roll = deque(maxlen=pre_roll_frames)

        post_roll_frames = max(1, int(
            (self.cfg.utterance_post_roll_ms / 1000)
            * self.cfg.sample_rate / self.webrtc.frame_samples
        ))
        post_roll_queue = deque(maxlen=post_roll_frames)

        # ── Interrupt detection state ─────────────────────────────────────────
        # Tracks sustained speech during TTS playback to detect real interrupts.
        interrupt_speech_ms   = 0.0   # accumulated confirmed-speech duration
        interrupt_recording   = []    # frames captured during potential interrupt
        interrupt_silero      = SileroGate()  # separate Silero instance for interrupt path
        interrupt_silero_buf  = np.zeros(0, dtype=np.int16)

        # ── Conversation timeout state ────────────────────────────────────────
        last_tts_end_time   = None   # monotonic time when TTS last finished
        warning_played      = False  # prevent replaying the warning

        def reset_utterance_state():
            nonlocal after_wake, silence_frames, command_start
            nonlocal silero_buf, recording
            after_wake     = False
            silence_frames = 0
            command_start  = None
            self.silero.state      = None
            self.silero.started_at = None
            self.silero.history    = []
            silero_buf = np.zeros(0, dtype=np.int16)
            recording  = []
            pre_roll.clear()

        def reset_interrupt_state():
            nonlocal interrupt_speech_ms, interrupt_recording, interrupt_silero_buf
            interrupt_speech_ms  = 0.0
            interrupt_recording  = []
            interrupt_silero_buf = np.zeros(0, dtype=np.int16)
            interrupt_silero.state      = None
            interrupt_silero.started_at = None
            interrupt_silero.history    = []

        frame_duration_ms = (self.webrtc.frame_samples / self.cfg.sample_rate) * 1000

        while self.running.is_set():
            pcm   = stream.read(self.webrtc.frame_samples, exception_on_overflow=False)
            t     = time.monotonic()
            frame = np.frombuffer(pcm, dtype=np.int16)

            pre_roll.append(frame)

            # ── Conversation timeout check ────────────────────────────────────
            # Only runs when in conversation mode and T2 is idle.
            if (self.conversation_mode.is_set()
                    and not self.is_speaking.is_set()
                    and not after_wake
                    and last_tts_end_time is not None):

                silent_for = t - last_tts_end_time
                timeout    = self.cfg.conversation_timeout_s
                warning_at = timeout - self.cfg.conversation_warning_s

                if not warning_played and silent_for >= warning_at:
                    # Play warning phrase — "Going to sleep soon".
                    self.phrases.play("going_to_sleep")
                    warning_played = True

                if silent_for >= timeout:
                    # Timeout fired — exit conversation mode.
                    self.phrases.play("goodbye")
                    self.conversation_mode.clear()
                    last_tts_end_time = None
                    warning_played    = False
                    reset_utterance_state()
                    print(f"[{stamp()}] Conversation timeout — returning to idle")
                    continue

            # ── Interrupt detection (while T2 is speaking) ────────────────────
            # Uses a separate Silero instance + RMS energy gate.
            # Requires interrupt_min_speech_ms of sustained speech before firing.
            if self.is_speaking.is_set() and self.conversation_mode.is_set():
                interrupt_recording.append(frame)
                interrupt_silero_buf = np.concatenate([interrupt_silero_buf, frame])

                while len(interrupt_silero_buf) >= interrupt_silero.chunk_size:
                    chunk = interrupt_silero_buf[:interrupt_silero.chunk_size]
                    interrupt_silero_buf = interrupt_silero_buf[interrupt_silero.chunk_size:]
                    prob = interrupt_silero.predict(chunk.tobytes())
                    _, new_state, _, _, _, _ = interrupt_silero.update(prob)

                    if new_state == "speech":
                        interrupt_speech_ms += frame_duration_ms
                    else:
                        # Reset counter — speech must be *sustained*, not sporadic.
                        interrupt_speech_ms = 0.0
                        interrupt_recording = []

                    # Fire interrupt only if:
                    #   1. Sustained speech for interrupt_min_speech_ms
                    #   2. RMS energy above threshold (filters hiss/breath noise)
                    if interrupt_speech_ms >= self.cfg.interrupt_min_speech_ms:
                        rms = self._rms(interrupt_recording)
                        if rms >= self.cfg.interrupt_energy_threshold:
                            print(f"[{stamp()}] Interrupt detected (rms={rms:.4f})")
                            self.phrases.play("i_heard_you")

                            # Hand interrupt audio to T2 via interrupt_queue.
                            try:
                                self.interrupt_queue.put_nowait(list(interrupt_recording))
                            except queue.Full:
                                pass  # T2 already has a pending interrupt

                            # Signal T2 to stop TTS.
                            self.cancel_event.set()
                            reset_interrupt_state()
                        else:
                            # Energy too low — discard and reset.
                            print(f"[{stamp()}] Interrupt suppressed: low energy (rms={rms:.4f})")
                            reset_interrupt_state()
                continue  # while speaking, skip normal wake/utterance path

            # Reset interrupt state when T2 is not speaking.
            if not self.is_speaking.is_set():
                reset_interrupt_state()

            # ── Pre-wake: WebRTC silence gate ─────────────────────────────────
            if not after_wake and not self.conversation_mode.is_set():
                is_speech = self.webrtc.vad.is_speech(pcm, self.webrtc.sample_rate)
                new_webrtc_state = "speech" if is_speech else "silence"

                if self.webrtc.state is None:
                    self.webrtc.state      = new_webrtc_state
                    self.webrtc.started_at = t
                    print(f"[{stamp()}] WebRTC {new_webrtc_state} (start)")
                elif new_webrtc_state != self.webrtc.state:
                    old = self.webrtc.state
                    print(f"[{stamp()}] WebRTC {old} -> {new_webrtc_state} "
                          f"after {t - self.webrtc.started_at:.2f}s")
                    self.webrtc.state      = new_webrtc_state
                    self.webrtc.started_at = t

                if self.webrtc.state != "speech":
                    continue

                # ── Wake word detection ───────────────────────────────────────
                score = self.wake.model.predict(frame).get(self.cfg.wakeword, 0.0)
                self.wake.hits = (self.wake.hits + 1) if score >= self.wake.threshold else 0

                if self.wake.hits >= self.wake.trigger_level:
                    self.wake.hits = 0
                    after_wake     = True
                    command_start  = t
                    silero_buf     = np.zeros(0, dtype=np.int16)
                    silence_frames = 0
                    self.silero.state      = None
                    self.silero.started_at = None
                    self.silero.history    = []
                    recording       = list(pre_roll)
                    post_roll_queue = deque(maxlen=post_roll_frames)
                    self.conversation_mode.set()
                    print(f"[{stamp()}] WakeWord detected: {self.cfg.wakeword} score={score:.3f}")
                    self.phrases.play("listening")

                continue  # always skip post-wake section on this frame

            # ── Post-wake: Silero utterance capture ───────────────────────────
            # Also entered directly if already in conversation_mode (no wake needed).
            if not after_wake and self.conversation_mode.is_set():
                # Conversation mode re-entry: start a fresh utterance capture.
                after_wake    = True
                command_start = t
                silero_buf    = np.zeros(0, dtype=np.int16)
                silence_frames = 0
                self.silero.state      = None
                self.silero.started_at = None
                self.silero.history    = []
                recording       = list(pre_roll)
                post_roll_queue = deque(maxlen=post_roll_frames)

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
                        print(f"[{stamp()}] Silero {old_state} -> {new_state} "
                              f"after {t2 - started:.2f}s avg={avg:.3f}")

                silence_frames = (silence_frames + 1) if self.silero.state == "silence" else 0

                command_age_ms = (t - command_start) * 1000 if command_start else 0
                enough_silence = silence_frames >= self.cfg.silero_stop_silence_frames
                enough_time    = command_age_ms >= self.cfg.utterance_min_ms
                past_floor     = command_age_ms >= self.cfg.utterance_floor_ms
                very_silent    = avg < self.cfg.silero_early_exit_threshold
                early_exit     = past_floor and very_silent and enough_silence

                if (enough_time and enough_silence) or early_exit:
                    if early_exit and not enough_time:
                        print(f"[{stamp()}] Early exit at {command_age_ms:.0f}ms (avg={avg:.3f})")
                    else:
                        print(f"[{stamp()}] Utterance ended")

                    samples = recording + list(post_roll_queue)
                    self.save_debug_wav(samples)

                    # Try to enqueue. If T2 is still processing the previous
                    # command, drop this one and tell the user.
                    try:
                        self.utterance_queue.put_nowait(samples)
                    except queue.Full:
                        print(f"[{stamp()}] Utterance dropped — T2 still busy")
                        self.phrases.play("please_wait")

                    reset_utterance_state()
                    break

    # ── Thread 2: Processing ──────────────────────────────────────────────────

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
            Plays 'repeat_that' phrase on blank result.
            """
            try:
                transcript = self.stt.transcribe(samples, self.cfg.sample_rate)
            except Exception as e:
                print(f"[{stamp()}] STT error: {e}")
                return None

            print(f"[{stamp()}] Transcript: {transcript}")

            if not transcript or not transcript.strip() or transcript.strip() in BLANK_TOKENS:
                self.phrases.play("repeat_that")
                return None

            return transcript.strip()

        def _run_llm(combined_transcript: str) -> str | None:
            """
            Play 'thinking', call LLM with history.
            Returns response string or None on error.
            """
            self.phrases.play("thinking")
            try:
                response = self.llm.generate(
                    combined_transcript,
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
            Checks cancel_event after each sentence.
            Returns True if playback completed, False if cancelled mid-way.
            Stops PyAudio immediately on cancel (mid-sentence cutoff).
            """
            import re, io
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response) if s.strip()]
            if not sentences:
                sentences = [response]

            for sentence in sentences:
                # Check for interrupt before synthesizing next sentence.
                if self.cancel_event.is_set():
                    print(f"[{stamp()}] TTS cancelled before sentence: {sentence[:40]}")
                    return False

                wav = self.tts._synthesize(sentence)
                if wav is None:
                    # Synthesis failed — beep already played inside _synthesize.
                    return False

                # Check again before playing — interrupt may have arrived
                # during the synthesis HTTP call.
                if self.cancel_event.is_set():
                    print(f"[{stamp()}] TTS cancelled after synthesis: {sentence[:40]}")
                    return False

                # Play the sentence through PyAudio.
                # We replicate _play_wav_bytes here so we can honour cancel_event
                # between chunks rather than blocking for the full sentence.
                import soundfile as sf
                buf        = io.BytesIO(wav)
                audio_data, sample_rate = sf.read(buf, dtype="float32")
                pcm        = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                pa         = __import__("pyaudio").PyAudio()
                stream     = pa.open(
                    format=__import__("pyaudio").paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=4096,
                )

                # Write in 4096-sample chunks, checking cancel_event each time.
                chunk_size = 4096
                offset     = 0
                cancelled  = False
                while offset < len(pcm):
                    if self.cancel_event.is_set():
                        print(f"[{stamp()}] TTS cut mid-sentence")
                        cancelled = True
                        break
                    end    = min(offset + chunk_size, len(pcm))
                    stream.write(pcm[offset:end].tobytes())
                    offset = end

                stream.stop_stream()
                stream.close()
                pa.terminate()

                if cancelled:
                    return False

                print(f"[{stamp()}] TTS streamed: {sentence[:60]}...")

            return True  # all sentences played

        while self.running.is_set():
            # Wait for an utterance from T1 (0.5s timeout to stay responsive to shutdown).
            try:
                samples = self.utterance_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # ── STT ──────────────────────────────────────────────────────────
            transcript = _transcribe_samples(samples)
            if transcript is None:
                continue  # blank audio — 'repeat_that' already played

            # Save user message immediately (even if LLM fails, history knows
            # what was asked).
            self.history.append({"role": "user", "content": transcript})

            # ── LLM ──────────────────────────────────────────────────────────
            response = _run_llm(transcript)
            if response is None:
                # LLM failed — speak fallback, attempt server restart.
                self._handle_server_error()
                continue

            # Save truncated assistant response to history.
            self.history.append({
                "role": "assistant",
                "content": response[:self.cfg.memory_assistant_max_chars],
            })

            # ── TTS with interrupt handling ───────────────────────────────────
            self.cancel_event.clear()
            self.is_speaking.set()

            completed = _speak_with_cancel(response)

            self.is_speaking.clear()

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
                    continue

                # Transcribe interrupt audio.
                interrupt_transcript = _transcribe_samples(interrupt_samples)
                if interrupt_transcript is None:
                    # Interrupt was noise — continue conversation normally.
                    print(f"[{stamp()}] Interrupt transcribed as blank — continuing")
                    continue

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

                # Fresh LLM call with updated history.
                response = _run_llm(combined)
                if response is None:
                    self._handle_server_error()
                    continue

                self.history.append({
                    "role": "assistant",
                    "content": response[:self.cfg.memory_assistant_max_chars],
                })

                self.cancel_event.clear()
                self.is_speaking.set()
                _speak_with_cancel(response)
                self.is_speaking.clear()
                self.cancel_event.clear()

            # Record when TTS last finished — drives conversation timeout.
            self._last_tts_end_time = time.monotonic()

    # ── Server error recovery ─────────────────────────────────────────────────

    def _handle_server_error(self) -> None:
        """Check all servers, restart any that are down, play fallback phrase."""
        for srv in SERVERS:
            if not _ping(srv["url"]):
                print(f"[{stamp()}] {srv['name']} down — restarting...")
                _start(srv)
                time.sleep(3)
        try:
            self.tts.speak(self.cfg.fallback_phrase)
        except Exception as e:
            print(f"[{stamp()}] Fallback TTS failed: {e}")

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        print("Pipeline: WebRTC -> openWakeWord -> Silero -> "
              "Whisper[8081] -> Gemma[8080] -> Kokoro[8082]")
        print(f"wakeword={self.cfg.wakeword}  sample_rate={self.cfg.sample_rate}")
        print(f"memory_turns={self.cfg.memory_turns}  "
              f"utterance_floor_ms={self.cfg.utterance_floor_ms}")
        print(f"conversation_timeout={self.cfg.conversation_timeout_s}s  "
              f"interrupt_min_speech={self.cfg.interrupt_min_speech_ms}ms")

        # Block until all three servers respond.
        wait_for_servers()

        # Pre-synthesize all system phrases (uses disk cache when available).
        self.phrases.warm_up()

        print("Press Ctrl+C to stop.")

        audio, stream = self.webrtc.open()

        # Share last_tts_end_time between T2 and T1 via instance variable.
        self._last_tts_end_time = None

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
            print("Stopped.")


if __name__ == "__main__":
    Pipeline().run()
