import numpy as np
import time
from collections import deque
from pathlib import Path
from wave import open as wave_open

from common import stamp
from common.config import load_config
from vad.webrtc import WebRTCGate
from wakeword.listen import WakeWordListener
from vad.silero import SileroGate
from stt.whisper_cpp import WhisperCppSTT
from llm.gemma import GemmaLLM
from tts.kokoro import KokoroTTS


class Pipeline:
    def __init__(self):
        # Load one shared config so every stage uses the same settings.
        self.cfg = load_config()

        # WebRTC gates silence before wake word detection.
        self.webrtc = WebRTCGate()

        # Wake word runs only when WebRTC says there is speech.
        self.wake = WakeWordListener()

        # Silero decides when the command utterance has ended.
        self.silero = SileroGate()

        # STT, LLM and TTS are now all server-based — no model loading here.
        self.stt = WhisperCppSTT(self.cfg)
        self.llm = GemmaLLM(self.cfg)
        self.tts = KokoroTTS(self.cfg)

        # State flags for one utterance cycle.
        self.after_wake         = False
        self.silence_frames     = 0
        self.command_start_time = None

        # Audio buffers for current command.
        self.silero_buf = np.zeros(0, dtype=np.int16)
        self.recording  = []

        # Pre-roll avoids clipping the start of the command.
        pre_roll_frames = max(1, int(
            (self.cfg.utterance_pre_roll_ms / 1000) * self.cfg.sample_rate / self.webrtc.frame_samples
        ))
        self.pre_roll = deque(maxlen=pre_roll_frames)

    def save_debug_wav(self, samples):
        # Save the command audio when debug saving is enabled.
        if not self.cfg.debug_save_wav or not samples:
            return
        out_dir = Path(self.cfg.debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"utterance_{int(time.time())}.wav"
        with wave_open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.cfg.sample_rate)
            wf.writeframes(np.concatenate(samples).astype(np.int16).tobytes())
        print(f"[{stamp()}] Debug WAV saved: {path}")

    def reset_command_state(self):
        # Reset post-wake state so the pipeline is ready for the next command.
        self.after_wake         = False
        self.silence_frames     = 0
        self.command_start_time = None
        self.silero.state       = None
        self.silero.started_at  = None
        self.silero.history     = []
        self.silero_buf         = np.zeros(0, dtype=np.int16)
        self.recording          = []
        self.pre_roll.clear()

    def run(self):
        # Post-roll captures a small tail after speech ends to avoid clipping.
        post_roll_frames = max(1, int(
            (self.cfg.utterance_post_roll_ms / 1000) * self.cfg.sample_rate / self.webrtc.frame_samples
        ))
        self.post_roll_queue = deque(maxlen=post_roll_frames)

        print("Pipeline: WebRTC -> openWakeWord -> Silero -> Whisper[8081] -> Gemma[8080] -> Kokoro[8082]")
        print(f"wakeword={self.cfg.wakeword}  sample_rate={self.cfg.sample_rate}")
        print(f"debug_mode={self.cfg.debug_mode}  debug_save_wav={self.cfg.debug_save_wav}")
        print("Press Ctrl+C to stop.")

        audio, stream = self.webrtc.open()
        try:
            while self.webrtc.running:
                pcm   = stream.read(self.webrtc.frame_samples, exception_on_overflow=False)
                t     = time.monotonic()
                frame = np.frombuffer(pcm, dtype=np.int16)

                self.pre_roll.append(frame)

                # --- Pre-wake: WebRTC gates silence cheaply ---
                if not self.after_wake:
                    new_webrtc_state = "speech" if self.webrtc.vad.is_speech(pcm, self.webrtc.sample_rate) else "silence"

                    if self.webrtc.state is None:
                        self.webrtc.state      = new_webrtc_state
                        self.webrtc.started_at = t
                        print(f"[{stamp()}] WebRTC {new_webrtc_state} (start)")
                    elif new_webrtc_state != self.webrtc.state:
                        old = self.webrtc.state
                        print(f"[{stamp()}] WebRTC {old} -> {new_webrtc_state} after {t - self.webrtc.started_at:.2f}s")
                        self.webrtc.state      = new_webrtc_state
                        self.webrtc.started_at = t

                    if self.webrtc.state != "speech":
                        continue

                    # --- Wake word detection ---
                    score = self.wake.model.predict(frame).get(self.cfg.wakeword, 0.0)
                    if score >= self.wake.threshold:
                        self.wake.hits += 1
                    else:
                        self.wake.hits = 0

                    if self.wake.hits >= self.wake.trigger_level:
                        self.wake.hits           = 0
                        self.after_wake          = True
                        self.command_start_time  = t
                        self.silero_buf          = np.zeros(0, dtype=np.int16)
                        self.silence_frames      = 0
                        self.silero.state        = None
                        self.silero.started_at   = None
                        self.silero.history      = []
                        self.recording           = list(self.pre_roll)
                        self.post_roll_queue     = deque(maxlen=post_roll_frames)
                        print(f"[{stamp()}] WakeWord detected: {self.cfg.wakeword} score={score:.3f}")
                    continue

                # --- Post-wake: Silero decides end of utterance ---
                self.recording.append(frame)
                self.post_roll_queue.append(frame)
                self.silero_buf = np.concatenate([self.silero_buf, frame])

                while len(self.silero_buf) >= self.silero.chunk_size:
                    chunk           = self.silero_buf[:self.silero.chunk_size]
                    self.silero_buf = self.silero_buf[self.silero.chunk_size:]
                    prob            = self.silero.predict(chunk.tobytes())
                    changed, new_state, old_state, avg, started, t2 = self.silero.update(prob)

                    if changed:
                        if old_state is None:
                            print(f"[{stamp()}] Silero {new_state} (start) avg={avg:.3f}")
                        else:
                            print(f"[{stamp()}] Silero {old_state} -> {new_state} after {t2 - started:.2f}s avg={avg:.3f}")

                    if self.silero.state == "silence":
                        self.silence_frames += 1
                    else:
                        self.silence_frames = 0

                    command_age_ms = (t - self.command_start_time) * 1000 if self.command_start_time else 0
                    enough_silence = self.silence_frames >= self.cfg.silero_stop_silence_frames
                    enough_time    = command_age_ms >= self.cfg.utterance_min_ms

                    if enough_time and enough_silence:
                        print(f"[{stamp()}] Utterance ended")
                        samples = self.recording + list(self.post_roll_queue)
                        self.save_debug_wav(samples)

                        # STT -> LLM -> TTS  (all HTTP, no subprocess)
                        transcript = self.stt.transcribe(samples, self.cfg.sample_rate)
                        print(f"[{stamp()}] Transcript: {transcript}")

                        response = self.llm.generate(transcript)
                        print(f"[{stamp()}] LLM: {response}")

                        self.tts.speak(response)

                        self.reset_command_state()
                        break

        except KeyboardInterrupt:
            print("\nStopping pipeline...")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            print("Stopped.")


if __name__ == "__main__":
    Pipeline().run()
