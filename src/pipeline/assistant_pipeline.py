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


class Pipeline:
    def __init__(self):
        # Load one shared config so every stage uses the same timing rules.
        self.cfg = load_config()

        # WebRTC is used only before wake word to avoid pointless wake checks.
        self.webrtc = WebRTCGate()

        # Wake word model runs only when WebRTC says there is speech.
        self.wake = WakeWordListener()

        # Silero takes over after wake word and decides when the command is over.
        self.silero = SileroGate()

        # Main state flags for one utterance.
        self.after_wake = False
        self.silence_frames = 0
        self.command_start_time = None

        # Buffers for the current command.
        self.silero_buf = np.zeros(0, dtype=np.int16)
        self.recording = []

        # Keep a little audio before wake word so the command start is not clipped.
        pre_roll_frames = max(1, int((self.cfg.utterance_pre_roll_ms / 1000) * self.cfg.sample_rate / self.webrtc.frame_samples))
        self.pre_roll = deque(maxlen=pre_roll_frames)

    def save_debug_wav(self, samples):
        # Save the command only when debug saving is turned on.
        if not self.cfg.debug_save_wav or not samples:
            return

        # Create the output folder if it does not already exist.
        out_dir = Path(self.cfg.debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write a simple mono 16-bit WAV file.
        path = out_dir / f"utterance_{int(time.time())}.wav"
        with wave_open(str(path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.cfg.sample_rate)
            wf.writeframes(np.concatenate(samples).astype(np.int16).tobytes())

        print(f'[{stamp()}] Debug WAV saved: {path}')

    def reset_command_state(self):
        # Reset only the post-wake command state.
        self.after_wake = False
        self.silence_frames = 0
        self.command_start_time = None
        self.silero.state = None
        self.silero.started_at = None
        self.silero.history = []
        self.silero_buf = np.zeros(0, dtype=np.int16)
        self.recording = []
        self.pre_roll.clear()

    def run(self):
        # Post-roll queue keeps a tiny tail after speech ends.
        post_roll_frames = max(1, int((self.cfg.utterance_post_roll_ms / 1000) * self.cfg.sample_rate / self.webrtc.frame_samples))
        self.post_roll_queue = deque(maxlen=post_roll_frames)

        # Print basic runtime info so debugging is easy.
        print('Pipeline: WebRTC (pre-wake) -> openWakeWord -> Silero (post-wake)')
        print(f'wakeword={self.cfg.wakeword} sample_rate={self.cfg.sample_rate}')
        print(f'debug_mode={self.cfg.debug_mode} debug_save_wav={self.cfg.debug_save_wav}')
        print('Press Ctrl+C to stop.')

        audio, stream = self.webrtc.open()
        try:
            while self.webrtc.running:
                # Read one fixed microphone frame.
                pcm = stream.read(self.webrtc.frame_samples, exception_on_overflow=False)
                t = time.monotonic()
                frame = np.frombuffer(pcm, dtype=np.int16)

                # Keep a small pre-roll history for the next possible wake word.
                self.pre_roll.append(frame)

                # Before wake word, let WebRTC cheaply reject silence.
                if not self.after_wake:
                    new_webrtc_state = 'speech' if self.webrtc.vad.is_speech(pcm, self.webrtc.sample_rate) else 'silence'

                    # Log only when WebRTC changes state.
                    if self.webrtc.state is None:
                        self.webrtc.state = new_webrtc_state
                        self.webrtc.started_at = t
                        print(f'[{stamp()}] WebRTC {new_webrtc_state} (start)')
                    elif new_webrtc_state != self.webrtc.state:
                        old = self.webrtc.state
                        print(f'[{stamp()}] WebRTC {old} -> {new_webrtc_state} after {t - self.webrtc.started_at:.2f}s')
                        self.webrtc.state = new_webrtc_state
                        self.webrtc.started_at = t

                    # Ignore silence before wake word.
                    if self.webrtc.state != 'speech':
                        continue

                    # Run wake word only on speech frames.
                    score = self.wake.model.predict(frame).get(self.cfg.wakeword, 0.0)
                    if score >= self.wake.threshold:
                        self.wake.hits += 1
                    else:
                        self.wake.hits = 0

                    # Once the hit count reaches the trigger, start command capture.
                    if self.wake.hits >= self.wake.trigger_level:
                        self.wake.hits = 0
                        self.after_wake = True
                        self.command_start_time = t
                        self.silero_buf = np.zeros(0, dtype=np.int16)
                        self.silence_frames = 0
                        self.silero.state = None
                        self.silero.started_at = None
                        self.silero.history = []
                        self.recording = list(self.pre_roll)
                        self.post_roll_queue = deque(maxlen=post_roll_frames)
                        print(f'[{stamp()}] WakeWord detected: {self.cfg.wakeword} score={score:.3f}')
                    continue

                # After wake word, only Silero decides speech start/end.
                self.recording.append(frame)
                self.post_roll_queue.append(frame)
                self.silero_buf = np.concatenate([self.silero_buf, frame])

                # Process Silero in fixed-size chunks.
                while len(self.silero_buf) >= self.silero.chunk_size:
                    chunk = self.silero_buf[:self.silero.chunk_size]
                    self.silero_buf = self.silero_buf[self.silero.chunk_size:]
                    prob = self.silero.predict(chunk.tobytes())
                    changed, new_state, old_state, avg, started, t2 = self.silero.update(prob)

                    # Log only real state changes.
                    if changed:
                        if old_state is None:
                            print(f'[{stamp()}] Silero {new_state} (start) avg={avg:.3f}')
                        else:
                            print(f'[{stamp()}] Silero {old_state} -> {new_state} after {t2 - started:.2f}s avg={avg:.3f}')

                    # Count silence only after Silero has actually decided silence.
                    if self.silero.state == 'silence':
                        self.silence_frames += 1
                    else:
                        self.silence_frames = 0

                    # Make sure the utterance is long enough before finalizing.
                    command_age_ms = (t - self.command_start_time) * 1000 if self.command_start_time else 0
                    enough_silence = self.silence_frames >= self.cfg.silero_stop_silence_frames
                    enough_time = command_age_ms >= self.cfg.utterance_min_ms

                    # Finalize only when the command is long enough and silence has stayed stable.
                    if enough_time and enough_silence:
                        print(f'[{stamp()}] Utterance ended')
                        self.save_debug_wav(self.recording + list(self.post_roll_queue))
                        self.reset_command_state()
                        break

        except KeyboardInterrupt:
            print('\nStopping pipeline...')
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            print('Stopped.')


if __name__ == '__main__':
    Pipeline().run()
