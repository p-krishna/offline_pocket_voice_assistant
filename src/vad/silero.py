import argparse
import collections
import signal
import time

import numpy as np
import pyaudio
import torch
from silero_vad import load_silero_vad

torch.set_num_threads(1)

def pcm16_to_float32(audio_bytes):
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    return audio / 32768.0

def main():
    parser = argparse.ArgumentParser(description="Live microphone gate using Silero VAD + PyAudio")
    parser.add_argument("--sample-rate", type=int, default=16000, choices=[8000, 16000])
    parser.add_argument("--frame-ms", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mic-index", type=int, default=None)
    args = parser.parse_args()

    model = load_silero_vad()
    running = True

    frames_per_buffer = int(args.sample_rate * args.frame_ms / 1000)

    history = collections.deque(maxlen=5)

    def handle_signal(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=args.sample_rate,
        input=True,
        input_device_index=args.mic_index,
        frames_per_buffer=frames_per_buffer,
    )

    last_state = None
    last_state_start = None

    print("Starting Silero VAD mic gate... Ctrl+C to stop")

    try:
        while running:
            frame = stream.read(frames_per_buffer, exception_on_overflow=False)
            x = pcm16_to_float32(frame)
            x = torch.from_numpy(x)

            prob = model(x, args.sample_rate).item()
            history.append(prob)
            avg_prob = sum(history) / len(history)

            state = "speech" if avg_prob >= args.threshold else "silence"
            now = time.monotonic()

            if last_state is None:
                last_state = state
                last_state_start = now
                print(f"{time.strftime('%H:%M:%S')} -> {state} prob={avg_prob:.3f}")
                continue

            if state != last_state:
                duration = now - last_state_start
                print(f"{time.strftime('%H:%M:%S')} -> {last_state} ended after {duration:.2f}s")
                print(f"{time.strftime('%H:%M:%S')} -> {state} prob={avg_prob:.3f}")
                last_state = state
                last_state_start = now

    finally:
        if last_state is not None and last_state_start is not None:
            duration = time.monotonic() - last_state_start
            print(f"{time.strftime('%H:%M:%S')} -> {last_state} ended after {duration:.2f}s")

        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("Stopped.")

if __name__ == "__main__":
    main()