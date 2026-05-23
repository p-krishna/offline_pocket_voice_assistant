import argparse
import signal

import pyaudio
import webrtcvad


def main():
    parser = argparse.ArgumentParser(description="Live microphone gate using WebRTC VAD + PyAudio")
    parser.add_argument("--sample-rate", type=int, default=16000, choices=[8000, 16000, 32000, 48000])
    parser.add_argument("--frame-ms", type=int, default=30, choices=[10, 20, 30])
    parser.add_argument("--aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--mic-index", type=int, default=None)
    args = parser.parse_args()

    vad = webrtcvad.Vad(args.aggressiveness)
    running = True

    frames_per_buffer = int(args.sample_rate * args.frame_ms / 1000)

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

    print("Starting live VAD gate... Ctrl+C to stop")
    print(
        f"sample_rate={args.sample_rate} frame_ms={args.frame_ms} "
        f"aggressiveness={args.aggressiveness} mic_index={args.mic_index}"
    )

    import time

    last_state = None
    last_state_start = None
    speech_count = 0
    silence_count = 0
    stable_state = None

    SPEECH_FRAMES_TO_START = 3
    SILENCE_FRAMES_TO_STOP = 5

    try:
        while running:
            frame = stream.read(frames_per_buffer, exception_on_overflow=False)
            speech = vad.is_speech(frame, args.sample_rate)
            now = time.monotonic()

            if speech:
                speech_count += 1
                silence_count = 0
            else:
                silence_count += 1
                speech_count = 0

            new_state = stable_state
            if stable_state != "speech" and speech_count >= SPEECH_FRAMES_TO_START:
                new_state = "speech"
            elif stable_state != "silence" and silence_count >= SILENCE_FRAMES_TO_STOP:
                new_state = "silence"

            if new_state is None:
                continue

            if stable_state is None:
                stable_state = new_state
                last_state = new_state
                last_state_start = now
                print(f"{time.strftime('%H:%M:%S')} -> {new_state} (start)")
                continue

            if new_state != stable_state:
                duration = now - last_state_start
                print(f"{time.strftime('%H:%M:%S')} -> {stable_state} ended after {duration:.2f}s")
                print(f"{time.strftime('%H:%M:%S')} -> {new_state} (start)")
                stable_state = new_state
                last_state = new_state
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