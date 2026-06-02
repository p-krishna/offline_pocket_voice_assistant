import io
import math
import struct
import wave as wavemod

import numpy as np
import pyaudio


def _make_tone(freq_start: int, freq_end: int, duration_ms: int, rate: int = 22050) -> bytes:
    """
    Generate a linear frequency-sweep tone as raw PCM int16 bytes.
    freq_start == freq_end produces a pure tone (no sweep).
    A short 5ms fade-in and fade-out removes clicks at the edges.
    """
    n = int(rate * duration_ms / 1000)
    fade_samples = int(rate * 0.005)  # 5ms fade window

    samples = []
    for i in range(n):
        # Linearly interpolate frequency across the tone duration.
        freq = freq_start + (freq_end - freq_start) * (i / max(n - 1, 1))
        s = math.sin(2 * math.pi * freq * i / rate)

        # Apply fade-in and fade-out envelope to remove clicks.
        if i < fade_samples:
            s *= i / fade_samples
        elif i > n - fade_samples:
            s *= (n - i) / fade_samples

        samples.append(int(32767 * s))

    return struct.pack(f"<{n}h", *samples)


def _play_pcm(pcm: bytes, rate: int = 22050) -> None:
    """Play raw int16 mono PCM bytes via PyAudio."""
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True)
    stream.write(pcm)
    stream.stop_stream()
    stream.close()
    pa.terminate()


def play_wake() -> None:
    """
    Wake-detected earcon: two rising tones (440 Hz → 660 Hz).
    Tells the user: "I heard you, I'm listening."
    Fires immediately after wake word trigger — adds no latency to STT.
    """
    rate = 22050
    # First tone: 440 Hz, 120ms
    tone1 = _make_tone(440, 440, 120, rate)
    # Short silence gap: 40ms
    gap = bytes(int(rate * 0.040) * 2)
    # Second tone: higher pitch, 120ms — signals readiness
    tone2 = _make_tone(600, 660, 120, rate)
    _play_pcm(tone1 + gap + tone2, rate)


def play_done() -> None:
    """
    Processing-done earcon: single falling tone (660 Hz → 440 Hz).
    Tells the user: "I've finished speaking, you can talk now."
    Fires after speak_streaming() returns.
    Mirrors the wake earcon in reverse — easy to learn the pattern.
    """
    rate = 22050
    # One descending sweep tone, 180ms
    tone = _make_tone(660, 440, 180, rate)
    _play_pcm(tone, rate)


# --- Standalone test ---
if __name__ == "__main__":
    import time
    print("Playing wake earcon...")
    play_wake()
    time.sleep(0.5)
    print("Playing done earcon...")
    play_done()
    print("Done.")