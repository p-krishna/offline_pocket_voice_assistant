#!/usr/bin/env python3
"""
create_test_clips.py — Create 5 named reference test clips for regression testing.

Clips created:
  1. quiet-short    — 3 s of near-silence (background room noise floor)
  2. noisy-short    — 3 s of white-noise burst (stress-tests STT rejection)
  3. long-query     — Record a ~10 s spoken query live from mic
  4. interrupt-query — Record a ~5 s spoken query live from mic
  5. blank-noise    — 2 s of pure digital silence (all zeros)

Clips are saved to: test_clips/ (configurable via TEST_CLIPS_DIR env var)
Use them with replay_stt.py or replay_pipeline.py for repeatable debugging.
"""

import os
import sys
import time
import wave
import numpy as np
import pyaudio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.config import load_config

# Output directory for reference clips
CLIPS_DIR = Path(os.getenv("TEST_CLIPS_DIR", "test_clips"))

# Clip definitions: (name, duration_s, source)
# source: "mic" = record from microphone, "synthetic" = generate in code
CLIPS = [
    ("quiet-short",    3,  "mic"),       # mic with no intentional speech
    ("noisy-short",    3,  "whitenoise"),# synthetic white noise
    ("long-query",     10, "mic"),       # speak a long question
    ("interrupt-query", 5, "mic"),       # speak a short question
    ("blank-noise",    2,  "silence"),   # pure zeros
]


def save_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    """Write int16 numpy array to a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.astype(np.int16).tobytes())
    print(f"  Saved: {path}  ({len(samples)/sr:.1f}s, {path.stat().st_size//1024}KB)")


def record_mic(duration_s: int, sr: int, device_index=None) -> np.ndarray:
    """Record from the default mic for duration_s seconds."""
    pa = pyaudio.PyAudio()
    chunk = 512
    frames = []
    stream = pa.open(
        format=pyaudio.paInt16, channels=1, rate=sr,
        input=True, input_device_index=device_index,
        frames_per_buffer=chunk,
    )
    print(f"  Recording {duration_s}s from mic...", end="", flush=True)
    for _ in range(0, int(sr / chunk * duration_s)):
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print(" done.")
    return np.concatenate(frames)


def main():
    cfg = load_config()
    sr = cfg.sample_rate
    device_index = getattr(cfg, "input_device_index", None)

    print(f"Creating reference test clips in: {CLIPS_DIR}/")
    print(f"Sample rate: {sr} Hz\n")

    for name, duration, source in CLIPS:
        out_path = CLIPS_DIR / f"{name}.wav"
        print(f"[{name}]")

        if source == "silence":
            # Pure digital zeros — blank_noise test
            samples = np.zeros(sr * duration, dtype=np.int16)
            save_wav(out_path, samples, sr)

        elif source == "whitenoise":
            # Gaussian noise at ~30% amplitude — stresses STT silence detection
            rng = np.random.default_rng(42)
            samples = (rng.normal(0, 0.3, sr * duration) * 32767).astype(np.int16)
            save_wav(out_path, samples, sr)

        elif source == "mic":
            input(f"  Press ENTER then speak for {duration}s ({name})...")
            samples = record_mic(duration, sr, device_index)
            save_wav(out_path, samples, sr)

        else:
            print(f"  Unknown source '{source}', skipping.")

    print(f"\nAll clips ready. Replay with:")
    print(f"  python src/tools/replay_stt.py {CLIPS_DIR}/*.wav")
    print(f"  python src/tools/replay_pipeline.py --llm {CLIPS_DIR}/*.wav")


if __name__ == "__main__":
    main()