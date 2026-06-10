#!/usr/bin/env python3
"""
replay_stt.py — Send a saved WAV file through the STT server without the mic.

Usage:
    python src/tools/replay_stt.py path/to/file.wav
    python src/tools/replay_stt.py debug_audio/utterance_*.wav

The script reads each WAV, sends it to whisper-server, and prints the transcript.
No wake word, no VAD, no LLM — pure STT path only.
"""

import sys
import wave
import numpy as np
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.config import load_config
from stt.whisper_cpp import WhisperCppSTT


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file into int16 numpy array. Returns (samples, sample_rate)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16)
    return samples, sr


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/tools/replay_stt.py <wav_file> [wav_file2 ...]")
        sys.exit(1)

    cfg = load_config()
    stt = WhisperCppSTT(cfg)

    for wav_path in sys.argv[1:]:
        print(f"\n--- File: {wav_path} ---")
        try:
            samples, sr = load_wav(wav_path)
        except Exception as e:
            print(f"  ERROR loading WAV: {e}")
            continue

        try:
            # Transcribe directly — same call the pipeline makes
            transcript = stt.transcribe(samples, sr)
            print(f"  Transcript: {transcript!r}")
        except Exception as e:
            print(f"  ERROR from STT server: {e}")


if __name__ == "__main__":
    main()