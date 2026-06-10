#!/usr/bin/env python3
"""
replay_pipeline.py — Replay a saved WAV through STT and optionally LLM.

Separates transcript issues (STT) from reasoning issues (LLM) without the mic.

Usage:
    # STT only (default)
    python src/tools/replay_pipeline.py path/to/file.wav

    # STT + LLM
    python src/tools/replay_pipeline.py --llm path/to/file.wav

    # Multiple files with LLM
    python src/tools/replay_pipeline.py --llm clip1.wav clip2.wav

Output is printed to stdout for easy diffing / shell capture.
"""

import sys
import argparse
import wave
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.config import load_config
from stt.whisper_cpp import WhisperCppSTT
from llm.gemma import GemmaLLM


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit WAV into numpy int16 array."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16), sr


def replay(wav_path: str, stt: WhisperCppSTT, llm: GemmaLLM | None) -> None:
    """Run one file through STT (and optionally LLM), print results."""
    print(f"\n{'='*60}")
    print(f"File : {wav_path}")

    # --- STT stage ---
    try:
        samples, sr = load_wav(wav_path)
    except Exception as e:
        print(f"  [LOAD ERROR] {e}")
        return

    t0 = time.monotonic()
    try:
        transcript = stt.transcribe(samples, sr)
    except Exception as e:
        print(f"  [STT ERROR] {e}")
        return
    stt_ms = (time.monotonic() - t0) * 1000

    print(f"  STT  ({stt_ms:.0f} ms): {transcript!r}")

    if not transcript or not transcript.strip():
        print("  → Blank/noise transcript. Skipping LLM.")
        return

    # --- LLM stage (optional) ---
    if llm is None:
        return

    t1 = time.monotonic()
    try:
        # No conversation history for isolated replay
        response = llm.generate(transcript, history=[])
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return
    llm_ms = (time.monotonic() - t1) * 1000

    print(f"  LLM  ({llm_ms:.0f} ms): {response!r}")


def main():
    parser = argparse.ArgumentParser(description="Replay WAV clips through STT/LLM.")
    parser.add_argument("wavs", nargs="+", help="WAV file paths to replay")
    parser.add_argument(
        "--llm", action="store_true",
        help="Also run transcript through LLM (default: STT only)"
    )
    args = parser.parse_args()

    cfg = load_config()
    stt = WhisperCppSTT(cfg)
    llm = GemmaLLM(cfg) if args.llm else None

    for wav_path in args.wavs:
        replay(wav_path, stt, llm)

    print(f"\n{'='*60}")
    print(f"Done. Replayed {len(args.wavs)} file(s).")


if __name__ == "__main__":
    main()