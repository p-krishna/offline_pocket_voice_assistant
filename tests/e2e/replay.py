#!/usr/bin/env python3
"""
Replay a recording (e.g. a saved debug WAV, or any 16 kHz-ish speech WAV) into the full pipeline
exactly as if it came from the microphone, and see how the assistant reacts.

    python tests/e2e/replay.py clip.wav [--lead 1.0] [--tail 25] [--echo 0.0] [--noise 0.002] [--out DIR]

Requires the three servers. Writes trace.jsonl, mic.wav, assistant.wav, stereo mix into --out
(default reports/replay/<timestamp>) and prints a turn summary.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests.e2e.harness import World               # noqa: E402
from tests.e2e.sim import Clip, SR                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="+", help="one or more recordings, played in order")
    ap.add_argument("--lead", type=float, default=1.0, help="silence before the first clip (s)")
    ap.add_argument("--gap", type=float, default=12.0, help="seconds to wait after each clip for the reply")
    ap.add_argument("--echo", type=float, default=0.0, help="loudspeaker→mic gain (0 = headset)")
    ap.add_argument("--noise", type=float, default=0.002, help="ambient noise RMS")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out or ROOT / "reports" / "replay" / time.strftime("%Y%m%d_%H%M%S"))
    w = World("replay", out, noise_rms=a.noise, echo_gain=a.echo)
    w.start()
    try:
        t = time.monotonic() + a.lead
        for path in a.wav:
            x, sr = sf.read(path, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SR:
                x = resample_poly(x, SR, sr).astype(np.float32)
            clip = w.room.add_clip(Clip(t, x, Path(path).name))
            w.user.said.append({"text": Path(path).name, "t_start": clip.t_start, "t_end": clip.t_end, "label": ""})
            print(f"playing {path} ({len(x) / SR:.1f}s)")
            time.sleep(max(0, clip.t_end - time.monotonic()) + 0.5)
            w.wait_for(lambda: len(w.find("turn_end")) >= 1 and not w.pipe.turn_active.is_set(), a.gap)
            w.wait_quiet(a.gap)
            t = time.monotonic() + 0.5
    finally:
        w.stop()
    for e in w.find("turn_end"):
        print(f"turn {e['turn']}: {e['outcome']:<12} heard={e['transcript']!r} said={e['spoken'][:80]!r} first_audio={e['first_audio_ms']}ms")
    print(f"artifacts in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
