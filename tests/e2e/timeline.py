#!/usr/bin/env python3
"""Print a readable merged timeline (ground truth + pipeline trace) for one scenario run dir.

    python tests/e2e/timeline.py reports/e2e/<run>/<scenario> [--amp]
"""
import json
import sys
from pathlib import Path

SKIP = {"amplitude"}


def main():
    d = Path(sys.argv[1])
    rows = []
    for l in open(d / "trace.jsonl"):
        e = json.loads(l)
        if e["ev"] in SKIP:
            continue
        det = {k: v for k, v in e.items() if k not in ("t", "ts", "ev")}
        rows.append((e["t"], f"{e['ev']:<18}", json.dumps(det)[:200]))
    gp = d / "ground_truth.json"
    if gp.exists():
        for g in json.load(open(gp)):
            rows.append((g["t0"], f"USER>> {g['kind']:<10}", f"\"{g['text'][:70]}\" until {g['t1']:.2f}"))
    t0 = min(r[0] for r in rows) if rows else 0
    for t, a, b in sorted(rows, key=lambda r: r[0]):
        print(f"{t - t0:8.2f}  {a} {b}")


if __name__ == "__main__":
    main()
