#!/usr/bin/env python3
"""
Summarise a structured trace (debug_logs/trace_*.jsonl) from a live session or an E2E run.

    PYTHONPATH=src python src/tools/analyze_trace.py [trace.jsonl]     # default: newest in debug_logs/

Prints one line per turn (what was heard, outcome, user-perceived latencies), plus counts of the
things that make a voice assistant feel broken: rejected captures / false wakes, barge-ins,
long silences while waiting, errors, echo-canceller health.
"""
import json
import statistics
import sys
from pathlib import Path


def load(path: Path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main() -> int:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        cands = sorted(Path("debug_logs").glob("trace_*.jsonl"))
        if not cands:
            print("no trace files found (debug_logs/trace_*.jsonl)")
            return 1
        path = cands[-1]
    ev = load(path)
    print(f"# {path}  ({len(ev)} events, {ev[-1]['t'] - ev[0]['t']:.0f}s)\n")

    turns = [e for e in ev if e["ev"] == "turn_end"]
    print(f"{'turn':>4} {'source':<9} {'outcome':<13} {'first_audio':>11} {'total':>7}  transcript → response")
    first_audio = []
    for e in turns:
        starts = [s for s in ev if s["ev"] == "turn_start" and s["turn"] == e["turn"]]
        src = starts[0]["source"] if starts else "?"
        fa = e.get("first_audio_ms")
        if fa is not None:
            first_audio.append(fa)
        print(f"{e['turn']:>4} {src:<9} {e['outcome']:<13} {('%d ms' % fa) if fa is not None else '-':>11} "
              f"{e['total_ms'] / 1000:>6.1f}s  {e['transcript'][:45]!r} → {e['response'][:50]!r}")
    if first_audio:
        print(f"\nfirst audio after you stopped talking: median {statistics.median(first_audio):.0f} ms, max {max(first_audio)} ms")

    rej = [e for e in ev if e["ev"] == "capture_end" and not e.get("accepted")]
    print(f"\ncaptures rejected (noise / false triggers): {len(rej)}  "
          f"(by reason: {', '.join(sorted({r['reason'] for r in rej})) or '-'})")
    wakes = [e for e in ev if e["ev"] == "wake"]
    print(f"wake events: {len(wakes)}   barge-ins: {len([e for e in ev if e['ev'] == 'barge_in'])}   "
          f"interjection cancels: {len([e for e in ev if e['ev'] == 'interject_confirmed'])}   "
          f"unclear transcripts: {len([e for e in ev if e['ev'] == 'stt_unclear'])}")
    prog = [e for e in ev if e["ev"] == "progress"]
    print(f"progress narration events: {len(prog)} ({len([p for p in prog if p['kind'] == 'phrase'])} spoken)")
    for k in ("sleep_warning", "sleep"):
        n = [e for e in ev if e["ev"] == k]
        if n:
            print(f"{k}: {len(n)}")

    # silences between assistant sounds while a turn was pending
    errs = [e for e in ev if e["ev"] == "error"]
    print(f"\nerrors: {len(errs)}")
    for e in errs[:10]:
        print(f"  t={e['t']:.1f} {e.get('where')}: {e.get('msg')}")
    aec = [e for e in ev if e["ev"] == "aec"]
    if aec:
        print(f"echo canceller: converged={aec[-1]['converged']} last ERLE={aec[-1]['erle_db']} dB, resets={aec[-1]['resets']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
