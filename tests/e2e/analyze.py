"""Trace analysis helpers + generic invariants applied to every scenario."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from tests.e2e.harness import Check, World


def norm_words(s: str) -> List[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = norm_words(ref), norm_words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def audio_items(events: List[dict]) -> List[dict]:
    """Pair audio_start/audio_end events into playback intervals."""
    out, cur = [], None
    for e in events:
        if e["ev"] == "audio_start":
            cur = {"kind": e["kind"], "label": e["label"], "t0": e["t"], "t1": None, "completed": None}
            out.append(cur)
        elif e["ev"] == "audio_end" and cur is not None:
            cur["t1"] = e["t"]
            cur["completed"] = e["completed"]
            cur = None
    return out


def say_end_t(w: World, clip) -> float:
    return w.tt(clip.t_end)


def say_start_t(w: World, clip) -> float:
    return w.tt(clip.t_start)


def sounds_between(items: List[dict], t0: float, t1: float, kinds=None) -> List[dict]:
    return [i for i in items if i["t0"] >= t0 and i["t0"] <= t1 and (kinds is None or i["kind"] in kinds)]


def max_silence_gap(items: List[dict], t_from: float, t_to: float) -> float:
    """Longest stretch in [t_from, t_to] with no assistant sound playing."""
    spans = sorted((max(i["t0"], t_from), min(i["t1"] or t_to, t_to)) for i in items
                   if i["t0"] < t_to and (i["t1"] or t_to) > t_from)
    gap, cursor = 0.0, t_from
    for a, b in spans:
        gap = max(gap, a - cursor)
        cursor = max(cursor, b)
    return max(gap, t_to - cursor)


def last_turn_end(w: World, since: int = 0) -> Optional[dict]:
    t = w.find("turn_end", since)
    return t[-1] if t else None


# ── invariants every scenario must satisfy ───────────────────────────────────
def generic_checks(w: World, allow_errors: bool = False) -> List[Check]:
    ev = w.events
    items = audio_items(ev)
    checks: List[Check] = []

    # 1. The assistant never talks over itself.
    overlaps = []
    for a, b in zip(items, items[1:]):
        if a["t1"] is not None and b["t0"] < a["t1"] - 0.02:
            overlaps.append((a["label"][:20], b["label"][:20], round(a["t1"] - b["t0"], 3)))
    checks.append(Check("no overlapping assistant audio", not overlaps, str(overlaps[:3])))

    # 2. No unexpected pipeline errors / exceptions.
    errs = [e for e in ev if e["ev"] == "error"]
    if not allow_errors:
        checks.append(Check("no error events", not errs, "; ".join(f"{e.get('where')}:{e.get('msg')}" for e in errs[:3])))
    bad = [r for r in w.log_problems if r.levelname in ("ERROR", "CRITICAL")]
    if not allow_errors:
        checks.append(Check("no ERROR/exception logs", not bad, "; ".join(r.getMessage()[:100] for r in bad[:3])))

    # 3. Every turn that started ended (no stuck turns).
    starts, ends = len(w.find("turn_start")), len(w.find("turn_end"))
    checks.append(Check("every turn ended", starts == ends, f"{starts} started / {ends} ended"))

    # 4. Pipeline returned to a listening state (T1 alive, no stuck turn flag).
    alive = all(t.is_alive() for t in w.pipe._threads)
    checks.append(Check("threads alive", alive))
    checks.append(Check("returned to idle (no stuck turn)", w.wait_quiet(15), ""))
    return checks
