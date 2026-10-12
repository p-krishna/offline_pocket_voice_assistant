#!/usr/bin/env python3
"""
Automated human-like end-to-end evaluation of the whole voice pipeline.

    python tests/e2e/run_e2e.py                    # all scenarios
    python tests/e2e/run_e2e.py -s basic_qa_and_followup,barge_in_stop
    python tests/e2e/run_e2e.py --tag core
    python tests/e2e/run_e2e.py --list

Requires the three local servers (make serve-all). Everything is local/offline.
Artifacts per run: reports/e2e/<timestamp>/{report.md,summary.json,<scenario>/{trace.jsonl,*.wav}}
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ASSISTANT_EVENTS_PATH", "/tmp/assistant_events_e2e.jsonl")
os.environ.setdefault("LOG_LEVEL", "INFO")

from tests.e2e.faults import LLMProxy                 # noqa: E402
from tests.e2e.harness import Check, World            # noqa: E402
from tests.e2e import scenarios as S                  # noqa: E402
from tests.e2e import scenarios_faults                # noqa: E402,F401  (registers more scenarios)
from tests.e2e.analyze import generic_checks          # noqa: E402
from common.servers import _ping, SERVERS             # noqa: E402


def preflight() -> list:
    return [s["name"] for s in SERVERS if not _ping(s["url"])]


def run_one(sc, proxy, out_root: Path, verbose: bool, suffix: str = "") -> dict:
    out = out_root / (sc.name + suffix)
    proxy.reset(**sc.proxy)
    ctx = S.Ctx(proxy)
    world_kw = dict(sc.world)
    world_kw.setdefault("llm_url", f"http://127.0.0.1:{proxy.port}")
    t0 = time.monotonic()
    checks, error = [], None
    w = None
    try:
        w = World(sc.name, out, **world_kw)
        w.start()
        checks = sc.run(w, ctx)
        checks += generic_checks(w, allow_errors=sc.allow_errors, allow_unsettled=sc.allow_unsettled)
    except Exception as e:                                           # scenario crashed
        error = traceback.format_exc()
        checks.append(Check("scenario ran without crashing", False, repr(e)))
    finally:
        if w is not None:
            try:
                w.stop()
            except Exception:
                error = (error or "") + traceback.format_exc()
    res = {
        "name": sc.name + suffix, "description": sc.description,
        "duration_s": round(time.monotonic() - t0, 1),
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail, "severity": c.severity} for c in checks],
        "metrics": {k: round(v, 1) for k, v in ctx.metrics.items()},
        "fail": sum(1 for c in checks if not c.ok and c.severity == "fail"),
        "warn": sum(1 for c in checks if not c.ok and c.severity == "warn"),
        "error": error,
    }
    status = "FAIL" if res["fail"] else ("WARN" if res["warn"] else "PASS")
    print(f"\n=== {sc.name}: {status} ({res['duration_s']}s) — {sc.description}")
    for c in checks:
        if verbose or not c.ok:
            print("   ", c)
    if res["metrics"]:
        print("    metrics:", ", ".join(f"{k}={v:.0f}" for k, v in res["metrics"].items()))
    if error:
        print(error)
    return res


def write_report(results, out_root: Path) -> None:
    lines = ["# E2E report", "", f"Run: {out_root.name}", ""]
    tot_f = sum(r["fail"] for r in results)
    tot_w = sum(r["warn"] for r in results)
    lines.append(f"**{sum(1 for r in results if not r['fail'])}/{len(results)} scenarios without hard failures; "
                 f"{tot_f} failed checks, {tot_w} warnings.**\n")
    lines.append("| scenario | status | fail | warn | secs |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        st = "FAIL" if r["fail"] else ("WARN" if r["warn"] else "PASS")
        lines.append(f"| {r['name']} | {st} | {r['fail']} | {r['warn']} | {r['duration_s']} |")
    lines.append("")
    # latency aggregate
    agg = {}
    for r in results:
        for k, v in r["metrics"].items():
            for suf in ("first_answer_ms", "stop_latency_ms", "max_dead_air_s", "answer_total_s", "endpoint_ms", "ack_ms"):
                if k.endswith(suf):
                    agg.setdefault(("echo_" if k.startswith(("echo_", "loud_echo")) else "") + suf, []).append(v)
                    break
    if agg:
        lines += ["## User-perceived timing (all scenarios)", "", "| metric | n | median | max |", "|---|---|---|---|"]
        for k, v in sorted(agg.items()):
            lines.append(f"| {k} | {len(v)} | {statistics.median(v):.0f} | {max(v):.0f} |")
        lines.append("")
    for r in results:
        lines += [f"## {r['name']}", r["description"], ""]
        for c in r["checks"]:
            mark = "PASS" if c["ok"] else ("WARN" if c["severity"] == "warn" else "FAIL")
            lines.append(f"- **{mark}** {c['name']}" + (f" — {c['detail']}" if c["detail"] else ""))
        if r["metrics"]:
            lines.append("- metrics: " + ", ".join(f"`{k}={v}`" for k, v in r["metrics"].items()))
        if r["error"]:
            lines += ["```", r["error"], "```"]
        lines.append("")
    (out_root / "report.md").write_text("\n".join(lines))
    (out_root / "summary.json").write_text(json.dumps(results, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--scenarios", help="comma separated names")
    ap.add_argument("--tag")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-r", "--repeat", type=int, default=1, help="run each selected scenario N times (flakiness check)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "e2e"))
    a = ap.parse_args()

    reg = S.REGISTRY
    if a.list:
        for n, sc in reg.items():
            print(f"{n:34s} {sc.tags}  {sc.description}")
        return 0
    names = a.scenarios.split(",") if a.scenarios else [n for n, sc in reg.items() if "soak" not in sc.tags]
    if a.tag:
        names = [n for n in reg if a.tag in reg[n].tags and (not a.scenarios or n in names)]
    down = preflight()
    if down:
        print(f"Servers not running: {down}. Start them with `make serve-all`.")
        return 2

    out_root = Path(a.out) / time.strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    proxy = LLMProxy().start()
    results = []
    try:
        for n in names:
            for i in range(a.repeat):
                results.append(run_one(reg[n], proxy, out_root, a.verbose, f"#{i + 1}" if a.repeat > 1 else ""))
    finally:
        proxy.stop()
    write_report(results, out_root)
    print(f"\nReport: {out_root / 'report.md'}")
    fails = sum(r["fail"] for r in results)
    print(f"{len(results)} scenarios, {fails} failed checks, {sum(r['warn'] for r in results)} warnings")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
