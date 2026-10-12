"""
Human-like end-to-end scenarios.

Each scenario describes how a *person* behaves (speaks the wake word, pauses,
hesitates, interrupts, mumbles, coughs, talks over the assistant, goes quiet)
and then checks what the assistant did from the user's point of view:
did it hear me, answer, keep me informed while waiting, stop when I cut in,
and go back to listening? Hard failures are FAIL; "feels worse than it should"
conditions are WARN so they stay visible without hiding real regressions.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from tests.e2e.analyze import (audio_items, generic_checks, last_turn_end, max_silence_gap,
                               say_end_t, say_start_t, sounds_between, wer)
from tests.e2e.harness import Check, World


@dataclass
class Scenario:
    name: str
    description: str
    run: Callable[[World, "Ctx"], List[Check]]
    world: dict = field(default_factory=dict)       # World kwargs
    proxy: dict = field(default_factory=lambda: {"mode": "pass"})
    tags: tuple = ("core",)
    allow_errors: bool = False


class Ctx:
    def __init__(self, proxy):
        self.proxy = proxy
        self.metrics: Dict[str, float] = {}


REGISTRY: Dict[str, Scenario] = {}


def scenario(name, description, world=None, proxy=None, tags=("core",), allow_errors=False):
    def deco(fn):
        REGISTRY[name] = Scenario(name, description, fn, world or {}, proxy or {"mode": "pass"}, tags, allow_errors)
        return fn
    return deco


# ── helpers ───────────────────────────────────────────────────────────────────
def turn_report(w: World, clip, since: int, ctx: Ctx, prefix: str, timeout=60) -> Optional[dict]:
    """Wait for the turn that answers `clip`, record user-perceived timings, return turn_end."""
    te = w.wait_ev("turn_end", since, timeout=timeout)
    if te is None:
        return None
    w.wait_quiet(30)
    ev = w.events
    items = audio_items(ev)
    t_end = say_end_t(w, clip)
    after = [i for i in items if i["t0"] >= t_end - 0.05]
    ack = next((i for i in after if i["label"] == "ack"), None)
    ans = next((i for i in after if i["kind"] == "tts"), None)
    if ack:
        ctx.metrics[prefix + "ack_ms"] = (ack["t0"] - t_end) * 1000
    if ans:
        ctx.metrics[prefix + "first_answer_ms"] = (ans["t0"] - t_end) * 1000
    qd = [e for e in ev[since:] if e["ev"] == "capture_end" and e.get("accepted")]
    if qd:
        ctx.metrics[prefix + "endpoint_ms"] = (qd[0]["t"] - t_end) * 1000
    end_play = [i for i in items if i["kind"] == "tts" and i["t0"] >= t_end]
    if ans and end_play and end_play[-1]["t1"]:
        ctx.metrics[prefix + "answer_total_s"] = end_play[-1]["t1"] - t_end
    if ans:
        ctx.metrics[prefix + "max_dead_air_s"] = max_silence_gap(items, t_end, ans["t0"])
    return te


def contains(text: str, *words) -> bool:
    t = text.lower()
    return any(w.lower() in t for w in words)


def check_latency(ctx: Ctx, prefix: str, ack_max=1.8, endpoint_max=1.3, first_answer_warn=6.0) -> List[Check]:
    out = []
    m = ctx.metrics
    if prefix + "endpoint_ms" in m:
        out.append(Check("end-of-speech detected promptly", m[prefix + "endpoint_ms"] <= endpoint_max * 1000,
                         f"{m[prefix + 'endpoint_ms']:.0f}ms after user stopped"))
    if prefix + "ack_ms" in m:
        out.append(Check("acknowledged quickly (earcon)", m[prefix + "ack_ms"] <= ack_max * 1000,
                         f"{m[prefix + 'ack_ms']:.0f}ms"))
    else:
        out.append(Check("acknowledged quickly (earcon)", False, "no ack sound"))
    if prefix + "first_answer_ms" in m:
        out.append(Check("answer starts reasonably soon", m[prefix + "first_answer_ms"] <= first_answer_warn * 1000,
                         f"{m[prefix + 'first_answer_ms']:.0f}ms after user stopped", severity="warn"))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Core conversation flow
# ═════════════════════════════════════════════════════════════════════════════
@scenario("basic_qa_and_followup",
          "Wake word, a question, then a follow-up WITHOUT the wake word that needs context.")
def basic_qa_and_followup(w: World, ctx: Ctx) -> List[Check]:
    ck: List[Check] = []
    m = w.mark()
    c1 = w.wake_and_say("What is the capital of France?")
    te = turn_report(w, c1, m, ctx, "q1_")
    ck.append(Check("wake word + question produced a turn", te is not None))
    if te:
        ck.append(Check("turn completed", te["outcome"] == "completed", te["outcome"]))
        ck.append(Check("answer is correct (Paris)", contains(te["response"], "paris"), te["response"][:100]))
        ck.append(Check("transcript matches what was said", wer("What is the capital of France?", te["transcript"].replace("Hey Jarvis", "")) <= 0.35,
                        te["transcript"]))
        ck += check_latency(ctx, "q1_")
    # Follow-up without wake word, after the assistant finished.
    time.sleep(0.9)
    m2 = w.mark()
    c2 = w.user.say("And what about Germany?", delay=0.2)
    te2 = turn_report(w, c2, m2, ctx, "q2_")
    ck.append(Check("follow-up accepted without wake word", te2 is not None))
    if te2:
        ck.append(Check("follow-up uses conversation context (Berlin)", contains(te2["response"], "berlin"), te2["response"][:100]))
        ck += check_latency(ctx, "q2_")
    return ck


@scenario("wake_in_one_breath",
          "'Hey Jarvis, <request>' spoken as one continuous phrase (command starts before the wake earcon ends).")
def wake_in_one_breath(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("what is two plus three?", pause=0)
    te = turn_report(w, c, m, ctx, "q_")
    ck.append(Check("turn produced", te is not None))
    if te:
        ck.append(Check("command not clipped (transcript has 'plus')", contains(te["transcript"], "plus", "+", "2", "two"), te["transcript"]))
        heard_two = contains(te["transcript"], " two ", "2 ", "2+")
        ck.append(Check("answer is 5/five" + ("" if heard_two else " (STT misheard 'two')"),
                        contains(te["response"], "five", "5"), te["response"][:100],
                        severity="fail" if heard_two else "warn"))
        ck += check_latency(ctx, "q_")
    return ck


@scenario("wake_then_pause_then_command",
          "Say 'Hey Jarvis' alone, wait for the prompt, then give the command (the Alexa-style two-step).")
def wake_then_pause(w, ctx):
    ck = []
    m = w.mark()
    c = w.user.say("Hey Jarvis.")
    te = w.wait_ev("turn_end", m, timeout=20)
    ck.append(Check("bare wake word handled", te is not None and te["outcome"] in ("wake_only", "blank"),
                    te["outcome"] if te else "no turn"))
    w.wait_quiet(10)
    ans = [i for i in audio_items(w.events) if i["label"] == "yes"]
    ck.append(Check("assistant prompts ('Yes?') after bare wake word", bool(ans)))
    m2 = w.mark()
    c2 = w.user.say("What is the largest planet in the solar system?", delay=0.5)
    te2 = turn_report(w, c2, m2, ctx, "q_")
    ck.append(Check("command after prompt answered", te2 is not None and contains(te2["response"], "jupiter"),
                    te2["response"][:100] if te2 else "no turn"))
    return ck


@scenario("instant_intents", "Time/date/hello/thanks answered instantly without the LLM.")
def instant_intents(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("What time is it?")
    te = turn_report(w, c, m, ctx, "time_")
    ck.append(Check("time answered via fast path", te is not None and te["outcome"] == "fast_time", te["outcome"] if te else "none"))
    if te:
        ck.append(Check("answer contains a clock time", bool(re.search(r"\d{1,2}:\d{2}", te["response"])), te["response"]))
        ck.append(Check("no LLM call needed", w.find("llm_start", m) == []))
    m = w.mark()
    c = w.user.say("What's today's date?", delay=0.8)
    te = turn_report(w, c, m, ctx, "date_")
    ck.append(Check("date answered via fast path", te is not None and te["outcome"] == "fast_date", te["outcome"] if te else "none"))
    m = w.mark()
    c = w.user.say("Thank you.", delay=0.8)
    te = turn_report(w, c, m, ctx, "thx_")
    ck.append(Check("thanks acknowledged", te is not None and te["outcome"] == "fast_thanks", te["outcome"] if te else "none"))
    return ck


@scenario("hesitant_speaker",
          "A person who pauses mid-sentence ('What is … the tallest mountain … in the world?') must not be cut off.")
def hesitant_speaker(w, ctx):
    ck = []
    m = w.mark()
    c = w.user.say_parts([("Hey Jarvis.", 0.5), ("What is,", 0.55), ("the tallest mountain", 0.5), ("in the world?", 0.0)])
    te = turn_report(w, c, m, ctx, "q_")
    ck.append(Check("single turn for a hesitant sentence", len(w.find("turn_start", m)) == 1, f"{len(w.find('turn_start', m))} turns"))
    if te:
        ck.append(Check("whole sentence captured", contains(te["transcript"], "world"), te["transcript"]))
        ck.append(Check("answer mentions Everest", contains(te["response"], "everest"), te["response"][:100]))
    return ck


@scenario("repeat_request", "'Repeat that' replays the previous answer without calling the LLM again.")
def repeat_request(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("Name the largest ocean on Earth.")
    te = turn_report(w, c, m, ctx, "q_")
    ck.append(Check("first answer", te is not None and contains(te["response"], "pacific"), te["response"][:80] if te else ""))
    time.sleep(0.8)
    m = w.mark()
    c2 = w.user.say("Can you repeat that?", delay=0.2)
    te2 = turn_report(w, c2, m, ctx, "rep_")
    ck.append(Check("repeat handled", te2 is not None and te2["outcome"] == "repeat", te2["outcome"] if te2 else "none"))
    ck.append(Check("repeat did not call the LLM", w.find("llm_start", m) == []))
    if te2:
        ck.append(Check("repeated answer is the same", contains(te2["spoken"], "pacific"), te2["spoken"][:80]))
    return ck
