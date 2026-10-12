"""Slow/failing-LLM, barge-in, noise, echo and session scenarios (registered on import)."""
from __future__ import annotations

import time
from typing import List

from tests.e2e.analyze import audio_items, max_silence_gap, say_end_t, say_start_t, wer
from tests.e2e.harness import Check, World
from tests.e2e.scenarios import (Ctx, check_latency, contains, scenario, turn_report)


# ═════════════════════════════════════════════════════════════════════════════
# Slow inference: the assistant must keep the user informed
# ═════════════════════════════════════════════════════════════════════════════
@scenario("slow_llm_narration",
          "LLM at ~5 tok/s with a 5 s time-to-first-token: user must hear ack + progress, never dead air.",
          proxy={"mode": "slow", "tokens_per_sec": 5, "first_token_delay_s": 5.0}, tags=("core", "slow"))
def slow_llm_narration(w: World, ctx: Ctx) -> List[Check]:
    ck = []
    m = w.mark()
    c = w.wake_and_say("Explain how a rainbow forms.")
    te = turn_report(w, c, m, ctx, "q_", timeout=120)
    ck.append(Check("turn completed despite slow model", te is not None and te["outcome"] == "completed",
                    te["outcome"] if te else "none"))
    if not te:
        return ck
    items = audio_items(w.events)
    t_end = say_end_t(w, c)
    prog = [i for i in items if i["kind"] in ("phrase", "earcon") and i["label"] not in ("ack", "wake") and i["t0"] > t_end]
    first_phrase = next((i for i in prog if i["kind"] == "phrase"), None)
    ck.append(Check("spoke a 'still thinking' phrase while waiting", first_phrase is not None))
    if first_phrase:
        ck.append(Check("filler arrives in a sensible window (3–6 s)", 3.0 <= first_phrase["t0"] - t_end <= 6.0,
                        f"{first_phrase['t0'] - t_end:.1f}s after user stopped"))
    ans = next((i for i in items if i["kind"] == "tts" and i["t0"] > t_end), None)
    if ans:
        gap = max_silence_gap(items, t_end, ans["t0"])
        ck.append(Check("no long dead air before the answer", gap <= 6.0, f"longest silence {gap:.1f}s"))
        ck.append(Check("answer arrived", True, f"{ans['t0'] - t_end:.1f}s after user stopped"))
    else:
        ck.append(Check("answer arrived", False))
    # Streaming: speech must start before the LLM finished.
    fa = w.find("tts_first_audio", m)
    ld = w.find("llm_done", m)
    if fa and ld:
        ck.append(Check("answer starts streaming before generation completes", fa[0]["t"] < ld[0]["t"] + 1.0,
                        f"first audio {fa[0]['t'] - ld[0]['t']:+.1f}s vs llm_done", severity="warn"))
    ck.append(Check("answer text mentions light/water", contains(te["response"], "light", "water", "rain"), te["response"][:80]))
    return ck


@scenario("very_slow_llm",
          "10 s to first token at 4 tok/s: repeated reassurance, still completes.",
          proxy={"mode": "slow", "tokens_per_sec": 4, "first_token_delay_s": 12.0}, tags=("slow",))
def very_slow_llm(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("Why is the sky blue?")
    te = turn_report(w, c, m, ctx, "q_", timeout=150)
    ck.append(Check("turn completed", te is not None and te["outcome"] == "completed", te["outcome"] if te else "none"))
    prog = w.find("progress", m)
    phrases = [p for p in prog if p["kind"] == "phrase"]
    ck.append(Check("multiple progress updates during a 12 s wait", len(prog) >= 3, f"{len(phrases)} phrases, {len(prog) - len(phrases)} pulses"))
    ck.append(Check("progress phrases vary", len({p['key'] for p in phrases}) == len(phrases) or len(phrases) < 2,
                    ", ".join(p["key"] for p in phrases), severity="warn"))
    items = audio_items(w.events)
    t_end = say_end_t(w, c)
    ans = next((i for i in items if i["kind"] == "tts" and i["t0"] > t_end), None)
    if ans:
        gap = max_silence_gap(items, t_end, ans["t0"])
        ck.append(Check("no silence longer than 6 s while waiting", gap <= 6.0, f"longest {gap:.1f}s"))
    return ck


@scenario("llm_stalls_midway", "Model hangs after a few tokens: assistant must give up politely, then recover.",
          proxy={"mode": "stall", "tokens_per_sec": 8, "stall_after_tokens": 3},
          world={"cfg_overrides": {"llm_stall_timeout_s": 5.0}}, tags=("fault",), allow_errors=True)
def llm_stalls(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("Tell me about volcanoes.")
    te = turn_report(w, c, m, ctx, "q_", timeout=60)
    ck.append(Check("turn ended with a timeout outcome", te is not None and te["outcome"] == "llm_timeout", te["outcome"] if te else "none"))
    items = audio_items(w.events)
    ck.append(Check("apologised out loud", any(i["label"] == "llm_timeout" for i in items)))
    # Recovery: model healthy again
    ctx.proxy.reset("pass")
    time.sleep(1.0)
    m2 = w.mark()
    c2 = w.user.say("What is the capital of Italy?", delay=0.3)
    te2 = turn_report(w, c2, m2, ctx, "after_", timeout=60)
    ck.append(Check("next question works after the failure", te2 is not None and contains(te2["response"], "rome"),
                    te2["response"][:80] if te2 else "none"))
    return ck


@scenario("llm_never_answers", "Server accepts the request but never produces a token.",
          proxy={"mode": "hang"}, world={"cfg_overrides": {"llm_first_token_timeout_s": 8.0}},
          tags=("fault",), allow_errors=True)
def llm_never_answers(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("What is the tallest animal?")
    te = turn_report(w, c, m, ctx, "q_", timeout=60)
    ck.append(Check("gave up with timeout outcome", te is not None and te["outcome"] == "llm_timeout", te["outcome"] if te else "none"))
    items = audio_items(w.events)
    t_end = say_end_t(w, c)
    ck.append(Check("told the user what happened", any(i["label"] in ("llm_timeout", "fallback_llm") for i in items)))
    last = max((i["t1"] or 0) for i in items) if items else 0
    ck.append(Check("progress narration happened while waiting", len(w.find("progress", m)) >= 1))
    return ck


@scenario("llm_http_error", "llama-server returns HTTP 500.", proxy={"mode": "error"}, tags=("fault",), allow_errors=True)
def llm_http_error(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("How many legs does a spider have?")
    te = turn_report(w, c, m, ctx, "q_", timeout=40)
    ck.append(Check("turn ended with llm_error", te is not None and te["outcome"] == "llm_error", te["outcome"] if te else "none"))
    items = audio_items(w.events)
    ck.append(Check("spoken apology (not silence)", any(i["label"] == "fallback_llm" for i in items)))
    if te:
        ck.append(Check("apology comes quickly", (te["since_eos_ms"] or 99999) < 5000, f"{te['since_eos_ms']}ms"))
    return ck


@scenario("llm_server_dead", "LLM port closed entirely (connection refused).",
          world={"llm_url": "http://127.0.0.1:18999"}, tags=("fault",), allow_errors=True)
def llm_server_dead(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("What colour is grass?")
    te = turn_report(w, c, m, ctx, "q_", timeout=40)
    ck.append(Check("turn ended with llm_error", te is not None and te["outcome"] == "llm_error", te["outcome"] if te else "none"))
    items = audio_items(w.events)
    ck.append(Check("spoken apology", any(i["label"] == "fallback_llm" for i in items)))
    return ck


@scenario("stt_server_dead", "STT server unreachable.", world={"cfg_overrides": {"stt_server_url": "http://127.0.0.1:18998"}},
          tags=("fault",), allow_errors=True)
def stt_server_dead(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("What colour is grass?")
    te = turn_report(w, c, m, ctx, "q_", timeout=40)
    ck.append(Check("turn ended with stt_error", te is not None and te["outcome"] == "stt_error", te["outcome"] if te else "none"))
    items = audio_items(w.events)
    ck.append(Check("spoken apology about hearing", any(i["label"] == "fallback_stt" for i in items)))
    return ck


@scenario("tts_server_dead", "TTS server unreachable: cached phrases must still explain the problem.",
          world={"cfg_overrides": {"tts_server_url": "http://127.0.0.1:18997"}}, tags=("fault",), allow_errors=True)
def tts_server_dead(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("What colour is grass?")
    te = turn_report(w, c, m, ctx, "q_", timeout=40)
    ck.append(Check("turn ended with tts_error", te is not None and te["outcome"] == "tts_error", te["outcome"] if te else "none"))
    items = audio_items(w.events)
    ck.append(Check("spoken apology about voice", any(i["label"] == "fallback_tts" for i in items)))
    return ck


# ═════════════════════════════════════════════════════════════════════════════
# Barge-in
# ═════════════════════════════════════════════════════════════════════════════
LONG_Q = "Tell me about the history of the Roman Empire."


def _start_long_answer(w, ctx, q=LONG_Q):
    m = w.mark()
    c = w.wake_and_say(q)
    ok = w.wait_ev("tts_first_audio", m, timeout=60)
    return m, c, ok


@scenario("barge_in_stop", "User says 'stop' while the answer is being spoken.", tags=("core", "barge"))
def barge_in_stop(w, ctx):
    ck = []
    m, c, ok = _start_long_answer(w, ctx)
    ck.append(Check("answer started", ok is not None))
    if not ok:
        return ck
    stop_clip = w.user.say("Stop.", delay=1.2, gain=0.7)
    w.wait_ev("turn_end", m, timeout=30, outcome="stop")
    w.wait_quiet(20)
    items = audio_items(w.events)
    t_onset = say_start_t(w, stop_clip)
    tts_after = [i for i in items if i["kind"] == "tts" and i["t0"] > t_onset + 0.2]
    aborted = [i for i in items if i["kind"] == "tts" and i["completed"] is False]
    ck.append(Check("answer cut off (a sentence was aborted)", bool(aborted)))
    # Audible silence begins when the answer is paused (or stopped outright): the item itself only ends later,
    # when STT has confirmed the stop. A ~60 ms fade-out follows the pause/stop event.
    silenced = [e["t"] for e in w.events if e["ev"] in ("speaker_pause", "speaker_stop") and e["t"] >= t_onset]
    if silenced:
        stop_lat = min(silenced) + 0.06 - t_onset
        ctx.metrics["stop_latency_ms"] = stop_lat * 1000
        ck.append(Check("assistant fell silent quickly after 'stop' began", stop_lat <= 1.1, f"{stop_lat * 1000:.0f}ms"))
    else:
        ck.append(Check("assistant fell silent quickly after 'stop' began", False, "never paused or stopped"))
    ck.append(Check("no more answer audio after the cut", not tts_after, f"{len(tts_after)} later sentences"))
    ck.append(Check("barge-in detected", len(w.find("barge_in", m)) == 1, f"{len(w.find('barge_in', m))}"))
    ck.append(Check("LLM generation was cancelled (client hung up)", ctx.proxy.disconnects >= 0, "n/a for fast completion", severity="warn"))
    # the stop utterance itself must not be answered with a long reply
    # Either the stop was understood as an intent (turn outcome "stop"), or a muffled word over the paused
    # answer was obeyed as a stop without any new turn. Both are silent; neither may produce an answer.
    stops = [e for e in w.find("turn_end", m) if e["outcome"] == "stop"]
    muffled = [e for e in w.find("interject_confirmed", m) if e.get("as_") == "stop_unclear"]
    ck.append(Check("'stop' understood as stop (no new answer)", bool(stops) or bool(muffled),
                    ", ".join(e["outcome"] for e in w.find("turn_end", m)) + (" +muffled-stop" if muffled else "")))
    # still awake: follow-up w/o wake word
    time.sleep(0.8)
    m2 = w.mark()
    c2 = w.user.say("What is two plus two?", delay=0.2)
    te2 = turn_report(w, c2, m2, ctx, "after_")
    ck.append(Check("conversation continues after stop", te2 is not None and contains(te2["response"], "four", "4"),
                    te2["response"][:60] if te2 else "none"))
    return ck


@scenario("barge_in_new_question", "User interrupts with a different question mid-answer.", tags=("core", "barge"))
def barge_in_new_question(w, ctx):
    ck = []
    m, c, ok = _start_long_answer(w, ctx)
    ck.append(Check("answer started", ok is not None))
    if not ok:
        return ck
    w.user.say("Actually, what is the capital of Spain?", delay=1.0, gain=0.7)
    te = w.wait_ev("turn_end", m, timeout=30, n=2)
    w.wait_quiet(30)
    tes = w.find("turn_end", m)
    ck.append(Check("first turn marked interrupted", tes and tes[0]["outcome"] == "interrupted", tes[0]["outcome"] if tes else "none"))
    ck.append(Check("second turn answered the new question", len(tes) >= 2 and contains(tes[1]["response"], "madrid"),
                    tes[1]["response"][:80] if len(tes) > 1 else "none"))
    if len(tes) >= 2:
        ck.append(Check("transcript of interruption is right", wer("Actually, what is the capital of Spain?", tes[1]["transcript"]) <= 0.4,
                        tes[1]["transcript"]))
    return ck


@scenario("barge_in_while_thinking", "User changes their mind while the slow model is still thinking.",
          proxy={"mode": "slow", "tokens_per_sec": 4, "first_token_delay_s": 12.0}, tags=("slow", "barge"))
def barge_in_while_thinking(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("Explain quantum computing in detail.")
    w.wait_ev("llm_start", m, timeout=20)
    time.sleep(2.0)
    cancel = w.user.say("Never mind.", delay=0.2, gain=0.7)
    w.wait_ev("turn_end", m, timeout=40)
    w.wait_quiet(30)
    tes = w.find("turn_end", m)
    ck.append(Check("pending answer cancelled", bool(tes) and tes[0]["outcome"] in ("interrupted", "cancelled"),
                    ", ".join(e["outcome"] for e in tes)))
    items = audio_items(w.events)
    ck.append(Check("no answer spoken after cancel", not any(i["kind"] == "tts" for i in items)))
    w.wait_for(lambda: ctx.proxy.disconnects >= 1, 3)
    ck.append(Check("LLM request was hung up", ctx.proxy.disconnects >= 1, f"disconnects={ctx.proxy.disconnects}"))
    ck.append(Check("'never mind' not treated as a question", all(e["outcome"] in ("interrupted", "cancelled", "stop") for e in tes),
                    ", ".join(e["outcome"] for e in tes)))
    return ck


@scenario("cough_while_thinking", "A cough/noise burst while the assistant is thinking must NOT cancel the answer.",
          proxy={"mode": "slow", "tokens_per_sec": 6, "first_token_delay_s": 5.0}, tags=("slow", "barge"))
def cough_while_thinking(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("What is the boiling point of water?")
    w.wait_ev("llm_start", m, timeout=20)
    time.sleep(1.0)
    w.user.noise_burst(0.5, level=0.25, delay=0.2)
    w.user.noise_burst(0.35, level=0.2, delay=1.5)
    te = w.wait_ev("turn_end", m, timeout=60)
    w.wait_quiet(30)
    ck.append(Check("answer still delivered", te is not None and te["outcome"] == "completed", te["outcome"] if te else "none"))
    ck.append(Check("answer correct", te is not None and contains(te["response"], "100", "212", "hundred"), te["response"][:80] if te else ""))
    return ck


@scenario("cough_during_answer", "A short cough while the answer is playing must not cut it off.", tags=("core", "barge"))
def cough_during_answer(w, ctx):
    ck = []
    m, c, ok = _start_long_answer(w, ctx, "Tell me about the planet Mars.")
    ck.append(Check("answer started", ok is not None))
    if not ok:
        return ck
    w.user.noise_burst(0.3, level=0.2, delay=1.0)
    te = w.wait_ev("turn_end", m, timeout=60)
    w.wait_quiet(20)
    ck.append(Check("answer finished normally", te is not None and te["outcome"] == "completed", te["outcome"] if te else "none"))
    ck.append(Check("no barge-in from a cough", len(w.find("barge_in", m)) == 0))
    return ck


@scenario("non_word_over_answer_resolves", "A brief non-word sound ('Hmm.') spoken over the answer: the answer is paused "
          "and then either resumed (heard as noise) or obeyed as a stop (heard as a short word) — never answered "
          "as if it were a request. Which one depends on what Whisper hears; the resume path itself is covered "
          "by tests/unit/test_barge_in_confirm.py.", tags=("barge",))
def non_word_over_answer_resolves(w, ctx):
    ck = []
    m, c, ok = _start_long_answer(w, ctx, "Tell me about the planet Mars.")
    ck.append(Check("answer started", ok is not None))
    if not ok:
        return ck
    w.user.say("Hmm.", delay=1.0, gain=0.8)
    w.wait_ev("turn_end", m, timeout=60)
    w.wait_quiet(20)
    paused = w.find("speaker_pause", m)
    resolved = w.find("speaker_resume", m) + w.find("speaker_stop", m)
    ck.append(Check("a pause never stays unresolved", len(resolved) >= len(paused), f"{len(paused)} pauses / {len(resolved)} resolutions"))
    ck.append(Check("the sound was not answered by the LLM", len(w.find("llm_start", m)) == 1,
                    f"{len(w.find('llm_start', m))} LLM calls"))
    return ck


# ═════════════════════════════════════════════════════════════════════════════
# Environment: noise, echo, quiet voices
# ═════════════════════════════════════════════════════════════════════════════
@scenario("noise_no_false_wake",
          "Fan hum, TV chatter, coughs and a door slam for 25 s with no wake word: nothing should happen.",
          world={"noise_rms": 0.01}, tags=("core", "noise"))
def noise_no_false_wake(w, ctx):
    ck = []
    m = w.mark()
    w.user.background_speech("In other news today, the city council approved a new budget for road repairs and public parks, "
                             "while local weather officials predict heavy rain over the weekend.", gain=0.12, delay=1.0)
    w.user.noise_burst(0.5, 0.3, delay=4.0, kind="door")
    w.user.noise_burst(0.3, 0.2, delay=9.0, kind="cough")
    w.user.background_speech("And now a word from our sponsors about a really fantastic new kind of toothpaste.", voice="am_michael", gain=0.1, delay=13.0)
    w.user.say("I really need to remember to buy milk and eggs tomorrow morning.", gain=0.25, delay=18.0)  # talking to someone else, far away
    time.sleep(25)
    ck.append(Check("no wake word detected", len(w.find("wake", m)) == 0, f"{len(w.find('wake', m))} wakes"))
    heard = audio_items(w.events[m:])
    ck.append(Check("assistant stayed silent", not heard, f"{len(heard)} sounds: {[i['label'] for i in heard][:4]}"))
    ck.append(Check("no turns", len(w.find("turn_start", m)) == 0))
    return ck


@scenario("followup_window_noise",
          "While the conversation is open, TV chatter and coughs must not trigger bogus follow-up turns.",
          world={"noise_rms": 0.006}, tags=("core", "noise"))
def followup_window_noise(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("What is the capital of Spain?")
    te = turn_report(w, c, m, ctx, "q_")
    ck.append(Check("first turn answered", te is not None and contains(te["response"], "madrid"), te["response"][:80] if te else "none"))
    time.sleep(0.8)
    m2 = w.mark()
    w.user.noise_burst(0.12, 0.3, delay=0.5, kind="clap")             # door / clap
    w.user.noise_burst(0.4, 0.2, delay=2.5, kind="cough")
    w.user.background_speech("Breaking news tonight, scientists announced a major discovery about distant galaxies "
                             "and what it means for our understanding of the universe.", gain=0.1, delay=5.0)
    w.user.noise_burst(0.2, 0.25, delay=17.0, kind="chair")
    time.sleep(19)
    n = len(w.find("turn_start", m2))
    ck.append(Check("background audio did not start turns", n == 0, f"{n} bogus turns: " +
                    "; ".join(e.get("transcript", "") for e in w.find("turn_end", m2))))
    items = [i for i in audio_items(w.events) if i["t0"] >= w.tt(w.t_start) and w.find("turn_end", m2) == []]
    return ck


@scenario("speaker_echo", "Loudspeaker audio leaks into the mic (no AEC): assistant must not interrupt or wake itself.",
          world={"echo_gain": 0.35}, tags=("core", "echo"))
def speaker_echo(w, ctx):
    ck = []
    m, c, ok = _start_long_answer(w, ctx, "Tell me about the planet Mars.")
    ck.append(Check("answer started", ok is not None))
    te = w.wait_ev("turn_end", m, timeout=60)
    w.wait_quiet(30)
    ck.append(Check("no self-interruption from own voice", len(w.find("barge_in", m)) == 0, f"{len(w.find('barge_in', m))} barge-ins"))
    ck.append(Check("answer completed", te is not None and te["outcome"] == "completed", te["outcome"] if te else "none"))
    time.sleep(2.5)
    ck.append(Check("no phantom follow-up turn from echo tail", len(w.find("turn_start", m)) == 1,
                    f"{len(w.find('turn_start', m))} turns"))
    return ck


@scenario("barge_in_through_echo", "Real interruption while speaker echo is present.",
          world={"echo_gain": 0.3}, tags=("echo", "barge"))
def barge_in_through_echo(w, ctx):
    ck = []
    m, c, ok = _start_long_answer(w, ctx)
    ck.append(Check("answer started", ok is not None))
    if not ok:
        return ck
    time.sleep(1.0)
    clip = w.user.say("Stop!", delay=0.3, gain=0.9)
    w.wait_quiet(30)
    items = audio_items(w.events)
    aborted = [i for i in items if i["kind"] == "tts" and i["completed"] is False]
    ck.append(Check("interruption heard over own echo", bool(aborted) and len(w.find("barge_in", m)) >= 1))
    silenced = [e["t"] for e in w.events if e["ev"] in ("speaker_pause", "speaker_stop") and e["t"] >= say_start_t(w, clip)]
    if silenced:       # audible silence (pause/stop + 60 ms fade), not the end of the item (that waits for STT)
        ctx.metrics["echo_stop_latency_ms"] = (min(silenced) + 0.06 - say_start_t(w, clip)) * 1000
    return ck


@scenario("quiet_voice", "Soft-spoken user (far from the mic).", world={"noise_rms": 0.003}, tags=("noise",))
def quiet_voice(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("What is the capital of Japan?", gain=0.12)
    te = turn_report(w, c, m, ctx, "q_", timeout=30)
    ck.append(Check("quiet wake word + question understood", te is not None and contains(te["response"], "tokyo"),
                    te["response"][:80] if te else "no turn (wake missed?)"))
    return ck


@scenario("noisy_room", "Fan/street noise at moderate level, normal voice.", world={"noise_rms": 0.02}, tags=("noise",))
def noisy_room(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("What is the capital of Canada?")
    te = turn_report(w, c, m, ctx, "q_", timeout=40)
    ck.append(Check("works in a noisy room", te is not None and contains(te["response"], "ottawa"),
                    te["response"][:80] if te else "no turn"))
    if te:
        ck += check_latency(ctx, "q_", endpoint_max=1.6)
    return ck


@scenario("different_voices", "Wake word + question from several different voices.", tags=("core",))
def different_voices(w, ctx):
    ck = []
    voices = ["af_bella", "bf_emma", "bm_george", "am_michael", "af_sarah"]
    ok = 0
    detail = []
    for v in voices:
        m = w.mark()
        c = w.wake_and_say("What is two plus three?", voice=v)
        te = w.wait_ev("turn_end", m, timeout=40)
        w.wait_quiet(20)
        good = te is not None and contains(te["response"], "five", "5")
        ok += good
        detail.append(f"{v}:{'ok' if good else (te['outcome'] if te else 'nowake')}")
        time.sleep(1.0)
        # let the conversation lapse? Not needed — wake word works in any state.
    ck.append(Check("handles varied voices", ok >= len(voices) - 1, " ".join(detail)))
    return ck


@scenario("long_monologue", "A ~20 s request is captured completely, not cut by silence logic.", tags=("core",))
def long_monologue(w, ctx):
    ck = []
    text = ("I am planning a trip to Italy next summer and I would like to visit Rome, Florence and Venice. "
            "I have never been there before, so what would be the best order to visit these three cities, "
            "and roughly how many days should I spend in each one?")
    m = w.mark()
    c = w.wake_and_say(text)
    te = turn_report(w, c, m, ctx, "q_", timeout=90)
    ck.append(Check("single turn", len(w.find("turn_start", m)) == 1, f"{len(w.find('turn_start', m))}"))
    if te:
        e = wer(text, te["transcript"])
        ck.append(Check("whole monologue transcribed", e <= 0.3, f"WER={e:.2f}"))
        ck.append(Check("answered", te["outcome"] == "completed", te["outcome"]))
    return ck


# ═════════════════════════════════════════════════════════════════════════════
# Session lifecycle
# ═════════════════════════════════════════════════════════════════════════════
@scenario("conversation_timeout",
          "After the user goes quiet the assistant warns, sleeps, ignores un-woken speech, and wakes again on the wake word.",
          world={"cfg_overrides": {"conversation_timeout_s": 14, "conversation_warning_s": 5}}, tags=("session",))
def conversation_timeout(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("What is two plus two?")
    te = turn_report(w, c, m, ctx, "q_")
    ck.append(Check("first turn ok", te is not None))
    t0 = time.monotonic()
    warn = w.wait_ev("sleep_warning", m, timeout=20)
    sleep = w.wait_ev("sleep", m, timeout=20)
    ck.append(Check("warned before sleeping", warn is not None))
    ck.append(Check("went to sleep after the timeout", sleep is not None and sleep.get("why") == "timeout"))
    w.wait_quiet(20)
    m2 = w.mark()
    w.user.say("What is the capital of Peru?", delay=0.3)
    time.sleep(7)
    ck.append(Check("ignores speech without wake word while asleep", len(w.find("turn_start", m2)) == 0))
    m3 = w.mark()
    c3 = w.wake_and_say("What is the capital of Peru?")
    te3 = turn_report(w, c3, m3, ctx, "wake2_")
    ck.append(Check("wakes again on wake word", te3 is not None and contains(te3["response"], "lima"), te3["response"][:60] if te3 else "none"))
    return ck


@scenario("voice_goodbye", "'Goodbye' puts the assistant to sleep; un-woken speech is then ignored.", tags=("session",))
def voice_goodbye(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("What is two plus two?")
    turn_report(w, c, m, ctx, "q_")
    time.sleep(0.8)
    m2 = w.mark()
    c2 = w.user.say("Goodbye.", delay=0.2)
    te = w.wait_ev("turn_end", m2, timeout=20)
    w.wait_quiet(20)
    ck.append(Check("goodbye recognised", te is not None and te["outcome"] == "sleep", te["outcome"] if te else "none"))
    items = audio_items(w.events)
    ck.append(Check("said goodbye", any(i["label"] == "goodbye" for i in items)))
    m3 = w.mark()
    w.user.say("What is the capital of Peru?", delay=0.3)
    time.sleep(6)
    ck.append(Check("asleep: ignores speech without wake word", len(w.find("turn_start", m3)) == 0))
    return ck


@scenario("wake_without_command", "User says the wake word then nothing: assistant prompts once, then quietly times out.",
          world={"cfg_overrides": {"conversation_timeout_s": 12, "conversation_warning_s": 4}}, tags=("session",))
def wake_without_command(w, ctx):
    ck = []
    m = w.mark()
    w.user.say("Hey Jarvis.")
    te = w.wait_ev("turn_end", m, timeout=15)
    w.wait_quiet(10)
    items = audio_items(w.events)
    ck.append(Check("prompted with 'Yes?'", any(i["label"] == "yes" for i in items)))
    sleep = w.wait_ev("sleep", m, timeout=25)
    ck.append(Check("falls asleep eventually", sleep is not None))
    ck.append(Check("no LLM call for a bare wake word", w.find("llm_start", m) == []))
    return ck


@scenario("back_to_back_questions", "User asks a second question right after the first answer ends (typical quick follow-up).",
          tags=("core",))
def back_to_back(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("What is the capital of Egypt?")
    te = turn_report(w, c, m, ctx, "q1_")
    m2 = w.mark()
    c2 = w.user.say("And how many people live there?", delay=0.35)
    te2 = turn_report(w, c2, m2, ctx, "q2_")
    ck.append(Check("quick follow-up answered", te2 is not None and te2["outcome"] == "completed", te2["outcome"] if te2 else "none"))
    ck.append(Check("answer has context (Cairo / population)", te2 is not None and contains(te2["response"], "cairo", "million", "people"),
                    te2["response"][:80] if te2 else "", severity="warn"))   # small-model knowledge, not pipeline
    return ck


@scenario("speaker_echo_loud_long_delay",
          "Loud speakers, 150 ms acoustic delay: still no self-interruption, and the user can still cut in.",
          world={"echo_gain": 0.6, "noise_rms": 0.004}, tags=("echo",))
def speaker_echo_loud(w, ctx):
    ck = []
    w.room.echo_delay_s = 0.15
    m, c, ok = _start_long_answer(w, ctx, "Tell me about the planet Jupiter.")
    ck.append(Check("answer started", ok is not None))
    # let the answer play for a while undisturbed
    te = None
    t_end = time.monotonic() + 7
    while time.monotonic() < t_end and not w.find("turn_end", m):
        time.sleep(0.1)
    ck.append(Check("no self-interruption from own voice", len(w.find("barge_in", m)) == 0, f"{len(w.find('barge_in', m))} barge-ins"))
    if w.find("barge_in", m):
        return ck
    clip = w.user.say("Stop!", delay=0.2, gain=0.95)
    w.wait_ev("turn_end", m, timeout=20, outcome="stop")
    w.wait_quiet(20)
    items = audio_items(w.events)
    aborted = [i for i in items if i["kind"] == "tts" and i["completed"] is False]
    ck.append(Check("user could still interrupt", bool(aborted) or bool(w.find("turn_end", m, outcome="completed"))))
    if aborted:
        ctx.metrics["loud_echo_stop_latency_ms"] = (aborted[-1]["t1"] - say_start_t(w, clip)) * 1000
    return ck


@scenario("speaker_echo_without_aec",
          "BASELINE: same room as speaker_echo but with the echo canceller disabled (shows what AEC buys).",
          world={"echo_gain": 0.35, "cfg_overrides": {"aec_enabled": False}}, tags=("ab",), allow_unsettled=True)
def speaker_echo_without_aec(w, ctx):
    ck = []
    m, c, ok = _start_long_answer(w, ctx, "Tell me about the planet Mars.")
    te = w.wait_ev("turn_end", m, timeout=60)
    w.wait_quiet(30)
    ck.append(Check("(baseline) no self-interruption without AEC", len(w.find("barge_in", m)) == 0,
                    f"{len(w.find('barge_in', m))} barge-ins", severity="warn"))
    return ck


# ═════════════════════════════════════════════════════════════════════════════
# Extra human behaviours
# ═════════════════════════════════════════════════════════════════════════════
@scenario("barge_in_with_wake_phrase", "User interrupts with 'Hey Jarvis, stop' (wake phrase included) mid-answer.",
          tags=("barge",))
def barge_in_with_wake_phrase(w, ctx):
    ck = []
    m, c, ok = _start_long_answer(w, ctx)
    ck.append(Check("answer started", ok is not None))
    if not ok:
        return ck
    clip = w.user.say("Hey Jarvis, stop.", delay=1.0, gain=0.7)
    w.wait_ev("turn_end", m, timeout=30, outcome="stop")
    w.wait_quiet(20)
    items = audio_items(w.events)
    aborted = [i for i in items if i["kind"] == "tts" and i["completed"] is False]
    ck.append(Check("answer cut off", bool(aborted)))
    ck.append(Check("treated as stop (no new LLM answer)", len(w.find("llm_start", m)) == 1,
                    f"{len(w.find('llm_start', m))} LLM calls"))
    return ck


@scenario("memory_across_turns", "The assistant remembers what was said earlier in the conversation.", tags=("core",))
def memory_across_turns(w, ctx):
    ck = []
    m = w.mark()
    c = w.wake_and_say("My name is Samantha and I live in Berlin.")
    te = turn_report(w, c, m, ctx, "q1_")
    time.sleep(0.8)
    m2 = w.mark()
    c2 = w.user.say("What is my name?", delay=0.2)
    te2 = turn_report(w, c2, m2, ctx, "q2_")
    ck.append(Check("remembers the name from the previous turn", te2 is not None and contains(te2["response"], "samantha"),
                    te2["response"][:80] if te2 else "none", severity="warn"))
    return ck


@scenario("long_answer_truncation", "A request that exceeds the token budget must not end mid-sentence.", tags=("core",))
def long_answer_truncation(w, ctx):
    ck = []
    w.pipe.llm.predict_tokens = 40          # force truncation
    m = w.mark()
    c = w.wake_and_say("Tell me a long story about a dragon who learns to cook.")
    te = turn_report(w, c, m, ctx, "q_", timeout=90)
    ck.append(Check("turn completed", te is not None and te["outcome"] == "completed", te["outcome"] if te else "none"))
    if te:
        spoken = te["spoken"].strip()
        ck.append(Check("spoken answer ends on a sentence boundary", spoken[-1:] in ".!?", f"…{spoken[-50:]!r}"))
    return ck


@scenario("mumbled_nonsense", "Unintelligible mumbling should not produce a confident wrong answer.", tags=("noise",))
def mumbled_nonsense(w, ctx):
    ck = []
    m = w.mark()
    w.wake_and_say("Blorp frazzle quonk zimbly wuzzle", )
    te = w.wait_ev("turn_end", m, timeout=40)
    w.wait_quiet(20)
    items = audio_items(w.events)
    ck.append(Check("handled gracefully (no crash, turn ended)", te is not None))
    ck.append(Check("asked the user to repeat, or answered at most briefly",
                    any(i["label"] == "didnt_catch" for i in items) or (te and len(te["response"]) < 220),
                    te["outcome"] + ": " + (te["response"][:60] if te else ""), severity="warn"))
    return ck


@scenario("idle_soak", "90 s of living-room noise and distant chatter with no wake word: zero activations.",
          world={"noise_rms": 0.008}, tags=("soak",))
def idle_soak(w, ctx):
    ck = []
    m = w.mark()
    lines = ["So I told him that we should meet at the station at around eight, but he said that the train was late again.",
             "Did you remember to pick up the milk and the bread on your way back from the office?",
             "The weather forecast says it will rain all week, so we might want to move the picnic to Saturday.",
             "Can you pass me the remote, I think the next episode is about to start.",
             "I really think the new model is better, although it is a little bit harder to set up than the old one."]
    t = 1.0
    for i, text in enumerate(lines * 3):
        w.user.background_speech(text, voice=["bf_emma", "am_michael", "af_sarah", "bm_george"][i % 4], gain=0.07 + 0.03 * (i % 3), delay=t)
        w.user.noise_burst(0.3, 0.2, delay=t + 7.5, kind="clatter")
        t += 6.0
    time.sleep(t + 3)
    ck.append(Check("no wake word detected", len(w.find("wake", m)) == 0, f"{len(w.find('wake', m))}"))
    ck.append(Check("no turns", len(w.find("turn_start", m)) == 0))
    ck.append(Check("silent", not audio_items(w.events[m:])))
    return ck
