# Offline Pocket Voice Assistant — Day-by-Day Development Roadmap

---

## Project Context

The current stack runs a two-thread pipeline (T1: listen, T2: process+speak) with three persistent local HTTP servers for STT (Whisper), LLM (Gemma), and TTS (Kokoro). All settings are centralized in `src/common/config.py`. Debug WAV files are saved per utterance. The pipeline supports interrupt detection, conversation mode, and basic fallback speech.

**Execution order rationale:** Reliability first → Latency second → Accessibility third → Developer tools fourth → Edge-device readiness last. This order ensures the assistant is dependable before it becomes smarter.

---

## Phase 1 — Reliability & Instrumentation (Weeks 1–4)

*Goal: Make failures visible, measurable, and recoverable. Do not add features until you can measure what's already there.*

---

### Week 1 — Baseline Metrics & Latency Logging

**Weekly deliverable:** Per-turn timing summary printed to terminal. Manual test checklist written and baselined.

| Day | Task | File(s) touched |
|-----|------|----------------|
| 1 | Define success metrics: turn latency, STT latency, LLM latency, TTS latency, false wake count, dropped utterances, interrupt count. Write a short project note. | `docs/metrics.md` (new) |
| 2 | Add `time.monotonic()` timestamps around STT HTTP call start and end in processing thread | `src/pipeline/assistant_pipeline.py` |
| 3 | Add timestamps around LLM call start and end | `src/pipeline/assistant_pipeline.py` |
| 4 | Add timestamps around TTS synthesis start and end. Print one compact per-turn line: `turn=N stt=320ms llm=1450ms tts=210ms total=1980ms` | `src/pipeline/assistant_pipeline.py` |
| 5 | Add counters: queue-full drops, blank STT results, interrupt detections, server restart attempts. Print as part of per-turn or on graceful exit. | `src/pipeline/assistant_pipeline.py` |
| 6 | Create `docs/test_checklist.md` covering: quiet room, fan noise, short command, long command, interrupt during TTS, server down. Run 10 real turns and record baseline numbers. | `docs/test_checklist.md` (new) |
| 7 | *(Buffer)* Clean up logs, note the biggest pain points, update checklist. | — |

**Milestone:** You can now measure every turn and compare runs numerically.

---

### Week 2 — Startup Validation & Server Recovery

**Weekly deliverable:** Assistant refuses to start with a clear spoken/printed error if anything is misconfigured. Server restarts speak a specific failure reason.

| Day | Task | File(s) touched |
|-----|------|----------------|
| 1 | Add startup validation for all model paths in `config.py` before threads launch. Print a clear error and exit if any file is missing. | `src/common/config.py`, `src/pipeline/assistant_pipeline.py` |
| 2 | Validate microphone device index and sample rate at startup before opening the audio stream | `src/vad/webrtc.py`, `src/pipeline/assistant_pipeline.py` |
| 3 | Validate all three server URLs are reachable at startup; add a retry loop with backoff instead of fixed `time.sleep(3)` | `src/common/servers.py` |
| 4 | Improve `handle_server_error()` to name which server failed and speak a specific phrase per service | `src/pipeline/assistant_pipeline.py`, `src/common/config.py` |
| 5 | Add per-service fallback phrases to config: `stt_error_phrase`, `llm_error_phrase`, `tts_error_phrase` | `src/common/config.py` |
| 6 | Test deliberately killing each server in isolation. Confirm recovery behavior and spoken feedback match expectations. | — |
| 7 | *(Buffer)* Document what still breaks and list it as known-issues in README. | `README.md` |

**Milestone:** Assistant recovers from all single-server failures without hanging or crashing silently.

---

### Week 3 — TTS Self-Trigger Reduction

**Weekly deliverable:** Measured false wake count during and after TTS is lower than Week 1 baseline.

| Day | Task | File(s) touched |
|-----|------|----------------|
| 1 | Review all uses of `is_speaking`, `processing_busy`, `cooldown_until`, and conversation re-entry hits in the audio thread. Write a one-page map of how they interact. | `src/pipeline/assistant_pipeline.py` |
| 2 | Tune `ASSISTANT_AUDIO_COOLDOWN_S` for different utterance lengths. Add a longer cooldown after longer TTS. | `src/common/config.py` |
| 3 | Add a guard so wake word hits during `is_speaking` are dropped silently without resetting the counter | `src/pipeline/assistant_pipeline.py` |
| 4 | Improve conversation re-entry gating so the assistant waits a safe number of frames before resuming capture after a TTS play | `src/pipeline/assistant_pipeline.py`, `src/common/config.py` |
| 5 | Tune interrupt energy threshold separately from wake-word guard, since both affect post-speech listening | `src/common/config.py` |
| 6 | Run a focused "assistant hears itself" test: 20 turns with speaker volume at normal, log false wake count, compare to baseline. | — |
| 7 | *(Buffer)* Keep only settings that clearly reduce false triggers. | — |

**Milestone:** False wakes during TTS are rare enough not to interrupt conversations.

---

### Week 4 — Replayable Debugging Workflow

**Weekly deliverable:** Every turn is logged as a JSON record. Any saved WAV can be replayed through STT without the microphone.

| Day | Task | File(s) touched |
|-----|------|----------------|
| 1 | Add structured JSONL turn log: transcript, response length, STT/LLM/TTS/total latency, interrupt flag, drop flag, failure flag | `src/pipeline/assistant_pipeline.py` |
| 2 | Include saved debug WAV path in the JSONL record, and add session ID so multiple runs are distinguishable | `src/pipeline/assistant_pipeline.py` |
| 3 | Write `src/tools/replay_stt.py`: reads a WAV file, sends it to whisper-server:8081, prints transcript | `src/tools/replay_stt.py` (new) |
| 4 | Extend replay tool to optionally send the transcript to llama-server:8080 and print the LLM response | `src/tools/replay_stt.py` |
| 5 | Add `make replay FILE=debug/utterance.wav` Makefile target | `Makefile` |
| 6 | Record 5 reference test clips: quiet-short, noisy-short, long-query, interrupt-query, blank-noise. Store in `tests/audio/`. Run the replay suite. | `tests/audio/` (new) |
| 7 | *(Buffer)* Write `docs/debugging.md` explaining how to use replay and JSONL logs. | `docs/debugging.md` (new) |

**Milestone:** Any bug that can be reproduced with a WAV file can be debugged without a live microphone session.

---

## Phase 2 — Speed & Natural Turn-Taking (Weeks 5–8)

*Goal: Make the assistant feel faster and more natural to talk to. No major architecture changes — tune and extend what exists.*

---

### Week 5 — Conversation Memory Cleanup

**Weekly deliverable:** Three-turn conversations feel coherent. Memory settings are tuned and documented.

| Day | Task | File(s) touched |
|-----|------|----------------|
| 1 | Review how `history` deque works in the pipeline: `memory_turns` × 2 messages, assistant truncation at `memory_assistant_max_chars` | `src/pipeline/assistant_pipeline.py`, `src/common/config.py` |
| 2 | Add a test: ask three related questions and verify the assistant uses previous context correctly in responses | — |
| 3 | Add context summarization: if assistant responses are long, truncate them more aggressively before adding to history | `src/pipeline/assistant_pipeline.py` |
| 4 | Add a spoken "I don't remember that" path when history is too short to answer a reference question | `src/llm/gemma.py` |
| 5 | Tune `MEMORY_TURNS` and `MEMORY_ASSISTANT_MAX_CHARS` for a good default | `src/common/config.py` |
| 6 | Run a multi-turn session focused on continuity. Log turns and annotate where context helped or failed. | — |
| 7 | *(Buffer)* Finalize defaults. | — |

**Milestone:** Multi-turn conversations feel coherent within a 3-turn window without wasting tokens.

---

### Week 6 — Faster Response Feel

**Weekly deliverable:** Measured total turn latency is lower than Week 1 baseline. Per-stage bottleneck is identified.

| Day | Task | File(s) touched |
|-----|------|----------------|
| 1 | Analyze Week 1 and Week 4 latency data. Identify the slowest stage (typically LLM). | `docs/metrics.md` |
| 2 | Reduce `LLMPREDICT_TOKENS` slightly where response quality allows. Shorter responses also help TTS speed. | `src/common/config.py` |
| 3 | Tune utterance finalization settings (`UTTERANCE_SILENCE_HOLD_MS`, `SILERO_STOP_SILENCE_FRAMES`) for faster end-of-speech detection without cutting off speech | `src/common/config.py` |
| 4 | Add fast-path for trivial commands: "stop", "what time is it", "repeat that" — handle before sending to LLM | `src/pipeline/assistant_pipeline.py` |
| 5 | Add a repeat command: re-speak the last TTS response from the cached audio | `src/pipeline/assistant_pipeline.py`, `src/tts/kokoro.py` |
| 6 | Re-run timing tests with 10 turns. Compare total latency and per-stage latency against Week 1. | — |
| 7 | *(Buffer)* Document tuning decisions in `docs/tuning.md`. | `docs/tuning.md` (new) |

**Milestone:** The assistant feels noticeably quicker without swapping any models.

---

### Week 7 — Interrupt Polish

**Weekly deliverable:** Barge-in reliably works for "stop", "no", and "actually…" correction patterns.

| Day | Task | File(s) touched |
|-----|------|----------------|
| 1 | Trace the interrupt flow: T1 detection → `cancel_event` → T2 mid-sentence stop → `interrupt_queue` pickup → combined transcript → LLM rerun | `src/pipeline/assistant_pipeline.py` |
| 2 | Improve breath/hiss filtering in interrupt detection using current RMS + duration logic | `src/pipeline/assistant_pipeline.py`, `src/common/config.py` |
| 3 | Add a cancel acknowledgement earcon or spoken "OK" when an interrupt is confirmed | `src/pipeline/assistant_pipeline.py` |
| 4 | Test "stop": assistant stops immediately and waits for next command | — |
| 5 | Test "no" and "actually, I meant…": combined transcript should re-route the LLM correctly | — |
| 6 | Run 10 interrupt scenarios. Log how many triggered correctly, how many were missed, and how many were false positives. | — |
| 7 | *(Buffer)* Tune thresholds. | `src/common/config.py` |

**Milestone:** Users can confidently correct or stop the assistant mid-speech.

---

### Week 8 — Earcons & Spoken UX Feedback

**Weekly deliverable:** Every system state change produces a distinct, non-speech audio cue. Users can navigate with eyes closed.

| Day | Task | File(s) touched |
|-----|------|----------------|
| 1 | Define the earcon set: wake detected, listening started, thinking, speaking, error, going to sleep. Design tones that are distinguishable. | `docs/earcons.md` (new) |
| 2 | Generate or source earcon WAV files. Add paths to config. Wire into `PhrasePlayer`. | `src/tts/kokoro.py`, `src/common/config.py` |
| 3 | Add wake and listening-started earcons in the audio thread | `src/pipeline/assistant_pipeline.py` |
| 4 | Add thinking (before LLM call) and error earcons in the processing thread | `src/pipeline/assistant_pipeline.py` |
| 5 | Add going-to-sleep and goodbye earcons at conversation timeout, matching existing `conversation_mode` flow | `src/pipeline/assistant_pipeline.py` |
| 6 | Test all earcons with eyes closed. Verify each is distinguishable without being intrusive. Adjust volumes. | — |
| 7 | *(Buffer)* Tune earcon timing so cues feel natural, not cluttered. | — |

**Milestone:** A visually impaired user always knows the assistant's state from audio alone.

---

## Phase 3 — Developer Tools & Deployment (Weeks 9–12)

*Goal: Make the system easier to iterate on, document, and deploy. Prepare a clear path to edge-device testing.*

---

### Week 9 — Config Profiles, Docs & Developer Ergonomics

**Weekly deliverable:** Named config presets. Full tuning and testing documentation. Clean onboarding from scratch.

| Day | Task | File(s) touched |
|-----|------|----------------|
| 1 | Add named config presets: `desktop`, `noisy-room`, `low-latency`, `battery-saver`. Load via `PROFILE` env var on top of existing config | `src/common/config.py` |
| 2 | Print the active profile and key settings at startup for reproducibility | `src/pipeline/assistant_pipeline.py` |
| 3 | Add `make diag` Makefile target that pings all servers, checks audio devices, and prints config summary | `Makefile` |
| 4 | Write `docs/tuning.md` — explains all tunable settings: cooldown, thresholds, silence hold, token count, memory turns | `docs/tuning.md` |
| 5 | Write `docs/testing.md` — repeatable test scenarios with expected outcomes | `docs/testing.md` |
| 6 | Do a clean-start onboarding test: follow only the README and docs to get a session running. Note any gaps. | — |
| 7 | *(Buffer)* Fix documentation gaps. | `README.md`, `docs/` |

**Milestone:** Another developer can set up, tune, and debug the project using docs alone.

---

### Week 10 — Edge-Device Readiness Planning

**Weekly deliverable:** A concrete measurement plan for edge-hardware evaluation. A model decision framework.

| Day | Task | File(s) touched |
|-----|------|----------------|
| 1 | Define target hardware candidates (e.g., Raspberry Pi 5, Orange Pi 5) and list what must be measured: boot time, idle CPU, turn latency per stage, thermal behavior, power draw | `docs/hardware.md` (new) |
| 3 | Add latency-per-stage logging fields needed for cross-device comparison. Ensure JSONL logs capture stage timings consistently. | `src/pipeline/assistant_pipeline.py` |
| 4 | Create a measurement sheet template: 10-turn test protocol, fields to fill per device | `docs/hardware.md` |
| 5 | Build a model decision tree: if STT > 500ms → try `ggml-base.en`; if LLM > 2000ms → try a smaller quant; if TTS > 400ms → profile ONNX runtime. Document as a flow. | `docs/hardware.md` |
| 6 | Run the measurement protocol on the current desktop. Record it as the reference baseline. | — |
| 7 | *(Buffer)* Review hardware plan for gaps. | — |

**Milestone:** When edge hardware arrives, you know exactly what to measure and what decisions to make based on results.

---

### Week 11 — Long-Run Stability & Device-Style Startup

**Weekly deliverable:** System starts automatically, survives 30-minute sessions, and releases audio resources cleanly.

| Day | Task | File(s) touched |
|-----|------|----------------|
| 1 | Write a systemd service file or startup script so the three servers and the pipeline launch on boot | `scripts/start.sh` (new) |
| 2 | Add a startup health summary that prints and speaks which subsystems are ready before the assistant begins listening | `src/pipeline/assistant_pipeline.py` |
| 3 | Add a lightweight watchdog: check server health every 60 seconds in a daemon thread; restart any that have gone down | `src/common/servers.py` |
| 4 | Test long-session stability: run the pipeline for 30 minutes with 20–30 real interactions. Log any failures or hangs. | — |
| 5 | Improve graceful shutdown: confirm audio streams and PyAudio resources are always released, even after exceptions | `src/pipeline/assistant_pipeline.py` |
| 6 | Fix the top two issues found in long-session testing. | — |
| 7 | *(Buffer)* Re-run 30-minute session to confirm fixes. | — |

**Milestone:** The assistant behaves like a device, not a script. It starts, runs, and shuts down predictably.

---

### Week 12 — Final Validation & v1 Prototype

**Weekly deliverable:** Tested, demo-ready build. Release note written. Demo flow recorded.

| Day | Task | File(s) touched |
|-----|------|----------------|
| 1 | Run the full manual test checklist from Week 1 again. Compare final latency and error counts against baseline. | `docs/test_checklist.md` |
| 2 | Run an eyes-closed accessibility session: evaluate feedback clarity, interrupt confidence, and error recovery from a visually impaired user perspective | — |
| 3 | Run a noisy-environment session (fan, music, street noise). Log false wakes and missed wakes. | — |
| 4 | Freeze config defaults for the current best profile. Tag as `v1-prototype` in git. | `src/common/config.py`, `README.md` |
| 5 | Write `RELEASE.md`: what works, known limitations, next priorities | `RELEASE.md` (new) |
| 6 | Record a demo flow: wake → short question → follow-up question → interrupt correction → server failure recovery → conversation timeout to sleep | — |
| 7 | *(Buffer)* Final cleanup, ensure git history is clean, archive baseline log files. | — |

**Milestone:** A stable, documented, tested v1 prototype that is ready for edge-hardware trials.

---

## Weekly Deliverables Summary

| Week | Theme | Deliverable |
|------|-------|-------------|
| 1 | Instrumentation | Per-turn latency logging + manual test checklist |
| 2 | Reliability | Startup validation + server recovery with spoken feedback |
| 3 | Self-trigger reduction | Measured false wake count below Week 1 baseline |
| 4 | Debugging | Replayable test workflow + JSONL turn logs |
| 5 | Memory | Coherent 3-turn conversation memory |
| 6 | Speed | Measurably faster turns, fast-path commands, repeat feature |
| 7 | Interrupt | Reliable barge-in for stop/correct patterns |
| 8 | Accessibility | Earcons for every system state |
| 9 | Developer UX | Named config presets + full documentation |
| 10 | Hardware planning | Edge-device measurement protocol |
| 11 | Stability | Device-style startup, watchdog, 30-min stable session |
| 12 | Validation | Tested v1 prototype with release note and demo |

---

## Key Milestones

| Milestone | End of week | Meaning |
|-----------|-------------|---------|
| **Reliable prototype** | Week 4 | Failures are visible, measurable, and recoverable |
| **Usable daily assistant** | Week 8 | Turn-taking, interruption, and accessibility feedback feel natural |
| **Device-ready prototype** | Week 12 | Documented, stable, and ready for edge-hardware trials |

---

## Files Reference

| File | Role |
|------|------|
| `src/common/config.py` | Single source of truth for all settings — most tuning work happens here |
| `src/pipeline/assistant_pipeline.py` | Main orchestrator — T1 audio thread and T2 processing thread |
| `src/common/servers.py` | Server health ping, start, and wait-for-ready logic |
| `src/tts/kokoro.py` | TTS client — synthesis and playback |
| `src/llm/gemma.py` | LLM client — prompt construction and HTTP call |
| `src/stt/whisper_cpp.py` | STT client — WAV submission and transcript extraction |
| `src/vad/silero.py` | Post-wake utterance end detection |
| `src/vad/webrtc.py` | Continuous silence gate |
| `src/wakeword/listen.py` | openWakeWord wake word detection |
| `Makefile` | Short commands for install, serve, run, test, and check |

---
