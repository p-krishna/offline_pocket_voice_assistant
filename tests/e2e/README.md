# Human-like end-to-end evaluation

A simulated person talks to the **real** assistant (real wake word, VADs, Whisper, Gemma, Kokoro) inside a
simulated room, in real time, and the run is judged from the user's point of view.

```
UserSim ──speech clips──▶ Room ──mix(speech + noise + loudspeaker echo)──▶ SimMic ──30 ms frames──▶ Pipeline
                            ▲                                                                         │
                            └────────────── SimBackend (loudspeaker, real-time paced) ◀───────────────┘
                                                      LLMProxy (slow / stall / hang / 500) ─▶ llama-server
```

| file | role |
|---|---|
| `sim.py` | `UserSim` (voices, speed, loudness, hesitations, coughs, TV), `Room`, `SimMic`, `SimBackend` |
| `faults.py` | `LLMProxy` – re-streams llama-server output at N tok/s, delays/stalls/hangs/errors, counts client hang-ups |
| `harness.py` | `World` – wires everything, `Check`, trace access, waiting helpers, artifact saving |
| `analyze.py` | trace → audio intervals, WER, dead-air, generic invariants (no overlapping audio, no stuck turn, …) |
| `scenarios.py`, `scenarios_faults.py` | the scenarios (`@scenario(...)`) |
| `run_e2e.py` | runner + `report.md`/`summary.json` |
| `timeline.py` | merged ground-truth/trace timeline of one scenario for debugging |

Add a scenario: write a function `(w: World, ctx) -> List[Check]` decorated with `@scenario(name, description, ...)`.
Use `w.wake_and_say(...)`, `w.user.say(...)`, `w.user.noise_burst(...)`, `w.wait_ev("turn_end", ...)`,
`turn_report(...)` (records ack / first-answer / dead-air timings) and return `Check`s. Prefer asserting
what a person would notice; use `severity="warn"` for things that depend on model quality rather than the pipeline.

Notes
* User speech is synthesized with Kokoro and cached in `tests/e2e/.cache/` (first run is slower and perturbs timings
  because it shares the TTS server; re-run for clean numbers).
* Timing values are measured from the *end of the user's audio clip* (which includes ~150 ms of TTS trailing pad).
* The whole suite takes ~20 min; `--tag core` is the quick subset; `--tag soak` runs the 90 s idle-noise test.
