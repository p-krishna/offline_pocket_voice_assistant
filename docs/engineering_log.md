# Engineering log: learnings, mistakes, decisions, reviews and open work

The single common record of *why the project is the way it is*. README.md describes only the current state;
everything historical lives here. **Every entry carries a date** (`YYYY-MM-DD`): the day the change landed or the
mistake was found. Add new entries with today's date; do not rewrite old ones, add a correction instead.

| section | content |
|---|---|
| 1–3 | defect log of the hardening session (2026-10-08): start-up failures, bugs found by reading, defects found by the simulated user |
| 4–5 | measured behaviour and known limitations (as of 2026-10-08) |
| 6 | mistakes we made and the rule each one produced |
| 7 | design decisions, their justification and what was rejected |
| 8 | "before you change X, re-run Y" |
| 9 | microphone input selection (formerly `microphone.md`) |
| 10 | open work for robustness (audit of the former `roadmap.md`) |
| 11 | critical review of 2026-10-13: every fault found and the fix required |
| 12 | fixes applied for that review: what changed, why, how it was verified, what did *not* work |

Dating note: everything in the uncommitted `repurpose_as_german_tutor` working tree was written on 2026-10-08 (hardening
and test harness 00:11–02:42, microphone work 08:15–08:19, by file timestamps). Documentation was consolidated on
2026-10-13. Tooling: `tests/e2e` (see `tests/e2e/README.md`). All numbers are from the real local stack (Whisper
tiny.en, Gemma 4 E2B IQ2_M, Kokoro) on the 8-core / 32 GB desktop; "slow model" scenarios use the fault proxy to
emulate 4–6 tok/s and multi-second time-to-first-token.

## 1. The system did not start

| date | symptom | cause | fix |
|---|---|---|---|
| 2026-10-08 | `ModuleNotFoundError: pkg_resources` from `webrtcvad` | setuptools ≥ 81 removed it | `setuptools<81` in env files |
| 2026-10-08 | wake-word model fails to load | openWakeWord model files are not shipped in the pip package | `make download-wakeword-models` (one-time download) |
| 2026-10-08 | `tflite` crash: "NumPy 1.x cannot run in NumPy 2" | `kokoro-onnx` 0.6 needs NumPy ≥ 2, `tflite-runtime` needs < 2 | wake word runs on the ONNX backend (`WAKEWORD_FRAMEWORK=onnx`), no NumPy pin |
| 2026-10-08 | Kokoro server crashed: "pickled (object) data" | kokoro-onnx ≥ 0.5 expects an `.npz`, repo ships `voices.json` | server converts and caches `voices.npz` once |
| 2026-10-08 | importing the wake-word module parsed `sys.argv` | `argparse` at import time | moved under `__main__` |

## 2. Bugs found by reading the original pipeline

* **2026-10-08** — `datetime.now()` on the *module* (`import datetime`) → the "what time is it" fast path raised `AttributeError`.
* **2026-10-08** — `self.last_tts_end_time` vs `self._last_tts_end_time` → the conversation timeout was never armed after a normal turn.
* **2026-10-08** — `phrases.play("fallback_stt" | "fallback_llm" | "fallback_tts" | "fallback_generic")` — none of those phrases existed, so
  **every failure was silent**: the assistant just went quiet.
* **2026-10-08** — The interrupt queue was never filled, so the "correction"/"combined transcript" logic was dead code; stale history entries were popped instead.
* **2026-10-08** — Spoken "Listening" after the wake word was recorded by the mic and transcribed as the user's request.
* **2026-10-08** — Earcons, phrases and TTS were played from independent threads/devices → they could overlap and race.
* **2026-10-08** — Fixed `UTTERANCE_MIN_MS = 2000` after the wake word cut anyone off mid-sentence after a 320 ms pause, while "early exit"
  could end the capture after 700 ms of silence before the user had started their request.
* **2026-10-08** — Substring intents: any sentence containing *stop*, *hello*, *thanks* or *repeat* was hijacked ("I want to stop at the store…" → goodbye).
* **2026-10-08** — LLM client: a single blocking 20 s request timeout, no streaming — at 5–15 tok/s a 150-token answer simply timed out.
* **2026-10-08** — Duplicate whisper servers were left running by earlier sessions (port clash); the "restart on failure" logic spawned another copy of a server that was merely slow.

## 3. Defects the simulated user found (in order), and the fixes

| date | scenario | what a person would have experienced | root cause | change |
|---|---|---|---|---|
| 2026-10-08 | `slow_llm_narration`, `very_slow_llm` | LLM "failed" every time the model needed > 5 s to start | the 5 s *connect* timeout also covered waiting for response headers | connect timeout removed once connected; separate first-token / stall / total timeouts |
| 2026-10-08 | `barge_in_while_thinking` | cancelling a turn never stopped llama-server decoding | `http.client` drops `conn.sock` for close-delimited responses, so `abort()` hung up on nothing | keep our own socket handle; proxy confirms the hang-up |
| 2026-10-08 | `barge_in_stop` | saying just "Stop." was often not detected | speech flag reset on every frame that had no complete 512-sample Silero chunk (480-sample frames) + 5-chunk smoothing too slow for a one-word command | raw per-chunk probability, persistent accounting, 190 ms threshold, duck after 60 ms |
| 2026-10-08 | `barge_in_stop` | "Stop." transcribed as "Snobber." and answered | tiny.en mis-hears short words, nothing checked confidence | whisper `avg_logprob` gate (`STT_MIN_LOGPROB`): unclear cut-ins mean *stop*, unclear requests get "Sorry, I didn't catch that" |
| 2026-10-08 | `different_voices` | "Hey Jarvis … (1 s pause) … question" became two captures; the question's first words were lost under "Yes?" | fixed silence hold; wake word not recognised once the conversation was open | speculative STT adapts the hold (bare wake word → 2.2 s, trailing "and…" → 1.5 s, finished sentence → 0.56 s); wake word also detected during follow-up captures |
| 2026-10-08 | `followup_window_noise` | claps / TV chatter opened captures and got answered | WebRTC-only gate, absolute RMS threshold | Silero confirmation + relative-to-user loudness + noise-floor tracking |
| 2026-10-08 | `speaker_echo` | **the assistant interrupted itself** on loudspeakers (and answered its own echo tail) | its voice looks like speech to the VAD | echo canceller (PBFDAF + GCC-PHAT delay lock + double-talk freeze) trained by a soft startup noise; learned coupling gate as fallback; calibration window at playback start; mic sample-clock alignment (wall-clock timestamps jittered tens of ms and killed convergence); loudspeaker reference modelled as a gapless device queue (late writes create the same audible gap in the reference as in the room) |
| 2026-10-08 | `speaker_echo` (flaky) | passed 2 of 3 runs | **the simulator itself was wrong**: it resampled 24 kHz TTS to 16 kHz chunk by chunk with linear interpolation, a periodically time-varying filter no adaptive filter can model (ERLE collapsed, resets) | the speaker now resamples a whole item once with the same filter for room and reference; 9/9 repeated echo runs pass, 3/3 baselines without AEC still self-interrupt |
| 2026-10-08 | `barge_in_stop` (flaky) | "Stop." heard as "Snobber." / "Snava." and answered by the LLM | tiny.en on one-word utterances | a brief unrecognised word spoken over the assistant is a *stop*, not a request; cut-in acknowledgements ("okay", "got it", "no") also stop; low-confidence transcripts are never sent to the LLM |
| 2026-10-08 | `wake_in_one_breath` | LLM got "Harvest what is 2 plus 3" / "Darvis, …" | the capture started late in "Jarvis", leaving clipped wake-word fragments | 1.3 s wake lead-in so STT hears the whole "Hey Jarvis" (then stripped); fuzzy wake-phrase stripping; speculative STT transcribes exactly the audio that is finally sent |
| 2026-10-08 | `slow_llm_narration` | first spoken filler collided with an answer that was about to start | fixed timer | filler suppressed while tokens are already flowing |
| 2026-10-08 | (all) | answer only after the *whole* LLM response + TTS | blocking generate → synthesize | token stream → sentence chunker → TTS prefetch → gapless playback; first clause released early |
| 2026-10-08 | `llm_*`, `stt_server_dead`, `tts_server_dead` | silence | missing phrases | every failure mode has a spoken, cached (offline) message; dead servers are restarted without duplicating them |

## 4. Resulting behaviour (37 scenarios, 0 failed checks; measured 2026-10-08)

Measured from the end of the user's audio clip (clips carry ≈ 150 ms of TTS trailing pad, so real perceived numbers are slightly better):

| metric | median | worst |
|---|---|---|
| end of speech detected | 0.81 s | 1.5 s (hesitation hold; 2.3 s when STT is down and the spec transcript fails) |
| soft "I heard you" tick | 0.81 s | – |
| first word of a normal answer | 2.1 s | ≈ 4 s (cold llama.cpp prompt) |
| first word of an answer from the *slow* model (12 s TTFT) | 18 s, but never > 4 s without a sound | – |
| stop latency after the user starts saying "stop" | 0.40 s | 0.62 s |
| speaker echo → false self-interruption | 0 of 9 runs with AEC on; 3 of 3 runs with AEC off | – |
| idle noise / TV chatter → false wake | 0 | – |

## 5. Known limitations (as of 2026-10-08, still open unless §10 says otherwise)

* **Real acoustics are unverified.** E2E uses a room simulator (additive noise, TV chatter, delay + gain echo), not a
  real microphone. (We first believed the dev machine's mic was "digital silence"; that was the wrong device, see
  sections 6 and 9.) Real rooms add reverberation, non-linear speaker
  distortion and AGC; the AEC falls back to the learned-coupling gate if it cannot converge, and headsets bypass it,
  but loudspeaker setups should be re-checked on real hardware (`make analyze-trace` shows ERLE).
* Whisper tiny.en is the weakest link for short or unusual words ("Italy" → "early" in one run). `ggml-base.en`
  halves the error rate on our synthetic set but doubles STT latency (`WHISPER_MODEL=…`, `src/stt/serve_whisper.sh`).
* Gemma 4 E2B IQ2_M sometimes claims abilities it does not have ("I can look that up later"); the repo's system prompt
  is yours to tune (the working-tree prompt is currently the German-tutor WIP, English wake/STT/TTS are unchanged).
* First reply after startup pays ≈ 1–2 s of llama.cpp prompt evaluation (a warm-up request at start primes it).
* TTS (Kokoro CPU) is ≈ 3× real time and slows to ≈ 2× while the LLM is decoding.

## 6. Mistakes we made, and what they taught us

| date | mistake | how it showed up | lesson / rule going forward |
|---|---|---|---|
| 2026-10-08 | **Blamed the hardware for a software default.** Concluded the mic input was "digital silence" (rms≈20). | The app opened PortAudio's *default* input; the conda env's alsa-lib has no PulseAudio plugin, so `default` fell through to the unplugged onboard jack while the Jabra was the system default in Pulse. | Never assume the default device is the one you hear in the OS mixer. Select input by *name* (`MIC_NAME`), log the chosen device and a 5 s level line, and warn loudly on near-silence. Fixed in `audio/devices.py`, `audio/mic.py` (see section 9). |
| 2026-10-08 | **Trusted the simulator.** `speaker_echo` passed 2 of 3 runs and we first tuned the AEC. | The test room resampled 24 kHz TTS to 16 kHz chunk by chunk with linear interpolation: a periodically time-varying filter no adaptive filter can model. | When a test is flaky, first prove the *test world* is physically consistent. Resample whole items once, with the same filter for room and reference. Validate a fix by repeating (9/9 with AEC, 3/3 self-interrupts without). |
| 2026-10-08 | **Fixed symptoms in the wrong layer.** Early "stop" misses were attacked with VAD thresholds. | The real cause was a speech flag reset on every frame without a complete 512-sample Silero chunk (we feed 480-sample frames), plus 5-chunk smoothing. | Check frame-size / state-carry mismatches between components before tuning thresholds. |
| 2026-10-08 | **Used wall-clock timestamps for mic alignment.** | `read()` returning late after a loop stall gave tens of ms of jitter; AEC delay lock kept losing and ERLE collapsed. | Derive the mic time base from the sample count (`mic_t0 + k * frame`), never from "when the call returned". |
| 2026-10-08 | **Modelled the loudspeaker as "plays when written".** | Late writes leave a gap in the room that was absent in the reference, so the filter saw a mismatch exactly at underruns. | The reference is a gapless device *queue*: a late write creates the same audible gap in the reference as in the room. |
| 2026-10-08 | **One blunt LLM timeout.** | 5 s connect timeout also covered waiting for response headers; slow-but-alive models were declared dead and then *restarted* (duplicate servers, port clashes). | Separate connect / first-token / stall / total timeouts; "slow" is not "dead". Restart only after a failed health ping, never spawn a second copy of a merely slow server. |
| 2026-10-08 | **`http.client` closed our socket handle.** | `abort()` hung up on nothing for close-delimited responses; llama-server kept decoding after barge-in. | Keep our own socket reference and verify with the fault proxy that disconnects are counted. |
| 2026-10-08 | **Silent failure paths.** | `phrases.play("fallback_*")` referenced phrases that did not exist, so every failure was mute. | Every failure mode needs a *spoken, offline-cached* message; add a test (`llm_*`, `stt_server_dead`, `tts_server_dead`) so a missing key cannot go unnoticed. |
| 2026-10-08 | **Dead code that looked alive.** | Interrupt queue never filled; `last_tts_end_time` vs `_last_tts_end_time`; `datetime.now()` on the module. | Typos in attribute names fail silently in Python. The trace + E2E scenarios exist to catch "this branch never runs"; prefer asserting behaviour a person would notice. |
| 2026-10-08 | **Substring intent matching.** | "I want to stop at the store" → goodbye. | Intents match near-whole utterances only; cut-in acknowledgements are valid only *while the assistant is talking/thinking*. |
| 2026-10-08 | **Process hygiene.** | `pkill -f <pattern>` killed the shell whose command line contained the pattern; first E2E run after a cold cache perturbed timings because user-speech synthesis shared the TTS server. | Kill by PID or a pattern that does not appear in your own command line. Pre-synthesize user audio before the scenario starts and re-run for clean numbers. |
| 2026-10-13 | **Carried a pass/fail claim forward without opening the reports.** "37 scenarios, 0 failed checks" was copied from a summary written earlier; the report folder showed a failing `barge_in_stop`. | The docs asserted a result no run supported. | A number in the docs must name the run directory it came from (§12.3 does). |
| 2026-10-13 | **Tuned and judged on the same synthetic data.** Thresholds (`STT_MIN_LOGPROB`, 190 ms, …) were set on the scenarios that then "validated" them. | Real-room behaviour is unknown. | Keep a held-out set of real recordings; until then say "synthetic only". |
| 2026-10-13 | **Wrote reconstructed rationale as history.** §7's "rejected alternatives" were inferred from the code. | Plausible but unrecorded reasons look like decisions. | Label inferred rationale as inferred; record the decision when it is made. |
| 2026-10-13 | **Mis-diagnosed the echo canceller from a metric.** "ERLE ≈ 9 dB ⇒ weak filter" and raising the trust threshold to 12 dB broke barge-in through echo. | ERLE is measured against everything in the mic, including the room's noise floor. | Do not tune a safety margin from a metric before checking what bounds it; A/B it with the scenario that exercises it. |
| 2026-10-13 | **A derived flag that follows a noisy signal.** `trusted` was recomputed from the smoothed ERLE, which dips in quiet spells, so barge-in caution flipped on and off (`barge_in_through_echo` failed 2 of 7 runs vs 0 of 6 before). | Latched `converged` had hidden that. | Latch state flags; check flakiness with ≥ 5 repeats against the previous commit before accepting a change. |
| 2026-10-13 | **Changing a behaviour and leaving its measurement unchanged.** After barge-in became pause-then-confirm, `barge_in_stop` still timed the end of the audio item. | A correct change failed the test. | When behaviour changes on purpose, update and *document* the check in the same change (§12.2). |
| 2026-10-08 | **Env pin whack-a-mole.** | `numpy<2` vs kokoro-onnx, `setuptools>=81` vs webrtcvad. | Remove the conflict rather than pinning around it (wake word on ONNX; `setuptools<81`). Both are recorded in `configs/`. |

## 7. Design decisions and their justification

Format: **decision** — why — what we rejected. The *why* and *rejected* parts of the 2026-10-08 entries were
reconstructed from the code and the defect log on 2026-10-13 (review D1); only the defect log (§1–§3) is
contemporaneous. Config names are in `src/common/config.py`.

### Audio I/O
* **2026-10-08** · **One serialised speaker (`audio/speaker.py`) for earcons, cached phrases and TTS.** Independent threads/devices
  overlapped and raced; one queue means sounds cannot overlap and a single `stop()` silences everything.
  Rejected (inferred on 2026-10-13, not recorded when the decision was made): per-component playback with locks (every new sound source re-opens the race).
* **2026-10-08** · **Epoch-based cancellation.** `stop()` bumps an epoch; every queued item and every stage of a turn carries the
  epoch it began in and is dropped when it no longer matches. A TTS sentence that finishes synthesising *after* a
  barge-in therefore cannot play. Rejected (inferred on 2026-10-13, not recorded when the decision was made): cancelling threads individually (late arrivals slip through).
* **2026-10-08** · **20 ms write chunks (`CHUNK_S`).** Bounds stop latency to about one chunk plus the device buffer.
* **2026-10-08** · **Software AEC (PBFDAF, 30 ms blocks, 14 partitions ≈ 420 ms span).** On loudspeakers the assistant's voice is
  indistinguishable from speech to the VAD and it interrupted itself. We know exactly what we play, so we predict
  and subtract it. Partitioned frequency-domain filtering gives a long filter cheaply on CPU; block size equals the
  mic frame so no extra buffering/latency; span must cover device latency + room path (`AEC_PARTITIONS`).
  Freezing adaptation during double-talk stops it learning the user's voice; a divergence guard resets it; the
  freeze is lifted after ~3 s so a changed echo path is still tracked. Rejected (inferred on 2026-10-13, not recorded when the decision was made): "mute the mic while speaking" (kills
  barge-in, which is the point), and a fixed post-TTS cooldown only (still hears the tail, blocks the user).
* **2026-10-08** · **GCC-PHAT delay lock + ~2 s soft-noise calibration at startup (`AEC_CALIBRATE_ON_START`).** The delay is
  unknown (device buffers differ); noise is broadband so it trains every bin before the first reply, rather than
  converging on the first sentence of speech where the user might talk.
* **2026-10-08** · **Learned echo-coupling gate as a fallback, and `INTERRUPT_ECHO_MARGIN`.** If the canceller has not converged
  (reverberant room, AGC, non-linear speaker) barge-in must exceed the *measured* leak by a margin, and the margin is
  raised until convergence. Degrades to "slightly less sensitive" rather than "talks to itself". Headsets bypass AEC.
* **2026-10-08** · **Mic selection by name with logging (`MIC_NAME`, `MIC_INDEX`, `MIC_REQUIRED`).** See sections 6 and 9.
* **2026-10-08** · **Earcons are pure numpy, level-graded** (wake rising, ack soft tick, pulse low, interrupt falling, error
  descending): the user can tell state without looking and without extra TTS latency.

### Turn-taking and endpointing
* **2026-10-08** · **Adaptive silence hold instead of a fixed 2 s.** A fixed hold is wrong both ways: too long for finished
  sentences, too short for hesitation. Speculative STT starts after 420 ms of silence (`SPECULATIVE_STT_MS`) and the
  transcript decides the hold: finished sentence 0.56 s, trailing "and…" 1.5 s, bare wake word 2.2 s, default 0.8 s.
  The speculative transcript is of *exactly* the audio that is finally sent, so there is no second STT pass.
  Rejected (inferred on 2026-10-13, not recorded when the decision was made): bigger fixed hold (sluggish), end-of-turn ML model (extra dependency, offline footprint).
* **2026-10-08** · **1.3 s wake lead-in (`UTTERANCE_WAKE_PRE_ROLL_MS`) + fuzzy wake-phrase stripping.** openWakeWord fires late in
  "Hey Jarvis"; a short pre-roll left fragments ("Darvis", "Harvest") that went to the LLM. Capturing the whole
  phrase and stripping it is more reliable than trying to cut precisely.
* **2026-10-08** · **Wake word also honoured while the conversation is open.** It changes patience ("Hey Jarvis … pause …
  question") instead of being treated as the question.
* **2026-10-08** · **Follow-ups without wake word, guarded three ways:** WebRTC speech hits, Silero confirmation (claps / door slams
  are not speech), and a level at least 30 % of the user's own typical speaking level (`FOLLOWUP_LEVEL_RATIO`; TV and
  other rooms are quieter). Capture that never reaches minimum speech within 2.5 s is dropped as noise.
* **2026-10-08** · **Conversation timeout 45 s with 8 s warning, spoken.** Never go to sleep silently.

### Barge-in
* **2026-10-08** · **Raw per-chunk Silero probability, persistent accounting, 190 ms threshold.** "Stop" is ~8 chunks; smoothing
  delayed it past usefulness. Measured: 0.40 s median stop latency.
* **2026-10-08** · **Duck our voice to 25 % after 60 ms of speech (`BARGE_IN_DUCK_*`).** Instant feedback before confirmation and it
  reduces the echo fed back into the mic; restored on a false alarm so a cough does not destroy an answer.
* **2026-10-13** · **Cutting in over answer audio *pauses* it (fade-out, hold position); STT decides.** Replaces the
  2026-10-08 behaviour (stop immediately, no way back), whose "cheap to undo" rationale was wrong: nothing could undo
  it, so a cough that passed Silero destroyed the answer (F11). Now: noise ⇒ the answer resumes where it stopped;
  "stop" / a muffled short word ⇒ stop and stay silent; real speech ⇒ stop and answer it. The user hears silence
  as fast as before; only the *decision* waits for STT. An unresolved pause continues after 40 s
  (`BARGE_IN_PAUSE_TIMEOUT_S`) so a crashed confirmation can never leave the assistant mute. Over silence/filler
  (thinking phase) nothing is paused; the pending answer is cancelled only after STT confirms speech.
* **2026-10-13** · **Acknowledgements are answers when we just asked a question.** "yes/okay/no/sure" over a reply
  that ended in "?" go to the LLM instead of stopping it (F8); essential for a tutor that asks questions.
* **2026-10-08** · **Whisper confidence gate (`STT_MIN_LOGPROB=-0.7`).** tiny.en turns short words into confident-looking garbage.
  Unclear *cut-in* ⇒ treat as stop; unclear *request* ⇒ "Sorry, I didn't catch that"; never feed it to the LLM.
  Rejected (inferred on 2026-10-13, not recorded when the decision was made): always sending transcripts to the LLM (it answers "Snobber.").

### Latency and feedback
* **2026-10-08** · **Stream tokens → sentence chunker → TTS prefetch → gapless playback; first clause released early (12 words).**
  At 5–15 tok/s the full answer took 20 s+; speaking the first clause is the single biggest perceived win.
  Chunker handles abbreviations and a 40-word cap so TTS calls stay short.
* **2026-10-08** · **Layered feedback while waiting:** ack tick at ≈0.8 s, spoken filler at 3.5 s (`PROGRESS_FIRST_PHRASE_S`) unless
  tokens are already flowing, soft pulse every 4 s, spoken reassurance every 14 s. Goal: never more than ~4 s without
  a sound. Filler is suppressed once tokens flow because the single speaker would delay the real answer.
* **2026-10-08** · **Cached system phrases (`tts/phrases.py`), hash-keyed by text.** They must work when TTS is the thing that is
  down, and editing text invalidates its audio automatically.
* **2026-10-08** · **LLM timeouts 60 s first token / 30 s stall / 120 s total.** Sized for a cold 2-bit model on CPU, deliberately
  generous because narration covers the wait.
* **2026-10-08** · **Memory: 3 turns, assistant text trimmed to 240 chars at a sentence boundary, only what the user actually
  heard.** Keeps the prompt small for a tiny context/CPU, avoids the model "remembering" text that was interrupted.

### Reliability and observability
* **2026-10-08** · **Strict intents + hallucination filter** (blank tokens, whisper junk like "thanks for watching", repetition
  loops) run before the LLM so noise never costs a generation.
* **2026-10-08** · **Server recovery restarts in a new session with project cwd** (`common/servers.py`) so a restart survives the
  parent's Ctrl+C and resolves relative paths; `_ping` is checked before spawning.
* **2026-10-08** · **Two logs:** human console log with ms timestamps and thread names; structured JSONL *trace* of the conversation.
  The E2E judge and `make analyze-trace` read the same trace, so a live session is debuggable exactly like a test.
  The visualiser event log is best-effort (`OSError` swallowed) so telemetry can never break the assistant.
* **2026-10-08** · **Wake word on the ONNX backend** (`WAKEWORD_FRAMEWORK`) to coexist with NumPy 2.

### Testing strategy (`tests/e2e`, `tests/unit`)
* **2026-10-08** · **Real models, simulated world, real time.** Timing behaviours (hold, cooldown, barge-in, narration) cannot be
  tested faster than real time or with mocks. The user is a synthesized, cached voice (varied speaker/speed/level)
  in a room with noise, TV chatter and echo.
* **2026-10-08** · **Fault proxy in front of llama-server** reproduces slow, stalled, hung and erroring models on a fast machine and
  counts disconnects (proves cancellation hangs up).
* **2026-10-08** · **Assert what a person would notice** (dead air, self-interruption, wrong answer to a cut-in); `warn` severity for
  model-quality outcomes. Pure logic (text utils, AEC maths) is covered by fast unit tests that need no servers.

## 8. Before you change X, re-run Y

| if you touch | re-run | because |
|---|---|---|
| silence holds, speculative STT, wake pre-roll | `different_voices`, `wake_in_one_breath`, `followup_window_noise` | easy to trade cut-offs against sluggishness |
| Silero/WebRTC thresholds, frame sizes, duck gain | `barge_in_stop`, `barge_in_while_thinking`, `speaker_echo` | one-word "stop" is the tightest budget |
| `audio/aec.py`, `speaker.py` reference, mic clock | `make test-unit`, `speaker_echo` (repeat ≥ 3) | flaky ≠ fine: fail rate must be 0 of N |
| LLM client / timeouts | `slow_llm_narration`, `very_slow_llm`, `llm_*` | slow vs dead distinction |
| system prompt, language | the tests override the prompt (`tests/e2e/harness.py`); STT is `*.en`, TTS voice is English | see below |
| STT model | `make test-e2e` + compare latency: base.en ≈ 2× tiny.en | accuracy vs responsiveness |
| `Speaker.pause/resume`, `_confirm_interject`, intents / wake stripping | `make test-unit`, then `barge_in_stop`, `barge_in_new_question`, `barge_in_with_wake_phrase`, `non_word_over_answer_resolves`, `cough_*` | the decision after speech over an answer has five outcomes; unit tests pin them, E2E proves the audio side |
| any flag derived from `erle_db` or other smoothed signals | `barge_in_through_echo` ×5 against the previous commit | a flag that follows a noisy signal made barge-in flaky (§6, 2026-10-13) |

**Language mismatch to resolve:** the working-tree system prompt is the German tutor (`config.py`), but Whisper
`tiny.en`, `WHISPER_LANGUAGE=en`, the English Kokoro voice and the English filler phrases are unchanged, so German
speech will be mis-transcribed and spoken with an English voice. A real German-tutor mode needs a multilingual
Whisper model, a German Kokoro voice, German phrase cache entries and a revisit of the hallucination/intent lists.

## 9. Microphone input selection

Dated 2026-10-08 (08:15–08:19). Merged from the former `docs/microphone.md`.

**Quick start**
```bash
make list-devices                         # lists inputs + which one will be selected
PYTHONPATH=src python -W ignore -m audio.devices --test   # records 3 s, prints level
make run-pipeline                         # then say "Hey Jarvis, ..."
```
Config (env vars): `MIC_NAME` (substring, default `Jabra`; empty = PortAudio default), `MIC_INDEX` (PortAudio
index, wins over name; legacy `WAKEWORD_MIC_INDEX` still works), `MIC_REQUIRED=1` (abort if `MIC_NAME` matches
nothing instead of falling back to default).

**What was wrong (2026-10-08).** The app opened PortAudio's *default* input (index 7, ALSA `default`). The conda
env's alsa-lib has no PulseAudio plugin (`libasound_module_pcm_pulse.so` missing), so `default` fell through to
**card 0, the onboard Intel PCH analog input** (`/proc/asound/card0/pcm0c` was RUNNING) — a line/mic jack with
nothing plugged in, i.e. near-silence (rms≈20). PulseAudio's default source *was* the Jabra, but PortAudio never went
through Pulse. The Jabra is PortAudio index 0 (`hw:1,0`, mono, opens fine at 16 kHz). It was first
misdiagnosed as "the machine's mic is digital silence" and the original findings text justified the room
simulator with it (the simulator is still the right tool for repeatable timing tests, but that premise was wrong).

**What changed (2026-10-08)**
- `src/audio/devices.py` (new): lists inputs, resolves index → name → default (warns loudly).
- `src/audio/mic.py`: uses the resolver, logs stream parameters, errors, and a 5 s input level line (WARNING when
  peak < 20 = "near-silence: wrong device, muted, or unplugged").
- `assistant_pipeline.py::_validate_microphone_early`, `wakeword/listen.py`, `vad/silero.py`, `vad/webrtc.py`: same
  resolver; clear exit message if no device.
- `common/config.py`: `microphone_name`, `microphone_required`, `MIC_INDEX`.

**How to confirm the Jabra is used.** Startup log must contain:
```
mic  selected input [0] Jabra BIZ 1500: USB Audio (hw:1,0) ... — reason: name matches MIC_NAME='Jabra'
mic  MICROPHONE OPEN: [0] Jabra BIZ 1500: USB Audio (hw:1,0) @ 16000 Hz mono
```
With `LOG_LEVEL=DEBUG`, `mic level (5 s) ... rms=… peak=…` lines appear; peak should jump to thousands when you
speak. A `falling back to PortAudio default input` warning means the headset was not found (unplugged /
re-enumerated); the index may change after replugging, which is why selection is by name. Cross-check:
`arecord -l`, `pactl list short sources`. The USB headset must be plugged in *before* start; hot-unplug yields
`mic read failed` errors and silence (the loop keeps running) — see §10.

## 10. Open work for robustness

Audit date: 2026-10-13, of the former 12-week `docs/roadmap.md` (deleted; it predated the rewrite and its
instrumentation / reliability / turn-taking goals are met by the work above). Status of every roadmap goal:

| roadmap goal | status 2026-10-13 | where |
|---|---|---|
| per-turn timing, counters, metrics summary | done | trace JSONL, `turns_<date>.jsonl`, `print_summary`, `make analyze-trace` |
| startup validation: ports, mic | done | `_validate_config`, `_validate_microphone_early` |
| startup validation: model files exist | done 2026-10-13 (servers that are not already running) | `Pipeline._validate_files` |
| server readiness polling | done (2 s poll, 120 s per server, clean exit message) | `servers.wait_for_servers` |
| per-service spoken failure phrases + restart | done | `phrases.py`, `Pipeline` recovery |
| watchdog | partial: separate `make watch-servers` process; pipeline restarts a server only after a failed request; neither spawns while the port is bound (2026-10-13) | `servers.watchdog`, `port_in_use` |
| TTS self-trigger reduction | done (AEC + coupling gate), real-room unverified | §7 |
| replayable debugging | done | `replay_stt.py`, `replay_pipeline.py`, `tests/e2e/replay.py` |
| memory window, repeat, instant intents | done; a spoken "I don't remember that" path was **not** built (no decision was recorded either way) | `textutils.py` |
| barge-in polish, earcons | done | §7 |
| graceful shutdown | done (SIGTERM, mic/speaker closed) | `Pipeline.stop` |
| config profiles (`PROFILE`), `make diag` | **open** | – |
| spoken startup health summary | **open** (only logs "ready") | – |
| device-style start (systemd/start script) | **open** | – |
| 30-minute multi-turn stability run | **open** (only a 90 s idle-noise soak exists) | `idle_soak` |
| edge-hardware measurement plan | **open** | – |

Open items, in the order that most improves robustness (updated 2026-10-13 after the fixes in §12):
1. **Real-hardware verification.** Microphone loss/reopen (F17), the pause-then-confirm barge-in, PortAudio stop
   latency and the echo canceller have only been exercised against the simulated room and a fake backend.
2. **German-tutor language stack** (F1): multilingual Whisper installed and selected, a German TTS voice (Kokoro
   v1.0 has none), German phrase cache, German hallucination list. Only intents and a startup warning exist.
3. **Real echo-cancellation measurement**: cancellation of the echo component alone, with simulator ground truth;
   today's ERLE is bounded by the noise floor (§12.2).
4. **Long-session soak with real turns** (30 min, 20–30 interactions): RSS, thread count, open fds, trace size.
5. **Always-on watchdog inside the pipeline** and a spoken startup health summary.
6. **Held-out real recordings** for the STT-confidence and barge-in thresholds (F5).
7. Config profiles, `make diag`, a systemd/launch script and an edge-device measurement protocol, when a target
   device exists.

Documentation hygiene (2026-10-13): `docs/e2e_findings.md` and `docs/microphone.md` were merged into this file and
`docs/roadmap.md` was removed once audited; README.md was rewritten as a snapshot of the current system, without
history. The findings text had been appended to README.md by mistake, which duplicated it — keep history here only.

## 11. Critical review of 2026-10-13

Scope: the whole uncommitted `repurpose_as_german_tutor` working tree (code, tests and these docs), reviewed by
reading the code, running the unit tests and probing the text logic. Each finding has an ID; what was done about each one,
and what did not work, is in §12. Nothing below was known to be wrong when the docs in §1–§10 were
written; several statements in those sections are corrected in §12.

### 11.1 The product does not do what it claims

| id | finding | fix required |
|---|---|---|
| F1 | German-tutor prompt, but Whisper `tiny.en` + `WHISPER_LANGUAGE=en`, English Kokoro voice, English phrases, English-only intents ("Danke"/"Tschüss" match nothing) | German intents; startup consistency check that warns loudly; document the TTS limitation (Kokoro v1.0 has no German voice) |
| F2 | "37 scenarios, 0 failed checks" is not backed by any run: the last full run (`reports/e2e/20261008_024726`) failed `barge_in_stop`; only an 8-scenario subset was clean afterwards | re-run the full suite, publish the real table |
| F3 | E2E numbers predate the 08:15 edits to pipeline / config / silero / webrtc | covered by the F2 re-run |
| F4 | simulated user is Kokoro TTS, echo is pure delay+gain, output device is simulated: tests are easier than reality; stop latency never touched PortAudio | document as limitation; keep a real-hardware checklist |
| F5 | thresholds (`STT_MIN_LOGPROB`, 190 ms, …) were tuned on the same synthetic scenarios that evaluate them | document; real-recording validation listed as open work |
| F6 | "0 of 9 echo runs" bounds the failure rate only to ≈ 28 % at 95 % confidence | document with the bound; repeat more runs |

### 11.2 Logic bugs

| id | finding | fix required |
|---|---|---|
| F7 | `is_hallucination("Okay.")` and `("Bye.")` are True (`"okay."` survives the `- {"okay","ok"}` exclusion) and the hallucination check runs before intents, so cut-in "okay" never stops and a bare "Bye." is dropped | exclude by normalised form; let confident stop/sleep intents win over the junk list |
| F8 | "yes / sure / right / fine / okay" over the assistant mean *stop* even when it just asked a question | treat acknowledgements as answers when the last reply ended in a question |
| F9 | time/date patterns are unanchored: "What time is it in Tokyo?" → local time; "date of the French revolution" → today's date | anchor to near-whole utterances |
| F10 | wake-phrase stripping runs on every transcript and its fuzzy list contains real words ("service", "harvest", "travis") | strip only for captures that contained the wake phrase |
| F11 | barge-in over answer audio kills the answer for good with no STT confirmation and no resume | pause (fade out, hold position) → confirm by STT → stop, or resume if it was noise |
| F12 | AEC gives up permanently after 8 failed delay estimates; comment says it keeps retrying | re-arm estimation after a stretch of far-end audio |
| F13 | "converged" = 8 dB ERLE and ends the cautious barge-in margin | separate, higher trust threshold for dropping the margin |
| F14 | one Silero streaming state is shared by capture, barge-in and re-entry on non-contiguous audio | reset at the start of each independent use |
| F15 | recovery can spawn duplicates (untracked Makefile-started servers; watchdog + pipeline both restarting); server output goes to DEVNULL | never spawn while the port is bound; log server output to files |
| F16 | `SERVERS` hard-codes 127.0.0.1:8080/81/82 and ignores `*_SERVER_URL` | derive from config |
| F17 | lost microphone → endless silence, no reopen, no notice | reopen by name with back-off; say so once |

### 11.3 Privacy, robustness, maintainability

| id | finding | fix required |
|---|---|---|
| F18 | `DEBUG_SAVE_WAV=1` and full transcripts/responses in trace + turn logs by default, never pruned | opt-in; transcripts redacted unless `LOG_TRANSCRIPTS=1`; prune debug WAVs |
| F19 | `_log_turn` in a `finally` has no error handling | wrap |
| F20 | a newer utterance silently replaces an unprocessed older one | keep (rare); document |
| F21 | `[interrupted]` marker is stored in history and fed back to the model | store the spoken part with an ellipsis only |
| F22 | ducking gain reset to 1.0 at every new audio item | keep ducking across items |
| F23 | 1,240-line pipeline with only two test files; endpoint logic untestable | extract pure endpoint classification; add tests for speaker, LLM reader, servers, devices |
| F24 | absolute `/home/puli/...` paths; `make run-pipeline` needs `nc` which `install-system` does not install | env overrides in launch scripts; add netcat to apt list |
| F25 | `serve_gemma.sh` sets no `-c` / `-np` | explicit, overridable context size and single slot |
| F26 | `wait_for_servers` shares one deadline for three servers and aborts with a traceback | per-server deadline, clean message |
| F27 | open work from §10: model-file validation at start | validate files of servers that are not running |
| F28 | (added during the fix work) simulated echo only reaches ≈ 9–11 dB ERLE, suggesting a weak or misaligned filter | **retracted** — ERLE is bounded by the room noise floor, see §12.2 |

### 11.4 Faults in the documentation itself

| id | finding | fix required |
|---|---|---|
| D1 | "Rejected: …" alternatives in §7 were inferred from the code, not recorded at the time | label them as inferred |
| D2 | §10 says the "I don't remember" path was "judged low value"; nobody judged that | reword |
| D3 | §6 says the mic was misdiagnosed "for a day" (speculation) | reword |
| D4 | §7 "stopping an answer is cheap to undo" is false: nothing resumed it | replace with the pause/resume design (F11) |
| D5 | §4 "37 scenarios, 0 failed" (F2) | replace with the re-run result |
| D6 | every entry is dated 2026-10-08 from file mtimes; it says when a change landed, not when the lesson was learned | state this once, keep dates |

## 12. Fixes applied for the 2026-10-13 review

All dated 2026-10-13. Unit tests: 137 pass (`make test-unit`, 73 new). The E2E result of the final code is in §12.3.

### 12.1 What changed

| ids | change | files | verified by |
|---|---|---|---|
| F7 | junk list compared on the normalised form and with plain acknowledgements excluded; a confident stop/sleep intent beats the junk list ("Bye." after a goodbye is a goodbye, not whisper noise) | `pipeline/textutils.py`, `pipeline/assistant_pipeline.py::_is_noise` | `test_review_fixes_text.py` |
| F8 | `classify_intent(..., awaiting_answer=)`: yes/okay/no/sure over a reply that ended in "?" are answers, not "stop" | `textutils.py`, pipeline `_awaiting_answer` | unit tests |
| F9 | time/date patterns anchored to the whole utterance (only polite/temporal tails allowed) | `textutils.py` | unit tests |
| F10 | `Utterance.wakeish`; wake phrase stripped only for captures that contained it; endpoint logic moved to the pure `textutils.endpoint_class` and shares that rule | `textutils.py`, pipeline | unit tests |
| F1 | German stop / goodbye / thanks / greeting / repeat / time / date intents and German cut-ins and dangling words; startup `LANGUAGE STACK` warning when prompt, Whisper model/language and TTS disagree. **Not solved:** a multilingual Whisper model must be installed by the user and Kokoro v1.0 has no German voice | `textutils.py`, pipeline `_check_language_stack` | unit tests; startup log |
| F11, F22 | `Speaker.pause()/resume()`: fade out in one 20 ms chunk, hold position, nothing discarded; `stop()` clears pause and duck; ducking is carried across items instead of reset to 1.0; unresolved pause continues after `BARGE_IN_PAUSE_TIMEOUT_S`. Barge-in over answer audio pauses and starts an "interject" capture; `_confirm_interject` resumes on noise, stops on a muffled short word, stops-and-answers on speech | `audio/speaker.py`, pipeline | `test_speaker.py`; E2E `barge_in_*`, `non_word_over_answer_resumes` |
| F12 | AEC delay estimation re-armed after ≈ 18 s of far-end audio instead of staying bypassed forever | `audio/aec.py` | `test_aec.py` |
| F13 | `EchoCanceller.trusted` (latched like `converged`) + `AEC_TRUSTED_ERLE_DB`. **Default kept at 8 dB (= no behaviour change)** — see §12.2 and the latch story in §6 | `audio/aec.py`, config | E2E |
| F14 | Silero streaming state reset at the start of a barge-in session and of each follow-up trigger run | pipeline | – (covered by E2E only) |
| F15 | `port_in_use()`: never spawn a server while its port is bound (pipeline recovery and watchdog); server output logged to `debug_logs/server_<name>.log` | `common/servers.py`, pipeline | `test_servers_devices.py` |
| F16 | `SERVERS` built from `LLM/STT/TTS_SERVER_URL` | `common/servers.py` | unit test |
| F26 | per-server startup deadline; clean message + exit instead of a traceback | `common/servers.py`, pipeline | – |
| F17 | `PyAudioMic` counts consecutive failures, reopens by name every 2 s, calls `on_status("lost"/"restored")`; pipeline says "I lost my microphone…" / "My microphone is back." from the phrase cache | `audio/mic.py`, `tts/phrases.py`, pipeline | – (needs a real unplug) |
| F18 | `DEBUG_SAVE_WAV` default off, pruned to `DEBUG_MAX_WAVS`; trace / turn log / console redact transcripts and responses unless `LOG_TRANSCRIPTS=1` (the E2E harness sets it) | config, pipeline | E2E (evaluator still reads text) |
| F19 | turn log write failure is logged, not raised from `finally` | pipeline | – |
| F21 | interrupted answers are remembered as the spoken text followed by " …" instead of a bracketed marker | pipeline | – |
| F23 | new unit tests: text fixes, speaker pause/resume/stop/duck, barge-in confirm decisions with a stubbed STT, LLM client against a fake SSE server (stream, error, first-token timeout, stall, cancel hangs up), servers/devices, AEC re-arm and trust latch | `tests/unit/` | 137 pass |
| F24, F25 | `serve_gemma.sh`: `--ctx-size ${LLM_CTX:-4096}`, `--parallel ${LLM_PARALLEL:-1}` (llama.cpp's defaults are "from the model" and "auto"; checked with `llama-server --help`), model/binary/port overridable; `serve_whisper.sh` binary/port overridable; `netcat-openbsd` added to `install-system` | scripts, Makefile | `bash -n`; takes effect on next server start |
| F27 | `_validate_files()`: missing Whisper/Kokoro files are reported before start for servers that are not already running | pipeline | – |
| F20 | unchanged by design: a newer capture replaces an unprocessed older one (traced as `utterance_replaced`) | – | – |
| D1–D6 | §6, §7, §10 corrected as described in §11.4 | this file | – |

### 12.2 What did not work, and what the review itself got wrong

* **F13 was a wrong diagnosis.** Raising the "trusted" ERLE threshold from 8 to 12 dB made `barge_in_through_echo`
  fail (a real "Stop!" over echo was never detected): in the simulated room ERLE hovers at 6–11 dB, so the cautious
  margin (×1.6 level, ×2 duration) never switched off. ERLE is measured against everything in the microphone,
  including the room's noise floor (0.003–0.008 RMS in the scenarios), so it is *bounded by the noise*, not a measure
  of filter quality. The real guard is the residual-based echo floor in the barge-in detector. The review's F28
  ("AEC only reaches ≈ 9 dB, probably misaligned") is therefore **not established**; the correct statement is "ERLE as
  logged cannot distinguish a weak filter from a noisy room". A proper measurement (cancellation of the echo
  component only, with ground truth from the simulator) is open work.
* **The E2E scenario `barge_in_stop` measured the wrong thing after F11.** It took the end of the audio item as
  "fell silent"; with pause-then-confirm the item ends when STT has decided (≈ 1.5 s) although the room was silent
  after ≈ 0.2 s. The check now uses the `speaker_pause` / `speaker_stop` event (+ 60 ms fade), and accepts either
  an intent-level stop or a muffled-word stop. That is a change of the test, made because the observable behaviour
  changed on purpose; it is recorded here so it is not mistaken for moving a goalpost silently.
* **F10 introduced a regression that only E2E caught.** Restricting wake-phrase stripping to captures flagged
  `wakeish` broke `barge_in_with_wake_phrase`: "Hey Jarvis, stop." said over an answer is an *interject* capture, and
  wake detection only ran for follow-up captures, so the text reached the LLM unstripped ("I am not Jarvis…"). Fixed
  twice over: wake detection now also runs in interject captures, and a **strict** strip (explicit "hey/hi/okay/hello"
  + the name) is applied to every capture that was not flagged, so a stop with a spoken wake phrase never depends on
  the wake model having heard it. Unit tests cover the strict form ("Service animals…" is left alone).
* **`non_word_over_answer_resumes` assumed what Whisper hears.** It expected "Hmm." to be transcribed as noise;
  tiny.en heard "Come in.", which the (documented) rule "a brief unrecognised word over our answer is a stop" turned
  into a silent stop — correct behaviour, wrong test. The scenario became `non_word_over_answer_resolves` (a pause
  must be resolved by resume or stop, and the sound must never reach the LLM), and the resume / muffled-stop /
  speech decisions are covered deterministically in `tests/unit/test_barge_in_confirm.py` with a stubbed STT.
* **A first full run was discarded.** It was started before the F13 revert; it showed 3 failing checks in 2
  scenarios (`barge_in_stop`, `barge_in_through_echo`) at the time it was stopped; a second full run (36/38, with
  `barge_in_with_wake_phrase` and the Hmm scenario failing, both described above) was replaced as well, so the
  table in §12.3 is the run of the code that is committed.
* **Limitations that cannot be fixed in this repository** (F4–F6): the simulated user is the same TTS engine as the
  assistant, the echo path is delay + gain, and the thresholds were tuned on the scenarios that evaluate them.
  Confidence in real rooms needs real recordings (`tests/e2e/replay.py`) and a loudspeaker session with the
  Jabra unplugged. "0 of 9 echo runs" bounds the failure rate only to ≈ 28 % (95 %); do not read more into it.

### 12.3 Results of the code that is committed

Run `reports/e2e/20261013_010834` (full suite, 38 scenarios, final code; the report folders are git-ignored, the
numbers are copied here): **37 of 38 scenarios without a hard failure; 1 failed check; 1 expected warning.**

| item | result |
|---|---|
| `speaker_echo` | 1 failed check: *no self-interruption from own voice — 1 barge-in*. The trace shows the new design absorbing it: echo triggered a pause at 13.4 s, STT/endpoint found no speech, the answer resumed after 1.8 s and completed (before F11 the answer would have been killed). It is still an audible 1.8 s gap, so it counts. Repeat ×8: 8/8 pass on this code, 8/8 on the pre-fix commit; 1 failure in 9 runs of the final code overall. Not distinguishable from a rare pre-existing flake (95 % upper bound on the failure rate ≈ 30 %) |
| `speaker_echo_without_aec` | warning, as designed (baseline shows what AEC buys); its "settles to idle" checks are warnings since the baseline is a self-feeding loop (`allow_unsettled`) |
| `barge_in_through_echo` | 5/5 pass in the final repeat. Before the `trusted` latch fix: 2 failures in 7 runs (pre-change commit: 0 of 6) — see §6 |
| `wake_in_one_breath` | 5/5 without hard failure, 3 warnings; one earlier run failed on STT hearing "2 plus 3" as "Topless 3" / "Top Boss 3" (tiny.en). Model quality, not pipeline |
| everything else (34 scenarios) | pass |

Timing, measured from the end of the user's audio (same method as §4; clips carry ≈ 150 ms of trailing TTS pad):

| metric | 2026-10-08 (§4) | 2026-10-13 |
|---|---|---|
| end of speech detected, median | 0.81 s | 0.82 s |
| first word of a normal answer, median | 2.1 s | 2.1 s |
| slow model (12 s TTFT) first word, max | 18 s | 16.5 s |
| time until the room is silent after "stop" starts | 0.40 s | 0.39 s (pause + 60 ms fade) |
| max dead air in a normal turn | – | 4 s |

Run history of this review, so the table above can be audited: (1) full run with the 12 dB trust threshold —
stopped at 29/37, `barge_in_stop` and `barge_in_through_echo` failing; (2) full run after reverting it — 36/38,
`barge_in_with_wake_phrase` (real regression from F10) and the Hmm scenario (wrong test) failing; (3) full run —
35/38, `wake_in_one_breath` (STT), `barge_in_through_echo` (flaky, trust latch), baseline settle checks;
(4) the committed code — above. The unit suite is 137 tests, ~12 s.

What this does **not** show: nothing here ran on a real microphone/loudspeaker, with a real person, or with German
speech. Items under §10 remain open for those reasons.

