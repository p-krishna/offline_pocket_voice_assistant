# offline_pocket_voice_assistant
Pocket-sized offline voice assistant for visually impaired users — built for privacy, portability, and low cost. It listens, understands, and speaks back locally with no cloud dependency, using a compact hardware setup designed for everyday carry.  This is a small, fully local voice assistant pipeline for edge-style devices. It listens continuously, filters silence, detects a wake word, and then uses speech end detection to know when the user has finished speaking.

## What this does

This project is organized as a simple local audio pipeline:

1. **WebRTC VAD** runs continuously and filters obvious silence.
2. **openWakeWord** runs only when speech is present.
3. **Silero VAD** runs after wake-word detection and decides when the utterance ends.

The goal is to keep the code small, readable, and easy to debug while still matching the behavior of a real edge assistant.

## Quick start

```bash
make install
make run-pipeline
```

If you want to check the config quickly:

```bash
make check
```

## Architecture

| File | What it does | Example |
|---|---|---|
| `src/common/config.py` | Stores the small runtime settings for the assistant. | `wakeword="hey_jarvis"`, `sample_rate=16000` |
| `src/vad/webrtc.py` | Continuously detects speech vs silence and prints only on changes. | `silence -> speech after 2.10s` |
| `src/wakeword/listen.py` | Runs openWakeWord and triggers when the wake word is detected. | `WakeWord detected: hey_jarvis score=0.621` |
| `src/vad/silero.py` | Checks speech after wake word and ends the utterance on silence. | `speech -> silence after 1.84s` |
| `src/pipeline/assistant_pipeline.py` | Main orchestrator: WebRTC -> openWakeWord -> Silero. | `Pipeline: WebRTC -> openWakeWord -> Silero` |
| `Makefile` | Short commands for install, run, and quick checks. | `make run-pipeline` |

## Repo layout

```text
LICENSE
Makefile
README.md
configs/
docs/
src/
```

## How it behaves

- WebRTC runs all the time and filters out obvious silence first.
- openWakeWord runs only when speech is present.
- Silero runs after wake-word detection and decides when the user finished speaking.
- Each module prints only when its state changes, so logs stay clean.

## Why this structure

This layout is meant to stay easy to understand from top to bottom:
- `config.py` keeps settings in one place.
- `webrtc.py`, `listen.py`, and `silero.py` each do one job.
- `assistant_pipeline.py` is the single entry point that connects them.

That makes debugging much easier on a small edge device, because each stage can be tested separately before combining the full loop.

## Simple mental model

Think of it like this:

1. **WebRTC** asks, “Is someone speaking?”
2. **openWakeWord** asks, “Did they say the wake word?”
3. **Silero** asks, “Are they done speaking now?”

## Notes

This project is intentionally kept small and local-first. The current focus is a working edge-style voice pipeline, not a large feature set.
