PYTHON ?= python
PIP    ?= pip

.PHONY: install install-system install-python-deps serve-llm serve-stt serve-tts \
        serve-all watch-servers \
        list-devices check-wakeword check-tts check-llm check-stt run-pipeline \
		kill-servers download-wakeword-models test test-unit test-e2e test-e2e-core test-e2e-one e2e-list analyze-trace

# ── System dependencies ───────────────────────────────────────────────────────
install-system:
	sudo apt-get update
	sudo apt-get install -y libspeexdsp-dev swig portaudio19-dev

# ── Python dependencies ───────────────────────────────────────────────────────
install-python-deps:
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r configs/requirements.txt

# One-time download of the openWakeWord model files (feature extractors + hey_jarvis)
download-wakeword-models:
	$(PYTHON) -c "import openwakeword; openwakeword.utils.download_models(['hey_jarvis'])"

# ── Server launchers (run in background) ─────────────────────────────────────
serve-llm:
	/bin/bash src/llm/serve_gemma.sh

serve-stt:
	/bin/bash src/stt/serve_whisper.sh

serve-tts:
	PYTHONPATH=src $(PYTHON) src/tts/serve_kokoro.py

# ── Standalone checks (for unit testing each stage) ───────────────────────────
list-devices:
	PYTHONPATH=src $(PYTHON) -W ignore -m audio.devices
	PYTHONPATH=src $(PYTHON) src/wakeword/listen.py --list-devices

check-wakeword:
	PYTHONPATH=src $(PYTHON) src/wakeword/listen.py

check-llm:
	PYTHONPATH=src $(PYTHON) src/llm/gemma.py

check-stt:
	PYTHONPATH=src $(PYTHON) src/stt/whisper_cpp.py src/stt/stt_test_sample.wav

check-tts:
	PYTHONPATH=src $(PYTHON) src/tts/kokoro.py

# ── Full pipeline ─────────────────────────────────────────────────────────────
run-pipeline:
	@echo "Checking servers..."

	@# LLM server on 8080
	@nc -z 127.0.0.1 8080 || ( \
		echo "LLM server not running — starting..."; \
		/bin/bash src/llm/serve_gemma.sh & \
	)

	@# STT server on 8081
	@nc -z 127.0.0.1 8081 || ( \
		echo "STT server not running — starting..."; \
		/bin/bash src/stt/serve_whisper.sh & \
	)

	@# TTS server on 8082
	@nc -z 127.0.0.1 8082 || ( \
		echo "TTS server not running — starting..."; \
		PYTHONPATH=src $(PYTHON) src/tts/serve_kokoro.py & \
	)

	@# The pipeline itself calls wait_for_servers() and will block
	@# until all three servers respond — no fixed sleep needed.
	@echo "Servers launched. Pipeline will wait until all are ready..."
	PYTHONPATH=src $(PYTHON) src/pipeline/assistant_pipeline.py


# start all three servers in the background (no pipeline)
serve-all:
	@echo "Starting all servers in background..."
	/bin/bash src/llm/serve_gemma.sh &
	/bin/bash src/stt/serve_whisper.sh &
	PYTHONPATH=src $(PYTHON) src/tts/serve_kokoro.py &
	@echo "Done. Use 'make watch-servers' to monitor them."


# run the watchdog — restarts any server that goes down
watch-servers:
	PYTHONPATH=src $(PYTHON) src/common/servers.py

check-servers:
	nc -z 127.0.0.1 8080 && echo "LLM up" || echo "LLM down" && nc -z 127.0.0.1 8081 && echo "STT up" || echo "STT down" && nc -z 127.0.0.1 8082 && echo "TTS up" || echo "TTS down"

kill-servers:
	@echo "Killing all servers..."
	-@pkill -f llama-server   && echo "LLM server killed."  || echo "LLM server not running."
	-@pkill -f whisper-server && echo "STT server killed."  || echo "STT server not running."
	-@pkill -f serve_kokoro   && echo "TTS server killed."  || echo "TTS server not running."
	@echo "Done."

## Replay a saved WAV through STT only
## Usage: make replay-stt WAV=debug_audio/utterance_1234.wav
replay-stt:
	python src/tools/replay_stt.py $(WAV)

## Replay a saved WAV through STT + LLM
## Usage: make replay-pipeline WAV=debug_audio/utterance_1234.wav
replay-pipeline:
	python src/tools/replay_pipeline.py --llm $(WAV)

## Create the 5 named reference test clips
create-test-clips:
	python src/tools/create_test_clips.py


# ── Automated testing ─────────────────────────────────────────────────────────
## Fast pure-logic tests (no servers, no audio hardware)
test-unit:
	PYTHONPATH=src:. $(PYTHON) -m pytest tests/unit -q

## Human-like end-to-end evaluation: a simulated person talks to the REAL pipeline
## (real wake word / VAD / Whisper / Gemma / Kokoro) inside a simulated room.
## Needs the three servers up (make serve-all). Report: reports/e2e/<timestamp>/report.md
test-e2e:
	PYTHONPATH=src:. $(PYTHON) -W ignore tests/e2e/run_e2e.py -v

## Subset, e.g. make test-e2e-core  or  make test-e2e S=barge_in_stop,speaker_echo
test-e2e-core:
	PYTHONPATH=src:. $(PYTHON) -W ignore tests/e2e/run_e2e.py -v --tag core

test-e2e-one:
	PYTHONPATH=src:. $(PYTHON) -W ignore tests/e2e/run_e2e.py -v -s $(S)

## Summarise the newest live-session trace (debug_logs/trace_*.jsonl) or a given one: make analyze-trace T=path
analyze-trace:
	PYTHONPATH=src $(PYTHON) src/tools/analyze_trace.py $(T)

e2e-list:
	PYTHONPATH=src:. $(PYTHON) -W ignore tests/e2e/run_e2e.py --list

test: test-unit test-e2e
