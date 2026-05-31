PYTHON ?= python
PIP    ?= pip

.PHONY: install install-system install-python-deps serve-llm serve-stt serve-tts \
        serve-all watch-servers \
        list-devices check-wakeword check-tts check-llm check-stt run-pipeline \
		kill-servers

# ── System dependencies ───────────────────────────────────────────────────────
install-system:
	sudo apt-get update
	sudo apt-get install -y libspeexdsp-dev swig portaudio19-dev

# ── Python dependencies ───────────────────────────────────────────────────────
install-python-deps:
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r configs/requirements.txt

# ── Server launchers (run in background) ─────────────────────────────────────
serve-llm:
	/bin/bash src/llm/serve_gemma.sh

serve-stt:
	/bin/bash src/stt/serve_whisper.sh

serve-tts:
	PYTHONPATH=src $(PYTHON) src/tts/serve_kokoro.py

# ── Standalone checks (for unit testing each stage) ───────────────────────────
list-devices:
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