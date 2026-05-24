PYTHON ?= python
PIP ?= pip

.PHONY: install install-system run run-no-vad run-ns list-devices check

install-system:
	sudo apt-get update
	sudo apt-get install -y libspeexdsp-dev swig portaudio19-dev

install-python-deps:
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r configs/requirements.txt

check-python-deps:
	$(PYTHON) --version
	$(PYTHON) -c "import openwakeword, numpy, kokoro-onnx; print('imports ok')"

serve-llm:
	/bin/bash src/llm/serve_gemma.sh

list-devices:
	PYTHONPATH=src $(PYTHON) src/wakeword/listen.py --list-devices

check-wakeword:
	PYTHONPATH=src/wakeword $(PYTHON) src/wakeword/listen.py

check-tts:
	PYTHONPATH=src $(PYTHON) src/tts/kokoro.py

check-llm:
	PYTHONPATH=src $(PYTHON) src/llm/gemma.py

check-stt:
	PYTHONPATH=src $(PYTHON) src/stt/whisper_cpp.py src/stt/stt_test_sample.wav

run-pipeline:
	@echo "Checking if LLM server is running on 127.0.0.1:8080..."
	@nc -z 127.0.0.1 8080 || (echo "LLM server is not running.\nTrying to start it..."; make serve-llm & sleep 5; echo "LLM server should be running now. Proceeding with the pipeline...")
	PYTHONPATH=src python src/pipeline/assistant_pipeline.py
