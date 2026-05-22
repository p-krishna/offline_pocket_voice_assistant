PYTHON ?= python
PIP ?= pip

.PHONY: install install-system run run-no-vad run-ns list-devices check

install-system:
	sudo apt-get update
	sudo apt-get install -y libspeexdsp-dev swig portaudio19-dev

install:
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r configs/wakeword/requirements.txt

run:
	PYTHONPATH=src/wakeword $(PYTHON) src/wakeword/listen.py

run-no-vad:
	WAKEWORD_ENABLE_VAD=0 PYTHONPATH=src/wakeword $(PYTHON) src/wakeword/listen.py

run-ns:
	WAKEWORD_ENABLE_SPEEX_NS=1 PYTHONPATH=src/wakeword $(PYTHON) src/wakeword/listen.py

list-devices:
	PYTHONPATH=src/wakeword $(PYTHON) src/wakeword/listen.py --list-devices

check:
	$(PYTHON) --version
	$(PYTHON) -c "import openwakeword, numpy; print('imports ok')"
