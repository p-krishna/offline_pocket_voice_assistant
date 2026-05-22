# Wake-word setup

This document records the current known-good wake-word development setup for the Linux desktop prototype of the offline voice assistant project. The current desktop LLM reference for the broader stack is `gemma-4-E2B-it-UD-IQ2_M.gguf`.

## Repo location

Store the wake-word setup files here:

```text
configs/wakeword/requirements.txt
configs/wakeword/environment.yml
configs/wakeword/versions.lock.json
docs/setup/wakeword.md
```

## Known-good environment

- OS: Ubuntu 22.04.5 LTS 64-bit
- Architecture: x86_64
- Python: 3.11
- Numpy: `<2`
- openWakeWord: `0.6.x`
- speexdsp-ns: installed successfully from PyPI

## Important notes

- Python 3.14 did not work for this stack because `tflite-runtime` was unavailable for that interpreter version in the tested environment.
- Installing old `speexdsp_ns` wheel URLs directly from GitHub caused wheel compatibility or 404 errors.
- The working path is to install `speexdsp-ns` from PyPI inside the Python 3.11 environment.
- Noise suppression is optional for first-pass wake-word validation.

## System packages

Install system dependencies first:

```bash
sudo apt-get update
sudo apt-get install -y libspeexdsp-dev swig portaudio19-dev
```

## Conda environment

```bash
conda env create -f configs/wakeword/environment.yml
conda activate wakeword311
```

## Pip environment alternative

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r configs/wakeword/requirements.txt
```

## Validation checks

```bash
python --version
python -c "import openwakeword, numpy; print('openwakeword ok'); print(numpy.__version__)"
pip show speexdsp-ns
```

## Suggested git policy

- Commit `requirements.txt`, `environment.yml`, `versions.lock.json`, and this setup document.
- Do not commit virtual environments, downloaded model binaries, or audio recordings.
- Keep custom wake-word models under a separate ignored local directory unless you want versioned releases.
