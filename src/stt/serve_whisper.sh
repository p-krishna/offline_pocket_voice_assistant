#!/bin/bash

# Start whisper-server as a persistent STT service on port 8081.
# Model is loaded once; each request sends a WAV file via multipart HTTP.
# WHISPER_MODEL=<path to ggml-*.bin> overrides the model (ggml-base.en.bin: ~6% vs ~10% word error on
# short commands in our synthetic benchmark, but ~680 ms vs ~300 ms per request).
# WHISPER_SERVER_BIN overrides the binary, STT_PORT the port (keep STT_SERVER_URL in sync).
"${WHISPER_SERVER_BIN:-/home/puli/projects/whisper/whisper.cpp/build/bin/whisper-server}" \
  -m "${WHISPER_MODEL:-/home/puli/projects/whisper/whisper.cpp/models/ggml-tiny.en.bin}" \
  --host 127.0.0.1 \
  --port "${STT_PORT:-8081}" \
  --inference-path /inference \
  --threads 4 \
  --suppress-nst \
  --no-language-probabilities \
  --convert
