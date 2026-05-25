#!/bin/bash

# Start whisper-server as a persistent STT service on port 8081.
# Model is loaded once; each request sends a WAV file via multipart HTTP.
/home/puli/projects/whisper/whisper.cpp/build/bin/whisper-server \
  -m /home/puli/projects/whisper/whisper.cpp/models/ggml-tiny.en.bin \
  --host 127.0.0.1 \
  --port 8081 \
  --inference-path /inference \
  --threads 4 \
  --convert
