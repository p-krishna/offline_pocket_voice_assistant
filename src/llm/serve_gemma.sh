#!/bin/bash
"/home/puli/projects/local llm/llama.cpp.mtp/build/bin/llama-server" \
  --no-ui \
  -m "/home/puli/projects/local llm/models.gguf/gemma-4-E2B-it-UD-IQ2_M.gguf" \
  --reasoning off \
  --predict 150 \
  --host 127.0.0.1 \
  --port 8080