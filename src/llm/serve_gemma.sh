#!/bin/bash
# Start llama-server as the persistent LLM service (default port 8080).
# Everything machine-specific can be overridden from the environment:
#   LLAMA_SERVER_BIN  path to llama-server       LLM_MODEL_PATH  path to the .gguf
#   LLM_PORT          listen port (keep LLM_SERVER_URL in sync)
#   LLM_CTX           context tokens (default 4096: system prompt + 3 turns + a 150-token answer fit easily;
#                     leaving it unset lets llama.cpp take the model's training context and allocate a
#                     much larger KV cache than a single-user assistant ever needs)
#   LLM_PARALLEL      parallel slots (default 1: one user, and one slot keeps the whole context for it and
#                     the prompt cache warm)
"${LLAMA_SERVER_BIN:-/home/puli/projects/local llm/llama.cpp/build/bin/llama-server}" \
  --no-webui \
  -m "${LLM_MODEL_PATH:-/home/puli/projects/local llm/models.gguf/gemma-4-E2B-it-UD-IQ2_M.gguf}" \
  --reasoning off \
  --predict 150 \
  --ctx-size "${LLM_CTX:-4096}" \
  --parallel "${LLM_PARALLEL:-1}" \
  --host 127.0.0.1 \
  --port "${LLM_PORT:-8080}"
