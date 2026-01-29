#!/usr/bin/env bash
set -euo pipefail

APP_DIR="..."
VENV_BIN=".../venv-llm"
HOST="..."
PORT="5001"

export PATH="$VENV_BIN:$PATH"

# llama3.2:3b is used in this project locally
export OLLAMA_MODEL_NAME="llama3.2:3b-instruct-q4_K_M"

cd "$APP_DIR"

exec uvicorn llm_service:app --host "$HOST" --port "$PORT"