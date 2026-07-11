#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-wsl-dev}"
REMOTE_VENV="${REMOTE_VENV:-/home/caius/lead-3d/venv/bin/activate}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/caius/projects/LifeOS-Agent}"
REMOTE_MINIMIND="${REMOTE_MINIMIND:-/home/caius/minimind}"
REMOTE_CHECKPOINT="${REMOTE_CHECKPOINT:-/home/caius/minimind/out/lifeos_agent_best_768.pth}"
PROMPT="${1:-我今天应该做什么？}"

ssh "$REMOTE_HOST" "source \"$REMOTE_VENV\" && cd \"$REMOTE_PROJECT\" && python lifeos_agent/main.py --minimind_repo \"$REMOTE_MINIMIND\" --tokenizer_path \"$REMOTE_MINIMIND/model\" --checkpoint_path \"$REMOTE_CHECKPOINT\" --prompt '$PROMPT'"
