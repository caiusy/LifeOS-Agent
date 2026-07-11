#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-wsl-dev}"
REMOTE_VENV="${REMOTE_VENV:-/home/caius/lead-3d/venv/bin/activate}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/caius/projects/LifeOS-Agent}"
REMOTE_MODEL="${REMOTE_MODEL:-/home/caius/models/minimind-3}"
PROMPT="${1:-我今天应该做什么？}"

ssh "$REMOTE_HOST" "source \"$REMOTE_VENV\" && cd \"$REMOTE_PROJECT\" && python lifeos_agent/main.py --hf_model_path \"$REMOTE_MODEL\" --prompt '$PROMPT'"
