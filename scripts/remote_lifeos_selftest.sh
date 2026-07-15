#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-ubuntu-ts}"
REMOTE_VENV="${REMOTE_VENV:-/home/caius/minimind/.venv-lifeos/bin/activate}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/caius/projects/LifeOS-Agent}"
REMOTE_MINIMIND="${REMOTE_MINIMIND:-/home/caius/minimind}"
REMOTE_CHECKPOINT="${REMOTE_CHECKPOINT:-/home/caius/minimind/out/lifeos_agent_production_768.pth}"

prompts=(
  "我之前学 SFTDataset 学到哪了？"
  "我今天应该做什么？"
  "17.66 涨停价是多少？"
  "你好，简单介绍一下你自己"
)

for prompt in "${prompts[@]}"; do
  echo "===== CASE: $prompt ====="
  ssh "$REMOTE_HOST" "source \"$REMOTE_VENV\" && cd \"$REMOTE_PROJECT\" && python lifeos_agent/main.py --minimind_repo \"$REMOTE_MINIMIND\" --tokenizer_path \"$REMOTE_MINIMIND/model\" --checkpoint_path \"$REMOTE_CHECKPOINT\" --prompt '$prompt'"
  echo
done
