#!/usr/bin/env bash
# Queue PPO, GRPO, and the guarded LifeOS Agent RL experiment serially.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-ubuntu-ts}"
REMOTE_VENV="${REMOTE_VENV:-/home/caius/minimind/.venv-lifeos/bin/activate}"
REMOTE_MINIMIND="${REMOTE_MINIMIND:-/home/caius/minimind}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/caius/projects/LifeOS-Agent}"
REWARD_MODEL="${REWARD_MODEL:-/home/caius/internlm2-1_8b-reward}"

# Stages are serial because concurrent policy/reference/reward models exceed a
# single RTX 3090 Ti's safe memory budget. Every output has a distinct name.
ssh "$REMOTE_HOST" "
set -euo pipefail
source '$REMOTE_VENV'
cd '$REMOTE_MINIMIND/trainer'
OUT='$REMOTE_MINIMIND/out'

while pgrep -f '[t]rain_dpo.py.*lifeos_agent_dpo_v1' >/dev/null; do sleep 60; done
test -f \"\$OUT/lifeos_agent_dpo_v1_768.pth\"
test -s '$REMOTE_PROJECT/dataset/lifeos_agent_rl.jsonl'
python '$REMOTE_PROJECT/scripts/patch_minimind_agent_rl.py' '$REMOTE_MINIMIND/trainer/train_agent.py'

while [ ! -f '$REWARD_MODEL/config.json' ] \
  || [ ! -f '$REWARD_MODEL/model-00001-of-00002.safetensors' ] \
  || [ ! -f '$REWARD_MODEL/model-00002-of-00002.safetensors' ] \
  || [ "\$(stat -c '%s' '$REWARD_MODEL/model-00001-of-00002.safetensors' 2>/dev/null || echo 0)" -ne 1981392544 ] \
  || [ "\$(stat -c '%s' '$REWARD_MODEL/model-00002-of-00002.safetensors' 2>/dev/null || echo 0)" -ne 1417790344 ]; do sleep 60; done

python train_ppo.py \
  --data_path /home/caius/datasets/minimind_dataset/rlaif.jsonl \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --max_gen_len 256 \
  --batch_size 1 --mini_batch_size 1 --epochs 1 --learning_rate 3e-7 \
  --critic_learning_rate 5e-7 --ppo_update_iters 2 --save_interval 250 \
  --save_dir \"\$OUT\" --save_weight lifeos_agent_ppo_v1 \
  --from_weight lifeos_agent_dpo_v1 --reward_model_path '$REWARD_MODEL' \
  --device cuda --dtype bfloat16 --num_workers 2 --use_compile 0 \
  > \"\$OUT/lifeos_agent_ppo_v1.log\" 2>&1

python train_grpo.py \
  --data_path /home/caius/datasets/minimind_dataset/rlaif.jsonl \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --max_gen_len 256 \
  --batch_size 1 --num_generations 4 --epochs 1 --learning_rate 3e-7 \
  --save_interval 250 --save_dir \"\$OUT\" --save_weight lifeos_agent_grpo_v1 \
  --from_weight lifeos_agent_dpo_v1 --reward_model_path '$REWARD_MODEL' \
  --device cuda --dtype bfloat16 --num_workers 2 --use_compile 0 \
  > \"\$OUT/lifeos_agent_grpo_v1.log\" 2>&1

python train_agent.py \
  --data_path '$REMOTE_PROJECT/dataset/lifeos_agent_rl.jsonl' \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --max_gen_len 96 --max_total_len 1000 \
  --batch_size 1 --num_generations 4 --epochs 1 --learning_rate 1e-7 --thinking_ratio 0 \
  --save_interval 50 --save_dir \"\$OUT\" --save_weight lifeos_agent_rl_v2 \
  --from_weight lifeos_agent_dpo_v1 --reward_model_path '$REWARD_MODEL' \
  --device cuda --dtype bfloat16 --num_workers 2 --use_compile 0 \
  > \"\$OUT/lifeos_agent_rl_v2.log\" 2>&1
"
