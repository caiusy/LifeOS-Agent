#!/usr/bin/env bash
# Runs on the remote Linux machine. It waits for prerequisites and then runs
# PPO -> GRPO -> Agent RL one at a time on the single RTX 3090 Ti.
set -euo pipefail

# Keep the RL stack isolated from the user's lead-3d environment. Transformers
# 4.57.6 can read MiniMind's tokenizers-0.22 JSON while retaining the legacy
# DynamicCache API required by InternLM2-Reward; Transformers 5.x cannot.
source "${LIFEOS_VENV_ACTIVATE:-/home/caius/minimind/.venv-lifeos/bin/activate}"
cd /home/caius/minimind/trainer

OUT=/home/caius/minimind/out
REWARD_MODEL=/home/caius/internlm2-1_8b-reward
RLAIF=/home/caius/datasets/minimind_dataset/rlaif.jsonl
# Never optimize LifeOS behavior on the generic six-tool corpus. The dedicated
# builder keeps tool schemas, mock execution, and deployment prompts aligned.
AGENT_DATA=/home/caius/projects/LifeOS-Agent/dataset/lifeos_agent_rl.jsonl

# DPO uses the v4 policy as both policy initialization and frozen reference.
# Do not start an RL stage until its resulting policy checkpoint is complete.
while pgrep -f '[t]rain_dpo.py.*lifeos_agent_dpo_v1' >/dev/null; do sleep 60; done
test -f "$OUT/lifeos_agent_dpo_v1_768.pth"
test -s "$AGENT_DATA"

# PPO/GRPO load a separate 1.8B reward model. Small metadata files arrive
# before multi-GB shards during rsync, so checking config.json alone has a
# race: PPO may start against an incomplete model. Require the exact upstream
# shard sizes before allocating any GPU models.
REWARD_SHARD_1="$REWARD_MODEL/model-00001-of-00002.safetensors"
REWARD_SHARD_2="$REWARD_MODEL/model-00002-of-00002.safetensors"
while [ ! -f "$REWARD_MODEL/config.json" ] \
  || [ ! -f "$REWARD_SHARD_1" ] \
  || [ ! -f "$REWARD_SHARD_2" ] \
  || [ "$(stat -c '%s' "$REWARD_SHARD_1" 2>/dev/null || echo 0)" -ne 1981392544 ] \
  || [ "$(stat -c '%s' "$REWARD_SHARD_2" 2>/dev/null || echo 0)" -ne 1417790344 ]; do
  sleep 60
done

python train_ppo.py \
  --data_path "$RLAIF" \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --max_gen_len 256 \
  --batch_size 1 --mini_batch_size 1 --epochs 1 --learning_rate 3e-7 \
  --critic_learning_rate 5e-7 --ppo_update_iters 2 --save_interval 250 \
  --save_dir "$OUT" --save_weight lifeos_agent_ppo_v1 \
  --from_weight lifeos_agent_dpo_v1 --reward_model_path "$REWARD_MODEL" \
  --device cuda --dtype bfloat16 --num_workers 2 --use_compile 0 \
  > "$OUT/lifeos_agent_ppo_v1.log" 2>&1

python train_grpo.py \
  --data_path "$RLAIF" \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --max_gen_len 256 \
  --batch_size 1 --num_generations 4 --epochs 1 --learning_rate 3e-7 \
  --save_interval 250 --save_dir "$OUT" --save_weight lifeos_agent_grpo_v1 \
  --from_weight lifeos_agent_dpo_v1 --reward_model_path "$REWARD_MODEL" \
  --device cuda --dtype bfloat16 --num_workers 2 --use_compile 0 \
  > "$OUT/lifeos_agent_grpo_v1.log" 2>&1

# Agent RL is deliberately last because it performs multi-turn tool rollouts
# and has the largest time cost. It starts from DPO to keep the shared base
# behavior comparable with PPO and GRPO experiments.
python train_agent.py \
  --data_path "$AGENT_DATA" \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --max_gen_len 96 --max_total_len 1000 \
  --batch_size 1 --num_generations 4 --epochs 1 --learning_rate 3e-7 \
  --save_interval 50 --save_dir "$OUT" --save_weight lifeos_agent_rl_v2 \
  --from_weight lifeos_agent_dpo_v1 --reward_model_path "$REWARD_MODEL" \
  --device cuda --dtype bfloat16 --num_workers 2 --use_compile 0 \
  > "$OUT/lifeos_agent_rl_v2.log" 2>&1
