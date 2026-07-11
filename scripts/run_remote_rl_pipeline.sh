#!/usr/bin/env bash
# Queue PPO, GRPO, and Agent RL after the already-running DPO stage.
# Every stage writes a distinct checkpoint so an unsuccessful RL experiment
# never overwrites the verified SFT/DPO weights.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-wsl-dev}"
REMOTE_VENV="${REMOTE_VENV:-/home/caius/lead-3d/venv/bin/activate}"
REMOTE_MINIMIND="${REMOTE_MINIMIND:-/home/caius/minimind}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/caius/projects/LifeOS-Agent}"
REWARD_MODEL="${REWARD_MODEL:-/home/caius/internlm2-1_8b-reward}"

# The remote command intentionally runs stages serially: PPO keeps actor,
# critic, reference model, and reward model in memory, so concurrent jobs
# would make a 24 GB RTX 3090 Ti unstable.
ssh "$REMOTE_HOST" "
set -euo pipefail
source '$REMOTE_VENV'
cd '$REMOTE_MINIMIND/trainer'
OUT='$REMOTE_MINIMIND/out'

while pgrep -f '[t]rain_dpo.py.*lifeos_agent_dpo_v1' >/dev/null; do sleep 60; done
test -f \"\$OUT/lifeos_agent_dpo_v1_768.pth\"
while [ ! -f '$REWARD_MODEL/config.json' ]; do sleep 60; done

nohup python train_ppo.py \\
  --data_path /home/caius/datasets/minimind_dataset/rlaif.jsonl \\
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --max_gen_len 256 \\
  --batch_size 1 --mini_batch_size 1 --epochs 1 --learning_rate 3e-7 \\
  --critic_learning_rate 5e-7 --ppo_update_iters 2 --save_interval 250 \\
  --save_dir \"\$OUT\" --save_weight lifeos_agent_ppo_v1 \\
  --from_weight lifeos_agent_dpo_v1 --reward_model_path '$REWARD_MODEL' \\
  --device cuda --dtype bfloat16 --num_workers 2 --use_compile 0 \\
  > \"\$OUT/lifeos_agent_ppo_v1.log\" 2>&1

python train_grpo.py \\
  --data_path /home/caius/datasets/minimind_dataset/rlaif.jsonl \\
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --max_gen_len 256 \\
  --batch_size 1 --num_generations 4 --epochs 1 --learning_rate 3e-7 \\
  --save_interval 250 --save_dir \"\$OUT\" --save_weight lifeos_agent_grpo_v1 \\
  --from_weight lifeos_agent_dpo_v1 --reward_model_path '$REWARD_MODEL' \\
  --device cuda --dtype bfloat16 --num_workers 2 --use_compile 0 \\
  > \"\$OUT/lifeos_agent_grpo_v1.log\" 2>&1

python train_agent.py \\
  --data_path '$REMOTE_PROJECT/dataset/minimind_dataset/agent_rl.jsonl' \\
  --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --max_gen_len 256 --max_total_len 1600 \\
  --batch_size 1 --num_generations 4 --epochs 1 --learning_rate 3e-7 \\
  --save_interval 250 --save_dir \"\$OUT\" --save_weight lifeos_agent_rl_v1 \\
  --from_weight lifeos_agent_dpo_v1 --reward_model_path '$REWARD_MODEL' \\
  --device cuda --dtype bfloat16 --num_workers 2 --use_compile 0 \\
  > \"\$OUT/lifeos_agent_rl_v1.log\" 2>&1
" 
