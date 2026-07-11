# Remote Setup

这份文档对应你当前可用的远程 `WSL + RTX 3090 Ti` 环境。

## 当前约定路径

- 远程项目目录：`/home/caius/projects/LifeOS-Agent`
- 远程数据目录：`/home/caius/datasets/minimind_dataset`
- 远程 HF 模型：`/home/caius/models/minimind-3`
- 远程 PyTorch 权重：`/home/caius/models/minimind-3-pytorch`
- 虚拟环境：`/home/caius/lead-3d/venv/bin/activate`

## 连接远程机器

```bash
ssh wsl-dev
```

## 激活环境

```bash
source /home/caius/lead-3d/venv/bin/activate
```

## 检查 CUDA

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

## 运行 LifeOS-Agent demo

```bash
cd /home/caius/projects/LifeOS-Agent
python lifeos_agent/main.py \
  --hf_model_path /home/caius/models/minimind-3 \
  --prompt "我今天应该做什么？"
```

如果要走 MiniMind 原生 `.pth` 权重模式，需要你远程机器上也有 `minimind-master` 源码目录，再补 `--minimind_repo`、`--tokenizer_path`、`--checkpoint_path`。

## 检查数据是否到位

```bash
cd /home/caius/projects/LifeOS-Agent
python scripts/prepare_lifeos_data.py
```
