# LifeOS-Agent

`LifeOS-Agent` 是一个基于 MiniMind Tool Calling 外部循环实现的最小 Agent Demo。

当前版本 `v0.1` 主要验证这条链路：

1. 用户提问
2. 模型输出 `<tool_call>`
3. 外部程序执行工具
4. 工具结果以 `role=tool` 回填
5. 模型第二轮继续生成最终回答

当前实现位于 [lifeos_agent](lifeos_agent/)。

详细说明见：

- [lifeos_agent/README.md](lifeos_agent/README.md)
- [lifeos_agent/SELFTEST.md](lifeos_agent/SELFTEST.md)
- [TRAINING_PLAN.md](TRAINING_PLAN.md)
- [REMOTE_SETUP.md](REMOTE_SETUP.md)
- [IMPLEMENTATION.md](IMPLEMENTATION.md)
- [TRAINING_EXPLAINED.md](TRAINING_EXPLAINED.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/TRAINING_FLOW.md](docs/TRAINING_FLOW.md)
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md)
- [docs/ROADMAP.md](docs/ROADMAP.md)

## 当前工具

- `calculate_math`
- `list_today_tasks`
- `search_fake_obsidian`

## 运行示例

```bash
python lifeos_agent/main.py \
  --minimind_repo /path/to/minimind-master \
  --tokenizer_path /path/to/minimind-master/model \
  --checkpoint_path /path/to/full_sft_768.pth \
  --prompt "我之前学 SFTDataset 学到哪了？"
```

## 数据检查

```bash
python scripts/prepare_lifeos_data.py
```

## 远程 3090 Ti 快速运行

```bash
bash scripts/run_remote_3090_demo.sh "我今天应该做什么？"
```

## 远程最佳权重运行

```bash
bash scripts/run_remote_lifeos_best.sh "我今天应该做什么？"
```

## 远程批量验收

```bash
bash scripts/remote_lifeos_selftest.sh
```
