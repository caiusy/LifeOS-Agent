# LifeOS-Agent

`LifeOS-Agent` 是一个基于 MiniMind Tool Calling 外部循环实现的最小 Agent Demo。

当前版本 `v0.1` 主要验证这条链路：

1. 用户提问
2. 模型输出 `<tool_call>`
3. 外部程序执行工具
4. 工具结果以 `role=tool` 回填
5. 模型第二轮继续生成最终回答

当前实现位于 [lifeos_agent](lifeos_agent/)。

## 文档导航

建议按以下顺序阅读，完整目录见 [docs/README.md](docs/README.md)：

1. [训练全解：SFT、DPO、PPO、GRPO 与 Agent RL](docs/TRAINING_METHODS_COMPLETE_GUIDE.md)
2. [数学推导附录](MATHEMATICAL_DERIVATIONS.md)
3. [最终项目报告](FINAL_PROJECT_REPORT.md)
4. [架构说明](docs/ARCHITECTURE.md)
5. [用户指南](docs/USER_GUIDE.md)
6. [评测集说明](eval/README.md)

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

## 本地学习演示与自动测试

不加载模型，逐步展示路由、schema、解析、工具执行和 `role=tool` 回填：

```bash
python scripts/run_learning_demo.py
```

运行工具链回归测试：

```bash
python -m unittest discover -s tests -v
```

生成并验证固定的 100 条模型能力评测集：

```bash
python scripts/build_eval_dataset.py
python scripts/validate_eval_dataset.py
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
