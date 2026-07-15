# Review-only materials

这个目录保存对话复盘和候选数据，不是训练数据目录。

约束：

- 所有 JSONL 记录必须包含 `"use_for_training": false`。
- 文件不得复制到 `dataset/`，不得被训练脚本 glob 或引用。
- 在人工检查事实、隐私、输出质量和许可之前，不得转换为 SFT/DPO/RL 数据。
- `scripts/validate_review_materials.py` 用于检查这些边界。

训练使用的数据只允许来自训练命令显式指定的 `dataset/...` 路径。
