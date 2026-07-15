# 2026-07-15 LifeOS-Agent 工作复盘

## 今天确认的事实

1. 远程 DPO、PPO、GRPO、Agent RL 全部训练结束。
2. Agent RL 完成 39,988 step，每个 prompt 生成 4 条轨迹，共 159,952 条 rollout。
3. 最终权重位于 `/home/caius/minimind/out/lifeos_agent_rl_v1_768.pth`。
4. 训练日志无 Traceback、OOM、NaN 或 killed。
5. 最终模型没有通过四条 LifeOS Tool Calling 原生验收。
6. 外部 router、parser、executor 的静态测试通过，但不能代表模型生成能力通过。

## 今天最重要的认识

“训练成功”至少包含三个层次：

- 工程成功：训练程序跑完并保存 checkpoint。
- 优化成功：训练目标上的 reward/loss 按预期变化。
- 产品成功：真实用户问题能稳定完成工具调用闭环。

本次前两个层次具备证据，第三个层次没有通过。

## 发现的核心差距

- 通用 Agent RL 的工具集合和 LifeOS 三个工具不一致。
- 训练与推理 prompt、thinking 模式存在差异。
- reward 强调 GT 覆盖与调用数量，但产品要求严格 XML/JSON 和工具结果 grounding。
- demo fallback 可能把模型失败包装成可用答案，因此评测必须记录 fallback 前输出。

## 下一步决策

先停止大规模通用 RL，转向小规模 LifeOS 专属 Tool Calling SFT 数据和严格模型评测。今天整理的 Codex 对话只作为复盘候选，不加入训练。

## 数据治理状态

- 候选文件：`review_materials/2026-07-15/codex_conversation_candidates.jsonl`
- `use_for_training`：全部为 `false`
- 训练目录：未复制到 `dataset/`
- 训练脚本：不允许引用 `review_materials/`
