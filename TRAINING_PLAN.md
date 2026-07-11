# LifeOS-Agent 3090 Ti Plan

这份计划面向单卡 `RTX 3090 Ti`，目标不是一次性追求完整工业级 Agent，而是按可验证、可迭代的顺序，把 `LifeOS-Agent` 从 `v0.1 demo` 推进到可训练版本。

## 当前结论

- `LifeOS-Agent v0.1` 已具备最小 Tool Calling 外循环。
- 当前最适合的官方数据不是单独的 toolcall 数据集，而是 `MiniMind` 主线 SFT 数据：
  - `sft_t2t_mini.jsonl`
  - 如需更完整复现，再考虑 `sft_t2t.jsonl`
- `Tool Calling` 样本已经混入 `sft_t2t_mini.jsonl` / `sft_t2t.jsonl`。
- 对于 `3090 Ti`，推荐先做轻量 SFT，再做私有增量数据，再考虑 Agentic RL。

## 下载状态

截至当前，本地和远程都已经有一批可直接使用的数据：

- 已完成：
  - `sft_t2t_mini.jsonl`
  - `pretrain_t2t_mini.jsonl`
  - `dpo.jsonl`
  - `agent_rl.jsonl`
  - `agent_rl_math.jsonl`
  - `lora_exam.jsonl`
  - `lora_identity.jsonl`
  - `lora_medical.jsonl`
  - `rlaif.jsonl`
- 已新增：
  - `dataset/lifeos_sft_seed.jsonl`
- 暂未完整拉取：
  - `sft_t2t.jsonl`
  - `pretrain_t2t.jsonl`

建议：

1. 先用现有 `sft_t2t_mini.jsonl`
2. 混合 `dataset/lifeos_sft_seed.jsonl` 做私有增量验证
3. 暂不继续拉取更大的 `sft_t2t.jsonl`

## 路线图

### 阶段 0：稳住当前 v0.1

目标：

- 保证工具路由、`<tool_call>` 解析、`role=tool` 回填、最大轮数限制都稳定

必须完成：

1. 普通闲聊默认不传 `tools`
2. 命中工具时只传候选工具，而不是全量工具
3. `execute_tool()` 对坏参数报清晰错误
4. 保证最大轮数限制为 `3`

当前状态：

- 已完成

### 阶段 1：跑通轻量 SFT

目标：

- 在 `3090 Ti` 上跑通 MiniMind 风格的 Tool Calling 微调链路

推荐数据：

- `sft_t2t_mini.jsonl`

推荐训练策略：

1. 先不做从零预训练
2. 先不碰 `agent_rl`
3. 先验证 SFT 能否让模型更稳定地产生 `<tool_call>`

推荐配置思路：

- `max_seq_len`: `768`
- 训练卡：单卡 `3090 Ti`
- 优先使用较小 `batch size`
- 使用 `gradient accumulation` 换显存
- 首轮只跑短周期验证，不追求完整收敛

本阶段验收：

1. `我之前学 SFTDataset 学到哪了？`
   - 期望工具：`search_fake_obsidian`
2. `我今天应该做什么？`
   - 期望工具：`list_today_tasks`
3. `17.66 涨停价是多少？`
   - 期望工具：`calculate_math`
   - 期望结果：约 `19.43`
4. `你好，简单介绍一下你自己`
   - 期望：不调用工具，直接回答

### 阶段 2：补 LifeOS-Agent 私有增量数据

目标：

- 让模型学会更贴近 `LifeOS-Agent` 的路由与回答风格

建议样本类型：

1. Obsidian 检索类
2. 今日任务 / 计划类
3. 计算类 / 涨停价类
4. 普通闲聊负样本

建议首批数据量：

- `300` 到 `1000` 条高质量样本

优先级：

1. `search_fake_obsidian`
2. `list_today_tasks`
3. `calculate_math`
4. `no-tool` 闲聊样本

### 阶段 3：替换 fake tools

目标：

- 逐步从 fake notes / fake tasks 迁到真实数据源

顺序建议：

1. `search_fake_obsidian` -> 本地 Markdown 扫描
2. 标题匹配 + 关键词检索
3. BM25
4. 向量检索
5. 写入类工具

不建议立即做：

- 向量数据库
- 多数据源同步
- 自动任务编排

### 阶段 4：再考虑 Agentic RL

目标：

- 等 SFT 和私有数据都稳了，再上多轮 rollout 与延迟奖励

建议时机：

- 只有当下面三项都稳定后才开始：
  1. 工具选择稳定
  2. tool result 利用稳定
  3. 普通对话不乱调工具

## 3090 Ti 显卡建议

`3090 Ti` 对当前阶段足够用，但建议策略是“先轻后重”：

### 推荐做

1. 单卡 `sft_t2t_mini`
2. `max_seq_len 768`
3. 小 batch + gradient accumulation
4. 先跑可验证版本

### 暂不推荐

1. 一上来用 `sft_t2t.jsonl`
2. 一上来跑 `agent_rl`
3. 一上来接真实向量库
4. 一上来做全量长期训练

## 具体实施步骤

### Step 1：检查现有数据完整性

目标：

- 确认远程和本地都能读到关键数据

完成标准：

- `python scripts/prepare_lifeos_data.py` 能正常输出记录数
- `dataset/lifeos_sft_seed.jsonl` 可被正常解析

### Step 2：确认 v0.1 自测通过

运行验证：

```bash
python lifeos_agent/main.py \
  --minimind_repo /path/to/minimind-master \
  --tokenizer_path /path/to/minimind-master/model \
  --checkpoint_path /path/to/full_sft_768.pth \
  --prompt "我今天应该做什么？"
```

完成标准：

1. 打印 `input_text`
2. 出现 `<tool_call>`
3. 工具结果被回填
4. 模型第二轮输出最终答案

### Step 3：扩充私有 SFT 数据

当前目录：

- `dataset/lifeos_sft_seed.jsonl`

当前已覆盖：

1. `search_fake_obsidian`
2. `list_today_tasks`
3. `calculate_math`
4. `no-tool` 闲聊

建议样本格式：

```jsonl
{"conversations":[
  {"role":"system","content":"# Tools ...","tools":"[...]"},
  {"role":"user","content":"我今天应该做什么？"},
  {"role":"assistant","content":"","tool_calls":"[{\"name\":\"list_today_tasks\",\"arguments\":{}}]"},
  {"role":"tool","content":"{\"tasks\":[\"整理 Tool Calling 笔记\",\"复习 SFTDataset\",\"跑通 LifeOS-Agent v0.1\"]}"},
  {"role":"assistant","content":"你今天建议先整理 Tool Calling 笔记，再复习 SFTDataset，最后跑通 LifeOS-Agent v0.1。"}
]}
```

### Step 4：开始轻量 SFT

目标：

- 让模型在你关心的 3 类任务上更稳定

建议训练顺序：

1. 只用官方 `sft_t2t_mini`
2. 官方 `sft_t2t_mini` + `dataset/lifeos_sft_seed.jsonl` 混合
3. 对比训练前后 4 个 case

### 3090 Ti 第一轮建议命令

先构造混合数据：

```bash
cd /home/caius/projects/LifeOS-Agent
source /home/caius/lead-3d/venv/bin/activate
python scripts/build_lifeos_sft_mix.py \
  --official dataset/minimind_dataset/sft_t2t_mini.jsonl \
  --seed dataset/lifeos_sft_seed.jsonl \
  --output dataset/lifeos_sft_mixed.jsonl \
  --official_limit 20000 \
  --seed_repeat 200
```

再启动第一轮 SFT：

```bash
cd /home/caius/projects/minimind-master/trainer
source /home/caius/lead-3d/venv/bin/activate
python train_full_sft.py \
  --data_path /home/caius/projects/LifeOS-Agent/dataset/lifeos_sft_mixed.jsonl \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 768 \
  --batch_size 4 \
  --accumulation_steps 4 \
  --epochs 1 \
  --learning_rate 1e-5 \
  --save_interval 500 \
  --from_weight pretrain
```

这组配置的意图是：

1. `batch_size 4 * accumulation 4`，先保守适配单卡 `3090 Ti`
2. `official_limit 20000`，先做快验证，不一上来吃完整 `90w+` 行
3. `seed_repeat 200`，让你的 LifeOS 样本在第一轮里足够“被看见”

## 验证问题与期望答案

### 验证 1

问题：

- `我之前学 SFTDataset 学到哪了？`

期望：

- 工具：`search_fake_obsidian`
- 返回内容应提到：
  - `SFTDataset`
  - `conversations`
  - `apply_chat_template`
  - `tool_calls` / `<tool_response>`

### 验证 2

问题：

- `我今天应该做什么？`

期望：

- 工具：`list_today_tasks`
- 工具结果应包含：
  - `整理 Tool Calling 笔记`
  - `复习 SFTDataset`
  - `跑通 LifeOS-Agent v0.1`

### 验证 3

问题：

- `17.66 涨停价是多少？`

期望：

- 工具：`calculate_math`
- 表达式接近：
  - `round(17.66 * 1.1, 2)`
  - 或 `17.66 * 1.1`
- 最终结果：
  - 约 `19.43`

### 验证 4

问题：

- `你好，简单介绍一下你自己`

期望：

- 不调用工具
- 直接给普通回答

## 待办清单

### 已完成

- `LifeOS-Agent v0.1` 最小闭环
- fake tools
- 工具路由
- `role=tool` 回填
- 最大轮数限制
- 普通闲聊不传工具
- README 修正

### 进行中

- 下载 `sft_t2t_mini.jsonl`

### 下一步

1. 补 `SELFTEST.md`
2. 生成第一批 `lifeos_sft_seed.jsonl`
3. 写 3090 Ti 的训练命令草案
4. 加入训练前后 4 个 case 的对比脚本
