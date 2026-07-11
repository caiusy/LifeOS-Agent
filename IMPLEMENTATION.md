# LifeOS-Agent Implementation Notes

这份文档记录当前 `LifeOS-Agent` 的实现状态、训练过程、远程部署方式，以及现阶段为了提高可用性加入的工程兜底逻辑。

## 1. 当前架构

当前版本仍然是一个最小 Tool Calling Agent，核心链路是：

1. 用户输入
2. 路由器挑选候选工具
3. `apply_chat_template(..., tools=tools)` 生成 prompt
4. 模型输出 `<tool_call>...</tool_call>`
5. 外部 Python 执行工具
6. 工具结果以 `role=tool` 追加回 `messages`
7. 模型第二轮生成最终回答

代码入口：

- [lifeos_agent/main.py](/Users/caius/Documents/LifeOS-Agent/lifeos_agent/main.py)
- [lifeos_agent/router.py](/Users/caius/Documents/LifeOS-Agent/lifeos_agent/router.py)
- [lifeos_agent/tools.py](/Users/caius/Documents/LifeOS-Agent/lifeos_agent/tools.py)
- [lifeos_agent/fake_notes.py](/Users/caius/Documents/LifeOS-Agent/lifeos_agent/fake_notes.py)

## 2. 当前工具

### `calculate_math`

- 输入：`expression: string`
- 输出：`{"result": ...}`

### `list_today_tasks`

- 输入：无
- 输出：

```json
{"tasks":["整理 Tool Calling 笔记","复习 SFTDataset","跑通 LifeOS-Agent v0.1"]}
```

### `search_fake_obsidian`

- 输入：`query: string`
- 输出：`{"results":[{"title":"...","content":"..."}]}`

## 3. 训练数据策略

官方训练数据使用：

- `dataset/minimind_dataset/sft_t2t_mini.jsonl`

私有增量数据使用：

- [dataset/lifeos_sft_seed.jsonl](/Users/caius/Documents/LifeOS-Agent/dataset/lifeos_sft_seed.jsonl)

当前 seed 数据已经扩展到：

- `26` 条样本
- 其中：
  - `18` 条 tool-use 样本
  - `8` 条 no-tool 聊天样本

## 4. 训练时踩过的关键坑

### 4.1 `tool_calls` 字段格式不兼容

`MiniMind` 的 `SFTDataset` 期望：

- `tools` 是字符串
- `tool_calls` 也是字符串

而不是 Python list / dict。

因此在 [scripts/build_lifeos_sft_mix.py](/Users/caius/Documents/LifeOS-Agent/scripts/build_lifeos_sft_mix.py) 里做了自动归一化。

### 4.2 训练样本里不能只给 `tool_calls`

只给：

- `assistant.tool_calls`
- `tool`
- `assistant`

还不够。

如果训练时 system 里没有把相关 `tools schema` 一起放进去，模型容易学会“见过 tool_call 的格式”，但学不会“在看到 `<tools>` 提示时该输出哪个 `<tool_call>`”。

当前 builder 已经修复这个问题：

- 从样本里的 `tool_calls` 反推工具名
- 只把相关工具 schema 注入 `system.tools`

## 5. 已完成的远程训练

远程机器：

- `WSL + RTX 3090 Ti`

关键路径：

- 项目：`/home/caius/projects/LifeOS-Agent`
- MiniMind 源码：`/home/caius/minimind`
- 最佳权重：`/home/caius/minimind/out/lifeos_agent_best_768.pth`

### 第一轮

- 混合数据：`lifeos_sft_mixed.jsonl`
- 目标：先让模型学会基础 Tool Calling

### 第二轮

- 混合数据：`lifeos_sft_mixed_v2.jsonl`
- 目标：加入 tool schema 注入，强化工具选择和 `<tool_call>` 格式

### 第三轮

- 混合数据：`lifeos_sft_mixed_v3.jsonl`
- 目标：扩充 no-tool 与 tool-after-response 样本，得到当前最佳权重

## 6. 当前最佳运行方式

### 单条运行

```bash
bash scripts/run_remote_lifeos_best.sh "我今天应该做什么？"
```

### 批量验收

```bash
bash scripts/remote_lifeos_selftest.sh
```

## 7. 当前验收状态

当前 `lifeos_agent_best_768.pth` 已经稳定具备：

- 正确选择候选工具
- 正确输出 `<tool_call>`
- 正确执行 fake tool
- 正确把 tool result 回填给模型
- 在多数 case 中基于 tool result 继续回答

表现较稳的 case：

1. `我之前学 SFTDataset 学到哪了？`
2. `17.66 涨停价是多少？`
3. `我今天应该做什么？`

仍然存在的质量问题：

1. 二轮最终回答有时会轻微重复
2. no-tool 聊天偶尔会复读 system prompt 的一部分

## 8. 运行时兜底

为了让当前版本更接近“可用”而不是“只在理想输出下可用”，在 [lifeos_agent/main.py](/Users/caius/Documents/LifeOS-Agent/lifeos_agent/main.py) 里加了一个轻量 fallback：

- 如果模型已经正确调用工具
- 但第二轮回答明显退化
  - 过长
  - 复读 system prompt
  - 出现明显异常片段

则程序会基于 `tool result` 直接渲染一个保底回答。

这不是长期终态，但它能显著提升当前 demo 的可用性。

## 9. 下一步建议

最优先的不是再大规模重训，而是：

1. 继续扩 `lifeos_sft_seed.jsonl`
2. 增加更多 no-tool 自然聊天样本
3. 增加更多“工具后最终回答去重复”的样本
4. 再做一轮小学习率精修

之后再进入：

1. `search_fake_obsidian -> 真实 Markdown 检索`
2. 更细的任务管理工具
3. 真实 Obsidian / LifeOS 数据接入
