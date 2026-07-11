# LifeOS-Agent Training Explained

这份文档解释这次 `LifeOS-Agent` 的训练原理，以及为了让 MiniMind 学会 Tool Calling，我们实际做了哪些代码、数据和运行流程改造。

## 1. 我们到底训练了什么

这次训练的目标不是从零训练一个大模型，而是在 MiniMind 已有能力上做一层小规模增量 SFT，让它更稳定地学会下面这条行为链：

```text
用户问题
-> 模型判断需要工具
-> 模型输出 <tool_call>
-> 外部 Python 执行工具
-> 工具结果作为 role=tool 回填
-> 模型基于 tool result 输出最终回答
```

所以训练的重点不是让模型“会算数学”或“真的有 Obsidian”，而是让模型学会：

1. 什么时候应该调用工具
2. 调哪个工具
3. `<tool_call>` 应该长什么样
4. 看到 `<tool_response>` 后怎么继续回答
5. 普通聊天时不要乱调用工具

## 2. Tool Calling 的核心原理

MiniMind 的 Tool Calling 不是模型自己执行工具，而是典型的“模型负责决策，外部程序负责执行”。

模型只输出类似这样的文本：

```xml
<tool_call>
{"name": "list_today_tasks", "arguments": {}}
</tool_call>
```

真正执行工具的是宿主程序，也就是 [lifeos_agent/main.py](/Users/caius/Documents/LifeOS-Agent/lifeos_agent/main.py)。

主程序会：

1. 用正则解析 `<tool_call>...</tool_call>`
2. 拿到工具名和参数
3. 调用 [lifeos_agent/tools.py](/Users/caius/Documents/LifeOS-Agent/lifeos_agent/tools.py) 里的 Python 函数
4. 把结果作为 `role="tool"` 追加回 `messages`
5. 再调用模型生成最终回答

这就是所谓的“外部循环”。

## 3. 为什么训练样本里必须有 tools schema

一开始我们只让训练样本包含：

- 用户问题
- assistant 的 `tool_calls`
- tool 的返回结果
- assistant 的最终回答

但这还不够。

推理时模型看到的 prompt 里会有 `<tools>...</tools>`，里面包含工具 schema，例如：

```xml
<tools>
{"type":"function","function":{"name":"list_today_tasks", ...}}
</tools>
```

如果训练时没有让模型见过“tools schema -> tool_call”的对应关系，它就容易只学到工具调用格式的一部分，而不能稳定地在推理时触发 `<tool_call>`。

所以我们修了 [scripts/build_lifeos_sft_mix.py](/Users/caius/Documents/LifeOS-Agent/scripts/build_lifeos_sft_mix.py)：

1. 从每条 seed 样本的 `tool_calls` 里提取工具名
2. 根据工具名找到对应 schema
3. 把 schema 注入到该样本的 `system.tools`
4. 只注入当前样本需要的候选工具，不全量注入

这一步是训练能明显变好的关键。

## 4. 为什么要混合官方数据和 LifeOS 私有数据

如果只用 `LifeOS` 的几十条 seed 样本训练，模型很容易过拟合：

- 只会背固定问题
- 普通聊天能力下降
- 输出变得机械
- 更容易复读 prompt

所以我们用两类数据混合：

1. 官方 MiniMind SFT 数据：`sft_t2t_mini.jsonl`
2. 私有 LifeOS seed 数据：`dataset/lifeos_sft_seed.jsonl`

官方数据负责维持通用聊天和基础语言能力，私有数据负责把模型往 `LifeOS-Agent` 的工具调用行为上推。

混合脚本是：

```bash
python scripts/build_lifeos_sft_mix.py
```

它支持两个关键参数：

- `--official_limit`：取多少条官方数据
- `--seed_repeat`：把 LifeOS seed 重复多少次，提高它在训练集里的权重

## 5. 三轮训练分别解决了什么

### 第一轮：跑通训练链路

第一轮使用：

- 官方样本：`20000`
- LifeOS seed 重复：`200`
- 输出：`full_sft_768.pth`

目标是验证：

1. 数据能不能被 MiniMind 的 `SFTDataset` 读取
2. 3090 Ti 能不能跑通训练
3. 模型是否开始学习 `<tool_call>`

这一轮暴露了第一个问题：`tool_calls` 字段格式不兼容。

MiniMind 期望 `tool_calls` 是字符串，而我们 seed 里最初是 Python list。于是我们在 builder 里加了自动归一化。

### 第二轮：补 tools schema

第二轮使用：

- 官方样本：`10000`
- LifeOS seed 重复：`400`
- 从第一轮权重继续训练
- 输出：`lifeos_sft_768.pth`

目标是让模型真的学会：

```text
看到工具 schema -> 输出对应 tool_call
```

这轮之后，三个工具类验收问题已经明显变好：

1. `我今天应该做什么？`
2. `17.66 涨停价是多少？`
3. `我之前学 SFTDataset 学到哪了？`

都能正确输出 `<tool_call>`。

### 第三轮：扩 seed，得到最佳权重

第三轮把 `lifeos_sft_seed.jsonl` 扩到 `26` 条：

- `18` 条 tool-use 样本
- `8` 条 no-tool 聊天样本

训练配置：

- 官方样本：`8000`
- LifeOS seed 重复：`150`
- 从 `lifeos_sft_768.pth` 继续训练
- 输出：`lifeos_agent_best_768.pth`

这一轮得到当前最佳权重：

```text
/home/caius/minimind/out/lifeos_agent_best_768.pth
```

## 6. 训练中做过的主要代码改动

### 6.1 主入口

文件：

- [lifeos_agent/main.py](/Users/caius/Documents/LifeOS-Agent/lifeos_agent/main.py)

主要改动：

1. 支持 HF 模型路径
2. 支持 MiniMind 原生 `.pth` 权重
3. 使用 `tokenizer.apply_chat_template(..., tools=tools)`
4. 打印完整 `input_text`
5. 解析 `<tool_call>`
6. 执行工具
7. 把工具结果追加为 `role="tool"`
8. 最多 3 轮，避免死循环
9. 加入 fallback，兜底明显退化的最终回答

### 6.2 工具层

文件：

- [lifeos_agent/tools.py](/Users/caius/Documents/LifeOS-Agent/lifeos_agent/tools.py)

实现工具：

1. `calculate_math`
2. `list_today_tasks`
3. `search_fake_obsidian`

同时增强了 `execute_tool()`：

1. 工具名不存在时返回错误
2. `arguments` 是 dict 时正常执行
3. `arguments` 是 JSON string 时先解析
4. JSON 格式错误时返回明确错误
5. 非 dict 参数返回明确错误

### 6.3 路由层

文件：

- [lifeos_agent/router.py](/Users/caius/Documents/LifeOS-Agent/lifeos_agent/router.py)

它负责根据用户输入选择候选工具。

重点是：

- 命中工具关键词时只传候选工具
- 普通聊天时返回空列表
- 空列表会让 `tools=None`

这样普通聊天不会被工具 schema 污染。

### 6.4 数据构建脚本

文件：

- [scripts/build_lifeos_sft_mix.py](/Users/caius/Documents/LifeOS-Agent/scripts/build_lifeos_sft_mix.py)

它做了几件关键事：

1. 读取官方 SFT 数据
2. 读取 LifeOS seed 数据
3. 归一化 `tools` 和 `tool_calls` 为字符串
4. 根据 `tool_calls` 自动注入对应 tools schema
5. 控制官方数据数量
6. 重复 LifeOS seed 数据，提高权重
7. shuffle 后写出混合数据

## 7. 为什么还加了 fallback

MiniMind 是一个很小的模型，当前只有约 64M 参数。它已经能学会工具调用链路，但在第二轮最终回答时偶尔会：

- 重复短语
- 复读 system prompt
- 输出过长 reasoning
- 把别的提示片段混进答案

所以我们在运行时加了一个轻量 fallback。

逻辑是：

1. 如果模型已经成功调用工具
2. 工具结果是可靠的
3. 但第二轮回答明显退化
4. 那就基于工具结果渲染一个保底回答

例如 `list_today_tasks` 的兜底回答是：

```text
你今天建议按这个顺序推进：先整理 Tool Calling 笔记，再复习 SFTDataset，最后跑通 LifeOS-Agent v0.1。
```

这不是长期理想形态，但对当前 `v0.1` 很实用。

## 8. 当前训练效果

当前最佳权重：

```text
/home/caius/minimind/out/lifeos_agent_best_768.pth
```

已验证通过：

1. `我之前学 SFTDataset 学到哪了？`
   - 调用 `search_fake_obsidian`
   - 基于笔记结果回答

2. `我今天应该做什么？`
   - 调用 `list_today_tasks`
   - 基于任务列表回答

3. `17.66 涨停价是多少？`
   - 调用 `calculate_math`
   - 工具结果为 `19.43`

4. `你好，简单介绍一下你自己`
   - 不传 tools
   - 不误调工具
   - 必要时 fallback 成简洁介绍

## 9. 怎么运行

### 单条对话

```bash
bash scripts/run_remote_lifeos_best.sh "我今天应该做什么？"
```

### 批量验收

```bash
bash scripts/remote_lifeos_selftest.sh
```

### 远程连续聊天

```bash
ssh wsl-dev
source /home/caius/lead-3d/venv/bin/activate
cd /home/caius/projects/LifeOS-Agent
python lifeos_agent/main.py \
  --minimind_repo /home/caius/minimind \
  --tokenizer_path /home/caius/minimind/model \
  --checkpoint_path /home/caius/minimind/out/lifeos_agent_best_768.pth
```

## 10. 这次训练的本质

可以把这次训练理解成：

```text
不是训练模型“拥有工具”
而是训练模型“学会请求工具”
```

真正的工具能力仍然在 Python 外部程序里。

模型学的是：

1. 读懂工具 schema
2. 输出合法 JSON tool call
3. 等待 tool response
4. 把工具返回结果组织成自然回答

这也是后面接真实 Obsidian、真实任务系统、真实 LifeOS 数据源时最重要的基础。

## 11. 下一步应该怎么做

接下来最值得做的是：

1. 把 `search_fake_obsidian` 换成真实 Markdown/Obsidian 检索
2. 把 `list_today_tasks` 换成真实任务来源
3. 继续扩充 `lifeos_sft_seed.jsonl`
4. 增加更多 no-tool 对话样本
5. 再做一轮小学习率精修

不建议现在立刻做：

1. 向量数据库
2. 复杂多 Agent
3. 长期记忆写入
4. Agentic RL

这些都可以做，但应该等真实 Obsidian 检索和基础工具链稳定以后再上。
