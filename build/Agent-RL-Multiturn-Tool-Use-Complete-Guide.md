---
title: Agent RL 全链路教程：从工具选择、多轮对话到 Reward、Advantage 与 Token Loss
date: 2026-07-13 23:40:00
updated: 2026-07-13 23:45:00
mathjax: true
description: "以 MiniMind 和 LifeOS-Agent 的真实代码为主线，用 tokenizer 实测的一条 361-token 工具轨迹和四条可复算 rollout，完整解释 schema、router、多轮 observation、张量维度、reward、group advantage、KL、CISPO token loss 与反向传播。"
categories:
  - AI与大模型
  - Agent
tags:
  - Agent-RL
  - Tool-Calling
  - MiniMind
  - LifeOS-Agent
  - GRPO
  - CISPO
type: deep-dive
difficulty: progressive
review_status: published
---

> 这是一篇从小学生直觉逐层走到高中数学与工程代码的独立教材。所有工程结论均映射到当前 MiniMind/LifeOS-Agent 实现，所有教学数字均明确标注，不把假设当实测。

<!-- more -->


> 这是一篇可以从头连续读完的独立教材。它只讨论一件事：一个语言模型怎样在外部程序的帮助下调用工具，并通过强化学习逐渐提高“何时调用、调用哪个、参数怎么写、拿到结果后怎样回答”的成功率。
>
> 本文的工程事实以当前 MiniMind `dataset/lm_dataset.py`、`trainer/train_agent.py` 和 LifeOS-Agent 代码为准；论文概念以文末原始论文为准。第 7～12 章的 token 长度由当前 tokenizer 实测，固定轨迹的 reward 按代码逐项计算；只有明确标为“受控教学值”的 log probability、ratio 和 KL 不是训练日志。

---

## 0. 读完后必须能回答的 12 个问题

读第一遍，建立完整地图；读第二遍，跟着数字手算。读完后你应该能独立回答：

1. Agent、模型、router、tool schema、executor、environment 分别是谁？
2. 工具为什么必须先以 schema 的形式放进 prompt？
3. 是谁筛选候选工具，又是谁在候选工具中作最终选择？
4. `messages = conversations[:-1]` 到底删掉了哪一条？为什么要删？
5. `<tool_call>` 是模型生成的字符串，还是 Python 函数调用？
6. 为什么模型生成字符串，Python 却能真的执行工具？
7. 多轮对话为什么要把 `role="tool"` 的结果重新喂给模型？
8. prompt、tool call、tool observation、final answer 哪些 token 计算 policy loss？
9. 一条 prompt 为什么要采样 4 条 rollout？
10. reward 怎样变成 advantage？
11. `input_ids [B×G,T]` 怎样经过 Transformer 变成 `logits [B×G,T,V]`？
12. 最后怎样从每个 token 的 log probability 得到一个标量 loss，并更新模型参数？

![Agent RL 学习地图](/images/lifeos-agent-training/agent_rl_learning_map.svg)

---

## 1. 先用小学生也能懂的故事解释

把 Agent 想成一个参加“开卷考试”的学生。

- **大模型**是学生的大脑。它会阅读、写字、推理，但心算可能出错。
- **工具清单**是桌上的工具说明书：计算器怎么用、天气查询需要什么参数。
- **router** 是发工具箱的老师。数学题只发计算器，不必把 500 把工具全部堆在桌上。
- **tool call** 是学生写给监考员的纸条：“请用计算器算 `2045*6994`”。
- **executor** 是监考员。它读纸条、检查格式、真的按计算器，然后把结果交回来。
- **tool response** 是计算器显示的 `14302730`。
- **最终回答** 是学生看到结果后写：“2045 × 6994 = 14302730。”
- **reward** 是老师给整次答题过程的分数。
- **强化学习**是让学生多答几遍，对高分答法提高复现概率，对低分答法降低概率。

最重要的一句话是：

> 模型不会直接运行 Python。模型只生成 token；外部程序把特定 token 解析为结构化请求，执行函数，再把结果变成新 token 回填。

所以 Agent 不是“一个突然获得系统权限的模型”，而是一个闭环系统：

```text
模型负责决定和表达行动
        ↓
外部程序负责验证和执行行动
        ↓
模型读取观察结果并继续决策
```

这与 ReAct 提出的“推理和行动交错”思想一致：行动让模型从外部环境获得新信息，随后推理可以根据新信息修正计划。

---

## 2. 把故事翻译成强化学习术语

### 2.1 状态、动作、环境、轨迹

在第 `t` 个生成位置：

- 状态 `s_t`：到目前为止模型看到的全部 token，包括 system、工具 schema、用户问题、先前输出和工具观察。
- 动作 `a_t`：模型在词表 `V` 中选出的下一个 token。
- 策略 `π_θ(a_t|s_t)`：参数为 `θ` 的模型给下一个 token 的概率。
- 环境：Python 外部循环与工具函数。
- 观察 `o_t`：工具返回的 JSON 文本。
- 轨迹 `τ`：从初始 prompt 到结束的完整交互序列。
- reward `R(τ)`：对整条轨迹的评分。

一段 `<tool_call>` 看起来是一个动作，但对自回归语言模型而言，它其实是几十个连续动作：

```text
<  tool  _  call  >  {  "name"  :  ...
↑   ↑    ↑    ↑   ↑  ↑      每个 token 都是一次动作
```

tokenizer 实际如何切分由词表决定；上面的空格只是示意，不能当成真实 tokenization。

### 2.2 Agent 的“行动”为什么仍是文字

语言模型的输出空间就是词表，所以最自然的接口是让行动也表示成文字：

```xml
<tool_call>{"name":"calculate_math","arguments":{"expression":"2045*6994"}}</tool_call>
```

外部程序约定：只要发现完整的 `<tool_call>...</tool_call>`，就把标签内部 JSON 解析成对象。于是：

```text
token 序列 → decode → XML 包裹的 JSON → json.loads → Python dict → execute_tool
```

“能调用工具”因此至少包含四种不同能力：

1. 决定此题是否需要工具。
2. 在候选 schema 中选择合法工具名。
3. 按 JSON schema 生成合法参数。
4. 看到 observation 后停止继续调用并组织最终答案。

SFT 可以教会格式和示范行为；Agent RL 通过结果奖励进一步优化整条行为链。

---

## 3. 四个角色不要混淆：router、LLM、parser、executor

![训练与推理阶段的工具选择](/images/lifeos-agent-training/agent_rl_tool_selection.svg)

### 3.1 router：筛候选集合

LifeOS-Agent 推理时，`select_tool_names(user_input)` 使用关键词规则筛候选工具。例如：

```python
if any(word in text for word in ["算", "计算", "多少", "乘以", "涨停价"]):
    selected.append("calculate_math")
```

输入“2045 乘以 6994 是多少”后，它返回：

```python
["calculate_math"]
```

然后 `get_tools_by_names` 把名称转成完整 JSON schema。router 不生成最终 `<tool_call>`，也不执行函数；它只缩小候选集合。

### 3.2 LLM：在 prompt 给出的候选中决定行动

`apply_chat_template(messages, tools=tools, ...)` 把候选 schema 渲染进 prompt。模型读取说明后，可以：

- 直接回答，不使用工具；
- 生成一个候选工具的调用；
- 在有多个候选时选其中一个；
- 也可能生成错误名称或错误参数，随后被评分或拦截。

### 3.3 parser：把文字变成数据结构

当前 MiniMind 的核心解析逻辑是：

```python
def parse_tool_calls(text):
    calls = []
    for block in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            calls.append(json.loads(block.strip()))
        except Exception:
            pass
    return calls
```

它只负责“识别标签 + 解析 JSON”。JSON 错误时调用被忽略，不会凭空修复。

### 3.4 executor：验证并真正执行

executor 根据 `name` 查注册表，再把 `arguments` 传给对应函数。生产系统还应做类型校验、权限检查、超时、速率限制、审计与结果截断。

### 3.5 训练时是谁选候选工具

MiniMind 当前 Agent RL 训练不会调用 LifeOS 的关键词 router。每条训练数据已经把候选工具写在 system message 的 `tools` 字段里：

```text
数据构造者提前选择候选 tools
        ↓
AgentRLDataset 读取 tools
        ↓
chat template 渲染 schema
        ↓
模型在这些候选中行动
```

这意味着：如果每条样本永远只给唯一正确工具，训练主要学习“怎样使用已提示的工具”；要强化“多个工具里选对一个”的能力，数据必须包含多候选、干扰工具和无需调用工具的负样本。

---

## 4. 一条真实数据的结构，以及 `[:-1]` 到底做什么

以下是根据 MiniMind `agent_rl.jsonl` 结构整理的**精简示例**。为便于阅读省略了此前历史消息，但字段结构不变：

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "",
      "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"calculate_math\",\"description\":\"计算数学表达式的结果\",\"parameters\":{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\"}},\"required\":[\"expression\"]}}}]"
    },
    {"role": "user", "content": "Compute 2045*6994 for me"},
    {"role": "assistant", "content": ""}
  ],
  "gt": ["14302730"]
}
```

`AgentRLDataset.parse_conversations` 的核心是：

```python
return messages[:-1], tools
```

Python 切片 `[:-1]` 表示“从开头取到倒数第一项之前”，所以具体结果是：

```python
conversations = [system, user, empty_assistant]
messages      = [system, user]
```

删掉的是最后一条占位 assistant，不是 user。原因是 RL rollout 要让当前策略自己生成 assistant 答案；如果把参考答案或空占位当成已有历史继续传入，就会破坏“从 user 后开始采样”的边界。

数据集最终返回：

```python
{
    "messages": [system_message, user_message],
    "tools": [calculate_math_schema],
    "gt": ["14302730"]
}
```

注意 `gt` 的角色：它不是逐 token label，不会像 SFT 那样强迫模型逐字模仿；它用于 rollout 完成后检查最终答案，从而产生 reward。

---

## 5. `apply_chat_template`：工具怎样真正进入 prompt

训练中实际调用：

```python
context = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools,
    open_thinking=open_thinking,
)
```

这个函数不是模型推理，它是**字符串渲染器**。它把：

- 对话 role 与 content；
- 工具名称、描述、参数类型、必填字段；
- assistant 即将开始生成的标记；

拼成 tokenizer 模板规定的文本，然后才由 tokenizer 变成整数 token IDs。

概念上的渲染结果类似：

```text
<system>
You may call the following functions:
calculate_math(expression: string) - 计算数学表达式的结果
</system>
<user>Compute 2045*6994 for me</user>
<assistant>
```

真实特殊标记和 schema 排版由 MiniMind tokenizer 的 chat template 决定，不能拿上面的概念文本冒充真实 `input_text`。调试时应直接打印 `input_text` 验证。

为什么不能假设模型“自己记得所有工具”？

1. 模型参数可能学到过某些工具名，但不知道当前运行环境究竟注册了什么。
2. schema 告诉它参数名、类型和必填项。
3. 工具版本和权限会变化，prompt 是当前会话的动态契约。
4. 不提供 schema 却允许模型随便猜函数名，会产生不存在的工具调用。

因此生产系统通常每轮传候选 schema，而不是依赖参数记忆。

---

## 6. 一条轨迹的完整三轮数据流

![Agent RL 多轮工具数据流](/images/lifeos-agent-training/agent_rl_multiturn_dataflow.svg)

下面用同一道题走完整个外部循环。这里的“轮”指一次模型生成，不是仅指 user/assistant 消息对。

### 6.1 第 0 步：准备初始状态

```python
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Compute 2045*6994 for me"},
]
tools = [calculate_math_schema]
```

chat template 和 tokenizer 得到：

```text
prompt_ids: [p0, p1, ..., p(P-1)]
```

`P` 是初始 prompt 的 token 数，取决于模板、schema、语言和 tokenizer。

### 6.2 第 1 轮：模型生成 tool call

理想输出：

```xml
<tool_call>{"name":"calculate_math","arguments":{"expression":"2045*6994"}}</tool_call>
```

rollout engine 同时记录每个生成 token 在旧策略下的 log probability：

```text
new_ids      = [a0, a1, ..., a(R1-1)]
new_logps    = [l0, l1, ..., l(R1-1)]
response_mask= [ 1,  1, ...,         1]
```

mask 为 1，因为这些 token 是模型采取的行动，要承担策略损失。

### 6.3 外部环境执行

Python 做四件事：

1. 正则找到 `<tool_call>`。
2. `json.loads` 得到 dict。
3. 根据名称执行 `calculate_math`。
4. 把结果编码为 JSON。

```json
{"result":"14302730"}
```

然后追加消息：

```python
messages.append({"role": "assistant", "content": tool_call_text})
messages.append({"role": "tool", "content": "{\"result\":\"14302730\"}"})
```

### 6.4 observation 怎样进入 token 序列

代码重新执行 chat template，得到包含历史调用和工具结果的新完整上下文。它只截取相对于已有 `prompt_ids + response_ids` 新增加的 token，记为 `obs_delta`：

```python
obs_delta = observe_ids[current_len:]
response_ids.extend(obs_delta)
response_mask.extend([0] * len(obs_delta))
response_old_logps.extend([0.0] * len(obs_delta))
```

observation mask 为 0，因为它是环境提供的事实，不是模型选择的动作。它仍进入下一轮上下文，影响模型后续 hidden state。

### 6.5 第 2 轮：模型读结果并回答

模型再次生成：

```text
2045 × 6994 = 14302730。
```

这些 final-answer token 同样是模型动作，mask 为 1。若模型又生成 tool call，循环继续；当前代码最多 `max_turns=3`，防止无限调用。

### 6.6 最大三轮不是三次工具一定都要用

循环提前结束条件是“本轮没有解析到 tool call”。常见轨迹：

```text
第 1 轮：tool call
环境：observation
第 2 轮：final answer
结束
```

只有在模型连续调用时才会进入第 3 轮。第 3 轮仍调用工具则 `unfinished=True`，reward 会额外扣 0.5。

---

## 7. 一条可复现的完整 Trace：从 JSON 到 361 个 Token

这一章不再使用随手假设的长度。下面的数据由当前 MiniMind tokenizer 在本机实际渲染、实际编码得到。复现脚本是 `scripts/trace_agent_rl_example.py`。

### 7.1 固定实验条件

```text
tokenizer: MiniMind model 目录中的当前 tokenizer
vocab size: 6400
open_thinking: false
system: 你是一个会正确调用工具的助手。
user: Compute 2045*6994 for me
candidate tools: [calculate_math]
```

工具 schema：

```json
{
  "type": "function",
  "function": {
    "name": "calculate_math",
    "description": "计算数学表达式的结果，支持加减乘除、幂运算",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "数学表达式，如123+456、2**10"
        }
      },
      "required": ["expression"]
    }
  }
}
```

### 7.2 Chat template 的真实输出

以下不是概念示意，而是当前 tokenizer 的 `apply_chat_template(..., tokenize=False)` 输出：

```text
<|im_start|>system
你是一个会正确调用工具的助手。

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如123+456、2**10"}}, "required": ["expression"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Compute 2045*6994 for me<|im_end|>
<|im_start|>assistant
<think>

</think>

```

这段文本共 763 个字符，编码后：

```text
prompt_ids.shape = [277]
P = 277 tokens
```

前 20 个真实 token ID：

```text
[1, 118, 4849, 234, 441, 1001, 508, 2476, 3266, 1618,
 296, 5127, 302, 234, 234, 38, 3450, 4605, 234, 234]
```

不要把字符数当 token 数。工具描述、JSON 标点和模板说明使这个简短问题变成 277-token prompt。

### 7.3 第一轮生成：35 个 action tokens

模型理想输出：

```xml
<tool_call>{"name":"calculate_math","arguments":{"expression":"2045*6994"}}</tool_call>
```

当前 tokenizer 实测：

```text
R1 = 35 tokens
response_mask = [1] × 35
response_old_logps = rollout 时记录的 35 个 log probability
```

其真实 token IDs 为：

```text
[21, 126, 37, 2533, 3149, 37, 102, 5811, 98, 112, 1831, 1812,
 37, 370, 106, 457, 1870, 3149, 126, 37, 3287, 115, 1592, 371,
 3149, 37, 1354, 4374, 45, 57, 3781, 55, 37, 5438, 22]
```

这里的每个 ID 都是一个 RL action。外层 `<tool_call>`、JSON 引号、工具名、参数名和数字都共同承担策略损失。

### 7.4 Parser 和 executor 的真实数据类型变化

```text
模型输出                         Python 类型
────────────────────────────────────────────────────────────
'<tool_call>...</tool_call>'     str
正则抽取标签内部                 str
json.loads(...)                  dict
call['name']                     str: 'calculate_math'
call['arguments']                dict: {'expression': '2045*6994'}
execute_tool(...)                dict: {'result': '14302730'}
json.dumps(...)                  str: '{"result": "14302730"}'
```

边界非常清楚：模型生成到 `str` 为止，Python 从解析 `str` 开始。模型没有直接获得函数对象，也没有直接执行权限。

### 7.5 工具回填不是只追加结果字符串

程序把 assistant 调用和 tool result 放回 `messages`，再重新渲染整个 chat template。新增的 `obs_delta` 实测为 35 tokens，解码后是：

```text
<|im_end|>
<|im_start|>user
<tool_response>
{"result": "14302730"}
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>

```

注意当前 MiniMind 模板把 `role="tool"` 渲染成包在 user 段内的 `<tool_response>`。这不是 Python 手工拼出来的，而是 tokenizer chat template 的协议约定。

对应数组追加：

```text
O = 35 tokens
response_mask += [0] × 35
response_old_logps += [0.0] × 35
```

为什么是 35 而不只是结果 JSON 的 12 tokens？因为增量还包含 assistant 结束标记、下一段 role 标记、`<tool_response>` 标签和新的 generation prompt。

### 7.6 第二轮最终回答：14 个 action tokens

```text
2045 × 6994 = 14302730。
```

实测：

```text
R2 = 14 tokens
response_mask += [1] × 14
```

完整单轨迹：

```text
区域       prompt     tool call     observation     final answer
长度          277             35              35               14
mask            0              1               0                1
```

所以：

{% raw %}
$$
L=277+35+35+14=361
$$
{% endraw %}

{% raw %}
$$
C=35+14=49
$$
{% endraw %}

总共 361 tokens，但只有 49 个由模型生成的 action tokens 直接进入 policy loss。其余 312 个 token 提供状态和上下文。

### 7.7 一条轨迹的数据对象如何逐步增长

```text
开始：
prompt_ids          277
response_ids          0
response_mask         0
old_logps             0

第一轮生成后：
prompt_ids          277
response_ids         35
response_mask        35 个 1
old_logps            35 个真实值

工具观察回填后：
prompt_ids          277
response_ids         70 = 35 action + 35 observation
response_mask        35 个 1 + 35 个 0
old_logps            35 个真实值 + 35 个 0.0

第二轮回答后：
prompt_ids          277
response_ids         84 = 35 + 35 + 14
response_mask        35 个 1 + 35 个 0 + 14 个 1
old_logps            35 个真实值 + 35 个 0.0 + 14 个真实值
```

这就是 `rollout_single` 返回给训练主循环的核心轨迹。

![单条样例从 JSON 到 Loss 的完整数据流](/images/lifeos-agent-training/agent_rl_end_to_end_trace.svg)

---

## 8. 从四条轨迹变成训练张量

### 8.1 先规定符号

| 符号 | 含义 |
|---|---|
| `B` | DataLoader 中原始 prompt 数量 |
| `G` | 每个 prompt 采样的轨迹数，默认 4 |
| `N=B×G` | 真正送入训练 forward 的轨迹数 |
| `P` | 初始 prompt token 数 |
| `R1` | 第一轮模型输出 token 数 |
| `O` | tool observation 新增 token 数 |
| `R2` | 第二轮模型输出 token 数 |
| `T` | 当前 batch 动态 padding 后总长度 |
| `V` | 词表大小，当前 MiniMind 默认 6400 |
| `H` | hidden size，默认配置 768 |

### 8.2 四条可验证的候选轨迹

同一 prompt 手工构造四条可被当前 reward 代码判定的轨迹。它们不是随机训练日志，而是为了逐项复算而固定的测试向量：

| 轨迹 | 第一轮行动 | 工具观察 | 最终回答 | `L` | action `C` |
|---|---|---|---|---:|---:|
| `τ1` | 1 次正确调用 | 正确结果 | 包含 GT | 361 | 49 |
| `τ2` | 1 次正确调用 | 正确结果 | “计算完成”但无 GT | 350 | 38 |
| `τ3` | 调用 `unknown_tool` | tool not found | 无法完成 | 351 | 39 |
| `τ4` | 重复 2 次正确调用 | 两个相同结果 | 包含 GT | 408 | 79 |

这些长度同样由当前 tokenizer 实测。最长轨迹 `T=408`，其他轨迹 padding 到 408。

以 `τ1` 为例：

```text
区域:  [ prompt 277 ][ tool_call 35 ][ observation 35 ][ final 14 ]
mask:  [  0 ... 0   ][    1 ... 1   ][     0 ... 0    ][  1 ... 1 ]
```

![Token 区域与损失 Mask](/images/lifeos-agent-training/agent_rl_token_mask.svg)

可训练 action token 数为：

{% raw %}
$$
C = R_1 + R_2 = 35 + 14 = 49
$$
{% endraw %}

不是总长度 361，也不是 response 区域的所有 token。

### 8.3 四条 rollout 动态 padding

四条测试轨迹长度为：

```text
[361, 350, 351, 408]
```

batch 内最大值 `T=408`，短序列右侧补 pad：

```text
input_ids.shape           = [4, 408]
full_response_masks.shape = [4, 408]
full_mask.shape           = [4, 408]
old_per_token_logps.shape = [4, 407]
```

为什么 old logps 少 1？因为长度为 `T` 的序列只有 `T-1` 个“用前面预测后一个 token”的位置。

### 8.4 `full_mask`、`response_mask` 和 `completion_mask`

- `full_mask = input_ids != pad_id`：控制 Transformer 不关注 padding。
- `full_response_masks`：未 shift 的 action 标记，与 `input_ids [N,T]` 对齐。
- `completion_mask = full_response_masks[:, 1:]`：控制哪些预测位置参与 policy loss。

为什么还要 `[:,1:]`？位置 `t` 的 logits 预测 `input_ids[t+1]`，因此 loss mask 也必须移动到 target token 一侧。前者是注意力有效性，后者是学习责任范围。把两者混用会造成严重 bug。

### 8.5 Packing 的真实执行顺序

当前训练代码严格按下面顺序处理每条轨迹：

```python
ids = prompt_ids + response_ids
mask = [0] * len(prompt_ids) + response_mask
old_logps = [0.0] * (len(prompt_ids) - 1) + response_old_logps

if len(ids) > max_total_len:
    ids = ids[-max_total_len:]          # 从左侧丢掉最老 token
    mask = mask[-max_total_len:]
    old_logps = old_logps[-(len(ids)-1):]

input_ids = dynamic_right_pad(ids)
full_response_masks = dynamic_right_pad(mask, value=0)
old_per_token_logps = dynamic_right_pad(old_logps, value=0.0)
```

这里有四个容易忽略的工程事实：

1. 超长时采用**左截断**，保留轨迹尾部；这可能丢掉 system 或部分工具 schema，必须监控截断率。
2. `input_ids` 长度是 `T`，old logps 长度是 `T-1`，因为第一个 token 没有前驱预测位置。
3. padding 在 batch 内动态进行，不是每条都无条件填到 `max_total_len`。
4. padding mask、response mask、old logps padding 值都为 0，但三者语义不同。

### 8.6 EOS 为什么还要再截一次 mask

代码寻找 action 区域中的第一个 EOS：

```python
is_eos = (input_ids[:, 1:] == eos_token_id) & completion_mask.bool()
completion_mask = completion_mask * (position <= first_eos_position)
```

若生成引擎或拼接过程在 EOS 后仍留下 token，它们不会参与策略损失。最终每条轨迹的有效 action 数：

```text
token_counts = completion_mask.sum(dim=1)   # [4]
valid_rows = token_counts > 0               # [4] bool
```

只有至少包含一个有效 action token 的轨迹进入 batch 平均。

---

## 9. Transformer 内部每一步的维度

当前默认 MiniMind 配置可从 `model/model_minimind.py` 核对：

```text
V=6400, H=768, layers=8
attention heads=8, KV heads=4, head_dim=96
```

以下使用第 8 章四条可验证轨迹组成的 `N=4,T=408` batch。

### 9.1 Embedding

```text
input_ids                         [4, 408]
Embedding table                   [6400, 768]
hidden_states                     [4, 408, 768]
```

每个整数 token ID 查表得到一个 768 维向量。

### 9.2 Q、K、V 投影

8 个 query head，每个 96 维：

```text
Q [4, 408, 8, 96]
K [4, 408, 4, 96]
V [4, 408, 4, 96]
```

这是 grouped-query attention：K/V 头少于 Q 头，代码会把 4 个 KV 头重复匹配到 8 个 Q 头。注意力分数概念维度：

```text
scores [4, 8, 408, 408]
```

最后一个 `408×408` 表示每个位置对其他位置的相关性。因果 mask 保证位置 `t` 看不到未来 token。实现若使用高效 attention，不一定真的把完整分数矩阵长期保存在显存中，但逻辑形状仍可这样理解。

### 9.3 8 层之后到词表 logits

每层输出仍保持：

```text
hidden_states [4, 408, 768]
```

语言模型头：

```text
lm_head weight [6400, 768]
logits          [4, 408, 6400]
```

`logits[n,t,v]` 是第 `n` 条轨迹、第 `t` 个位置对词表第 `v` 个 token 的未归一化分数。

训练代码丢掉最后一个预测位置：

```python
logits = res.logits[:, :-1, :]                 # [4, 407, 6400]
target = input_ids[:, 1:]                      # [4, 407]
```

再计算：

```python
per_token_logps = F.log_softmax(logits, dim=-1) \
    .gather(2, target.unsqueeze(-1)).squeeze(-1)
```

维度变化：

```text
log_softmax(logits)             [4, 407, 6400]
target.unsqueeze(-1)            [4, 407, 1]
gather 后                       [4, 407, 1]
squeeze 后 per_token_logps      [4, 407]
```

`gather` 不是取最大概率 token，而是取“轨迹中实际出现的那个 target token”的 log probability。

### 9.4 用元素数量理解显存，而不只背 shape

对这个 `N=4,T=408` 的例子：

| 张量 | 元素数 | FP16 仅数据大小 |
|---|---:|---:|
| hidden `[4,408,768]` | 1,253,376 | 约 2.39 MiB |
| logits `[4,408,6400]` | 10,444,800 | 约 19.92 MiB |
| 逻辑 attention scores `[4,8,408,408]` | 5,326,848 | 约 10.16 MiB/层 |

训练显存远大于这三个数之和，因为还包括 8 层中间激活、梯度、AdamW 状态、模型参数、reference model 和 rollout 缓存。高效 attention 也可能不显式保存完整 scores，所以表格用于建立数量级，不是精确显存账单。

---

## 10. 概率、log probability 和序列概率

### 10.1 从百分数开始

假设某一步模型给三个候选 token：

```text
"<tool_call>"  70%
"答案"         20%
"我"           10%
```

策略采样到 `<tool_call>` 的概率是 `0.7`。log probability 为：

{% raw %}
$$
\log(0.7) \approx -0.357
$$
{% endraw %}

概率越接近 1，log probability 越接近 0；概率越小，log probability 越负。

### 10.2 为什么使用 log

一段 40-token tool call 的联合概率是条件概率连乘：

{% raw %}
$$
P(a_{1:40}|s)=\prod_{t=1}^{40}\pi_\theta(a_t|s_t)
$$
{% endraw %}

许多小数连乘很容易数值下溢。取对数后，乘法变加法：

{% raw %}
$$
\log P(a_{1:40}|s)=\sum_{t=1}^{40}\log\pi_\theta(a_t|s_t)
$$
{% endraw %}

这就是为什么训练代码记录每个 token 的 log probability。

### 10.3 Softmax 从 logits 得到概率

对某位置词表中的第 `i` 个 token：

{% raw %}
$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}
$$
{% endraw %}

所有 `V=6400` 个概率之和为 1。`log_softmax` 直接稳定地计算 `log p_i`。

---

## 11. Reward：整条轨迹如何打分

当前 `calculate_rewards` 分两条支路。

### 11.1 有 tool call 的轨迹

主要组成：

1. tool 标签数量不闭合：每个差异扣 0.5。
2. 工具名必须在当前样本 `valid_names` 中。
3. 参数必须通过 `CHECK_ARGS`。
4. 合法调用数量与 `len(gt)` 对齐时加 0.5，否则按 gap 扣分。
5. 最终答案每覆盖一个 GT，按比例共享最多 2.5 分。
6. 达到最大轮数仍未完成，扣 0.5。
7. 重复文本扣分。
8. 总分裁剪到 `[-3,3]`。

对示例 `gt=["14302730"]`，若只有一次合法调用且最终答案包含结果：

```text
标签错误             0.0
tool_gap == 0        +0.5
命中 1/1 个 GT       +2.5
unfinished=False      0.0
无重复                0.0
--------------------------------
reward                3.0
```

这正好触及上限 3.0。

若调用合法，但最后只说“计算完成”而没有 `14302730`：

```text
工具对齐 +0.5，GT +0.0，总分约 +0.5
```

因此系统不只奖励“会调用”，还奖励“使用工具结果完成任务”。

### 11.2 没有 tool call 的轨迹

当前实现会根据回答长度、thinking 格式、可选 reward model 和重复度评分。一个必须正视的代码边界是：

> 这条 no-tool 分支没有用 `gt` 检查“本题本来是否必须调用工具”。

因此如果某条需要计算的轨迹完全不调用工具但写出一段格式正常的文字，仍可能获得一定 reward。严谨改进应在数据中增加 `requires_tool`，或在 no-tool 分支对非空 `tools/gt` 增加漏调用惩罚。

### 11.3 Reward 不是逐 token 答案

reward 是每条轨迹一个标量：

```text
rewards.shape = [N] = [B×G]
```

它不会直接告诉模型“JSON 的第 17 个 token 错了”。策略梯度通过这条轨迹中所有 action token 的 log probability 分配信用，这叫**轨迹级粗粒度信用分配**。

---

## 12. 为什么每个问题生成 4 条：组相对 Advantage

使用第 8 章四条固定轨迹。下面的 reward 逐项按当前 `calculate_rewards` 有工具分支计算，且假设重复惩罚为 0：

| 轨迹 | 行为摘要 | Reward | 代码路径 |
|---|---|---:|---|
| `τ1` | 1 次合法调用，答案命中 GT | 3.0 | `+0.5 tool +2.5 GT` |
| `τ2` | 1 次合法调用，答案不含 GT | 0.5 | `+0.5 tool +0 GT` |
| `τ3` | 1 次非法工具，答案不含 GT | -1.0 | `tool_gap=2 → -1.0` |
| `τ4` | 重复 2 次合法调用，答案命中 GT | 2.0 | `-0.5 tool gap +2.5 GT` |

`τ3` 为什么 `tool_gap=2`？`valid_call_count=0`、`len(gt)=1`、调用总数为 1：

{% raw %}
$$
tool\_gap=|0-1|+\max(0,1-0)=2
$$
{% endraw %}

### 12.1 组均值

{% raw %}
$$
\mu=\frac{3.0+0.5-1.0+2.0}{4}=1.125
$$
{% endraw %}

### 12.2 组标准差

代码使用总体标准差 `unbiased=False`：

{% raw %}
$$
\sigma=\sqrt{\frac{1}{4}\sum_{i=1}^{4}(R_i-\mu)^2}\approx1.5155
$$
{% endraw %}

### 12.3 标准化 Advantage

代码为：

```python
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

手算近似：

| 轨迹 | 计算 | Advantage |
|---|---|---:|
| `τ1` | `(3.0-1.125)/(1.5155+0.0001)` | `+1.2371` |
| `τ2` | `(0.5-1.125)/(1.5155+0.0001)` | `-0.4124` |
| `τ3` | `(-1.0-1.125)/(1.5155+0.0001)` | `-1.4020` |
| `τ4` | `(2.0-1.125)/(1.5155+0.0001)` | `+0.5773` |

解释：

- `A>0`：比同题其他答法好，应提高这条轨迹 action token 的概率。
- `A<0`：比同题其他答法差，应降低其概率。
- 比较发生在同一个 prompt 内，减少“有些题天生容易、有些题天生难”造成的尺度干扰。

如果 4 条 reward 完全相同，标准差接近 0，所有 advantage 接近 0，策略几乎学不到差异。这说明 rollout 必须有一定探索多样性，reward 也必须有区分度。

![四条 Rollout 从 Reward 到 Loss](/images/lifeos-agent-training/agent_rl_reward_to_loss.svg)

---

## 13. old policy、current policy、reference policy 各做什么

代码里容易混淆三个概率来源。

### 13.1 old policy：行为策略

rollout 时生成 token 的策略，记录：

```text
old_per_token_logps [N,T-1]
```

训练 forward 时参数可能尚未变化，也可能因复用 rollout、梯度累积或引擎同步产生差异。importance ratio 用它校正“数据由旧策略采样、现在由新策略学习”的差别。

### 13.2 current policy：正在更新的模型

它产生 `per_token_logps`，有梯度，反向传播最终更新的就是它的参数 `θ`。

### 13.3 reference policy：冻结锚点

`ref_model` 从相同初始权重加载，设为 `eval().requires_grad_(False)`。它不更新，用来约束 current policy 不要偏离基础语言能力太远。

MiniMind 当前入口默认：

```text
from_weight = full_sft
```

所以对当前代码，准确表述是“Agent RL 从 full SFT 权重启动”；不能没有证据地说它默认从 DPO 权重启动。

---

## 14. Importance Ratio：新旧策略差了多少

对第 `t` 个实际 action token：

{% raw %}
$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
=\exp\left(\log\pi_\theta-\log\pi_{old}\right)
$$
{% endraw %}

直觉：

- `r=1`：新旧策略给这个 token 的概率相同。
- `r=1.2`：新策略概率是旧策略的 1.2 倍。
- `r=0.5`：新策略概率只有旧策略的一半。

代码：

```python
ratio = torch.exp(per_token_logps - old_per_token_logps)
```

observation 的 old logp 被填为 0，但它的 completion mask 也是 0，因此不会参与最终平均；不能单看 old logp 的占位值判断它是否训练。

---

## 15. KL 惩罚：不要为了高分把语言能力改坏

代码先计算：

{% raw %}
$$
d_t=\log\pi_{ref}(a_t|s_t)-\log\pi_\theta(a_t|s_t)
$$
{% endraw %}

再使用非负近似：

{% raw %}
$$
KL_t=e^{d_t}-d_t-1
$$
{% endraw %}

因为对任意实数 `x`，`e^x ≥ 1+x`，所以 `e^x-x-1 ≥ 0`。

当 current policy 与 reference policy 接近时，`d_t≈0`，KL 约为 0；偏离越大，惩罚通常越大。`β` 控制约束强度，默认 `0.1`。

KL 的作用不是保证事实正确，而是限制策略漂移。例如只追逐工具 reward 可能让模型过度调用工具、输出僵硬标签或损坏普通聊天能力，reference policy 提供“不要离原模型太远”的软护栏。

---

## 16. 当前 MiniMind 的 CISPO loss：逐符号拆开

当前默认 `loss_type="cispo"` 分支：

```python
clamped_ratio = torch.clamp(ratio, max=epsilon_high).detach()
per_token_loss = -(
    clamped_ratio * advantages.unsqueeze(1) * per_token_logps
    - beta * per_token_kl
)
```

对应：

{% raw %}
$$
\ell_{i,t}
=-\left[
\operatorname{stopgrad}(\min(r_{i,t},\epsilon_{high}))
A_i\log\pi_\theta(a_{i,t}|s_{i,t})
-\beta KL_{i,t}
\right]
$$
{% endraw %}

### 16.1 为什么最外面有负号

我们想最大化“高 advantage 动作的 log probability”，但 PyTorch optimizer 默认最小化 loss，所以整体取负。

### 16.2 为什么 `detach`

`detach()` 把裁剪后的 ratio 当权重，不让梯度沿 ratio 分支传播。策略梯度主要沿 `log πθ` 传播。它不是把整个 token 梯度切断；`log πθ` 仍然有梯度。

### 16.3 一个高分 token 的手算

取 `τ1` 中某个 tool-call token，假设：

```text
A=+1.2371
log πθ=-0.7
r=1.1
KL=0.02
β=0.1
epsilon_high=5.0
```

则：

{% raw %}
$$
\ell=-[1.1\times1.2371\times(-0.7)-0.1\times0.02]
\approx0.9546
$$
{% endraw %}

先忽略 KL 对 `log πθ` 的附加梯度，主项导数：

{% raw %}
$$
\frac{\partial\ell}{\partial\log\pi_\theta}
=-1.1\times1.2371\approx-1.3608
$$
{% endraw %}

梯度下降执行 `参数 ← 参数 - 学习率 × 梯度`。梯度为负，更新倾向于提高这个 token 的 log probability，因此下一次更可能生成它。

若使用 `τ3` 的 `A=-1.4020`，主项导数变为正数，梯度下降倾向于降低该动作的 log probability。

### 16.4 `<tool_call>` 每个 token 到底算不算 loss

答案：**算**，只要它是模型 rollout 生成且 mask 为 1。

当前实现中：

| 区域 | 是模型生成的吗 | 进入上下文吗 | Policy loss mask |
|---|---:|---:|---:|
| system + tool schema | 否 | 是 | 0 |
| user question | 否 | 是 | 0 |
| `<tool_call>...` | 是 | 是 | 1 |
| `role=tool` observation | 否，环境给出 | 是 | 0 |
| final answer | 是 | 是 | 1 |
| padding | 否 | 否 | 0 |

tool call 字符串没有 SFT label，但有 RL policy loss。两者不是一回事：

- SFT：已知标准下一个 token，做交叉熵模仿。
- Agent RL：没有标准逐 token 路径，用整条轨迹 reward 评价实际采样动作。

### 16.5 四种位置的逐 token loss 账本

下面构造一个只有四个预测位置的微型账本，用来展示 mask。`A=1.2371` 来自前面的固定 rewards；log probability、ratio 与 KL 是**受控教学值**，不是训练日志。

| 位置 | 区域 | mask | `log πθ` | ratio | KL | 未 mask 的 token loss |
|---|---|---:|---:|---:|---:|---:|
| 1 | prompt 中的 user token | 0 | -0.30 | 1.00 | 0.00 | 不计 |
| 2 | tool call 中的工具名 | 1 | -0.70 | 1.10 | 0.02 | 0.9546 |
| 3 | tool observation 数字 | 0 | -0.10 | 占位 | 占位 | 不计 |
| 4 | final answer 数字 | 1 | -0.40 | 1.05 | 0.015 | 0.5211 |

位置 4 的计算：

{% raw %}
$$
\ell_4=-[1.05\times1.2371\times(-0.40)-0.1\times0.015]
\approx0.5211
$$
{% endraw %}

mask 后这条微型轨迹只剩位置 2 和 4：

{% raw %}
$$
L_i=\frac{0.9546+0.5211}{2}\approx0.7379
$$
{% endraw %}

prompt 和 observation 的模型 forward 仍然计算了 hidden state 与 logits，但乘上 mask 后不直接贡献 policy loss。这就是“参与前向上下文”和“承担学习责任”的区别。

---

## 17. 从 per-token loss 得到一个标量 loss

当前实现不是直接对所有 `N×(T-1)` 位置求平均，而是两层平均。

### 17.1 每条轨迹只平均有效 action token

对第 `i` 条轨迹：

{% raw %}
$$
L_i=\frac{\sum_t m_{i,t}\ell_{i,t}}{\sum_t m_{i,t}}
$$
{% endraw %}

`m` 是 completion mask。这样 observation、prompt、padding 不贡献 loss。

### 17.2 再对有效轨迹平均

{% raw %}
$$
L_{policy}=\frac{1}{N_{valid}}\sum_i L_i
$$
{% endraw %}

代码：

```python
policy_loss = (
    (per_token_loss * completion_mask).sum(dim=1)
    / token_counts.clamp(min=1)
)[valid_rows].mean()
```

若使用 MoE，还会加 `aux_loss`；普通 dense MiniMind 中它为 0。梯度累积时：

{% raw %}
$$
L=\frac{L_{policy}+L_{aux}}{K}
$$
{% endraw %}

其中 `K=accumulation_steps`。

### 17.3 为什么先按轨迹平均

若把全 batch 所有 token 直接平均，长回答会天然占更大权重。先对每条轨迹平均，再对轨迹平均，使长短轨迹更接近同等投票权。这是实现选择，不是唯一可能方案。

---

## 18. 反向传播怎样改到 6400 个词的概率

从标量 loss 开始，自动微分沿计算图反向：

```text
scalar loss
  ← selected token log probabilities [N,T-1]
  ← log_softmax logits              [N,T-1,V]
  ← lm_head hidden states           [N,T,H]
  ← 8 Transformer blocks
  ← token embeddings
```

虽然 `gather` 只取了实际 token 的 log probability，但 softmax 的归一化使梯度与同一位置所有词表 logits 有关。直觉上，提高实际 token 的概率，必须相对压低一部分其他 token 的概率，因为总和必须为 1。

随后：

```python
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

- `backward` 计算梯度。
- gradient clipping 防止一次异常 batch 造成过大更新。
- AdamW 根据梯度和动量更新参数。
- scheduler 调整学习率。
- 清空梯度，为下一个 step 做准备。

rollout engine 不一定每一步立即同步，当前代码在保存间隔或特定节点执行 `update_policy(model)`；理解训练新鲜度时要结合具体引擎。

---

## 19. 一次训练 step 的完整伪代码

```python
# 1. 数据：B 条 prompt，每条带候选 tools 与 gt
messages_batch, tools_batch, gt_batch = batch

# 2. 采样：每题 G=4 条，最多 3 轮
trajectories = []
for messages, tools in zip(messages_batch, tools_batch):
    for _ in range(G):
        trajectories.append(
            rollout_single(messages.copy(), tools, max_turns=3)
        )

# 3. 打分：每条轨迹一个 reward
rewards = calculate_rewards(trajectories, gt_batch, tools_batch)  # [B*G]

# 4. 同题组内标准化
grouped = rewards.view(B, G)                                    # [B,G]
advantages = normalize_within_each_row(grouped).reshape(B * G) # [B*G]

# 5. 动态 padding，送 current 与 reference 模型
input_ids, attention_mask, action_mask = pack(trajectories)
logp     = current_policy(input_ids)  # [B*G,T-1]
ref_logp = reference_policy(input_ids)# [B*G,T-1]

# 6. 新旧比率、KL、逐 token loss
ratio = exp(logp - old_logp)
kl = exp(ref_logp - logp) - (ref_logp - logp) - 1
token_loss = -(stopgrad(min(ratio, eps_high)) * advantage * logp - beta * kl)

# 7. mask 后先按轨迹平均，再按 batch 平均
loss = masked_trajectory_mean(token_loss, action_mask)

# 8. 更新策略模型；reference 不更新
loss.backward()
optimizer.step()
```

这段伪代码省略了 EOS 截断、MoE aux loss、梯度累积、分布式训练和保存，但保留了核心数学数据流。

---

## 20. SFT 与 Agent RL 对同一 tool call 学习方式的根本区别

### 20.1 SFT 数据

SFT 直接给标准 assistant 文本：

```xml
<tool_call>{"name":"calculate_math","arguments":{"expression":"2045*6994"}}</tool_call>
```

每个 assistant token 有 label，交叉熵：

{% raw %}
$$
L_{SFT}=-\frac{1}{C}\sum_t m_t\log\pi_\theta(y_t|y_{\lt t},x)
$$
{% endraw %}

它擅长教格式、工具名和典型参数。

### 20.2 Agent RL 数据

Agent RL 数据只提供 prompt、tools、GT。模型自由采样多个行为，环境执行后给结果，reward 判断整条轨迹。它擅长优化：

- 什么时候值得调用；
- 多个候选里哪个更好；
- 错误后如何继续；
- observation 是否被正确用于最终答案；
- 完成率与调用成本之间的折中。

### 20.3 能不能靠 Agent RL 从零学会写函数

要区分两件事：

- **生成一个已注册函数的调用字符串**：可以通过 SFT + RL 学习。
- **创造并部署一个全新的 Python 函数**：当前系统不会自动做到。

模型当然可能生成看似 Python 的代码，但 executor 只执行注册表内允许的工具。让模型自行写代码并执行需要沙箱、依赖控制、权限隔离、测试和审计，是另一类高风险系统，不应把它与普通 Tool Calling 混为一谈。

---

## 21. 模型有记忆吗：参数、上下文、外部记忆三层

### 21.1 参数记忆

训练把统计规律写进权重，例如“数学问题可能需要计算器”“tool call 常用 JSON 格式”。它不是可精确检索、可随时编辑的数据库。

### 21.2 上下文记忆

本轮 `messages` 中的历史、schema 和 observation 都在 context window 内。只要保留在 prompt 中，模型可以利用；超过上下文或新会话未带入，就看不到。

### 21.3 外部持久记忆

Obsidian、数据库、向量检索、日历才是可持久保存和查询的记忆。模型必须通过工具读取它们。

所以“模型自身没记忆吗”的准确回答是：有参数化统计记忆和当前上下文记忆，但没有天然可靠、可更新、跨会话的个人事实存储。

---

## 22. 怎样设计真正训练“工具选取”的数据

### 22.1 只有唯一正确工具：学习使用

```text
问题：2045*6994
候选：[calculate_math]
```

模型不需要在工具间竞争，主要学调用格式和参数。

### 22.2 加干扰工具：学习选择

```text
问题：2045*6994
候选：[calculate_math, search_notes, get_weather]
```

reward 只认可 `calculate_math`，这才训练相对选择。

### 22.3 无工具负样本：学习克制

```text
问题：你好，介绍一下你自己
候选：[calculate_math, search_notes]
期望：直接回答
```

如果所有样本都要求调用，模型会形成“看见 schema 就调用”的偏差。

### 22.4 参数近邻样本：学习 schema

```text
正确：{"expression":"17.66*1.1"}
错误：{"query":"17.66*1.1"}
错误：{"expression":17.66*1.1}  # schema 要 string 时类型不符
```

### 22.5 多轮恢复样本：学习观察与修正

```text
第 1 轮：参数缺失
工具返回：{"error":"expression is required"}
第 2 轮：补齐参数重新调用
第 3 轮：给最终回答
```

若数据和 reward 从未出现错误恢复，不能期待模型凭空学会稳定恢复。

---

## 23. 为什么 Agent RL 慢

SFT 通常每个 batch 做一次主要模型 forward/backward；Agent RL 每个 prompt 还要：

1. 采样 `G=4` 条完整轨迹。
2. 每条轨迹可能生成 2 到 3 轮。
3. 中间做 tokenizer 重渲染和 Python 工具执行。
4. current policy 做训练 forward。
5. reference policy 再做一次 forward 计算 KL。
6. 可选 reward model 还要额外推理。

若原始数据有 `D` 条、1 个 epoch、每条 `G=4`，至少会产生约 `4D` 条 rollout；多轮生成的 autoregressive 解码又是逐 token 的，所以耗时主要在 rollout，而不只是 backward。

对 3090 Ti，务实优化顺序是：

1. 先用 20 到 100 条 smoke set 验证数据、reward、mask。
2. 把 `max_gen_len` 和 `max_total_len` 控制在真实分位数，不盲目开到最大。
3. 先 `B=1,G=4` 保证能跑，再根据显存提高 batch。
4. bf16 支持不稳定时使用 fp16，并保留 grad clip。
5. 调试阶段提高日志可见性，正式训练再降低打印频率。
6. rollout 成为瓶颈时再评估 SGLang，不要在逻辑未验证时先引入复杂引擎。

---

## 24. 当前实现中值得审计的风险

### 24.1 Python `eval` 风险

MiniMind mock calculator 使用受限 `eval`，即使移除了 builtins，也不应直接当作生产安全计算器。LifeOS-Agent 应使用 AST 白名单解析。

### 24.2 正则解析不是完整协议解析器

正则足够做 demo，但嵌套、截断、多标签和恶意内容会变复杂。生产系统应增加 schema validation 和明确错误回填。

### 24.3 Reward hacking

模型会寻找“高分捷径”。例如 no-tool 分支不校验 GT，可能奖励看似正常但未完成任务的回答。任何 reward 都要用对抗测试验证。

### 24.4 轨迹级 reward 信用过粗

同一个 advantage 广播给一条轨迹所有 action token。高分轨迹里偶然出现的不良 token 也可能一起被增强。可通过过程奖励、工具参数级 reward、错误分类和更细 mask 改善。

### 24.5 工具副作用

训练阶段应优先只读、确定性、可回滚工具。真实发邮件、删文件、交易等工具必须审批和幂等保护，不能让 rollout 随机探索真实副作用。

---

## 25. 如何验证自己真的掌握了

### 第一遍：只看数据流

不推公式，能口述：

```text
数据取 messages[:-1] 和 tools
→ schema 进入 prompt
→ 每题采样 4 条
→ 模型生成 tool call
→ Python 执行并回填 observation
→ 模型最终回答
→ reward
→ 组内 advantage
→ masked token loss
→ backward
```

### 第二遍：重算数字

独立算出：

- `277+35+35+14=361`；
- action token 是 `35+14=49`；
- 四个 reward 均值 `1.125`；
- `τ1` advantage 约 `1.2371`；
- `logits [4,408,6400]` shift 后 `[4,407,6400]`；
- observation mask 是 0，但仍进入下一轮状态。

### 第三遍：对着代码追变量

在 `trainer/train_agent.py` 依次找到：

```text
response_ids
response_mask
response_old_logps
full_response_masks
completion_mask
rewards
grouped_rewards
advantages
per_token_logps
ratio
per_token_kl
per_token_loss
policy_loss
```

如果能解释每个变量的 shape、来源和是否有梯度，就已经掌握核心。

---

## 26. 一页速查表

| 问题 | 最短正确答案 |
|---|---|
| 工具是谁筛的？ | 推理时 router 筛候选；当前训练时数据构造者预置候选；LLM 决定实际调用。 |
| 工具都要传吗？ | 不必。通常只传与当前请求相关且获准使用的候选 schema。 |
| 模型直接执行 Python 吗？ | 不。模型生成 token，外部 parser/executor 执行。 |
| tool call token 算 loss 吗？ | Agent RL 中模型生成的 tool call token mask=1，算 policy loss。 |
| observation 算 loss 吗？ | mask=0，不直接算；但进入上下文并影响后续 token。 |
| final answer 算 loss 吗？ | 算，模型生成 token mask=1。 |
| GT 是 token label 吗？ | 不是。它用于验证最终结果并产生 reward。 |
| 为什么 4 条 rollout？ | 同题比较，构造组相对 advantage。 |
| 为什么需要 ref model？ | KL 约束策略不要偏离基础模型太远。 |
| `[:-1]` 删除谁？ | 删除最后一条 assistant 占位，让策略自己生成。 |
| 模型能凭 RL 创造新工具吗？ | 当前系统不能；它只能请求 executor 已注册的工具。 |
| 最后 loss 是什么 shape？ | 标量 `[]`，由 masked per-token loss 先按轨迹、再按 batch 平均。 |

---

## 27. 参考资料与证据边界

### 项目代码

- MiniMind `dataset/lm_dataset.py`：`AgentRLDataset` 如何解析 `tools` 并返回 `messages[:-1]`。
- MiniMind `trainer/train_agent.py`：多轮 rollout、tool observation mask、reward、group advantage、GRPO/CISPO loss。
- MiniMind `model/model_minimind.py`：默认词表、hidden size、attention heads 与 logits 维度。
- LifeOS-Agent `lifeos_agent/router.py`：推理时关键词候选工具路由。
- LifeOS-Agent `lifeos_agent/main.py`：schema 渲染、工具回填和最多三轮的外部循环。

### 原始论文

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)：推理轨迹与环境行动交错。
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)：学习何时调用、调用哪个 API、参数是什么以及怎样利用结果。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：采样环境交互数据并优化 surrogate objective 的 PPO 框架。
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)：提出 GRPO 作为 PPO 的变体，使用组相对信号并降低 critic 相关开销。

### 最后一个边界提醒

本文对 MiniMind 的描述是**当前仓库实现分析**，并不声称所有 Agent RL 系统都必须采用相同 reward、mask 或 CISPO 公式。理解工程时必须同时问两句话：

1. 论文中的算法原本怎样定义？
2. 当前代码实际上怎样实现？

能持续区分这两层，才是真正掌握，而不是背诵名词。
