---
title: LifeOS-Agent 从零到独立训练：Transformer、SFT、DPO、PPO、GRPO 与 Agent RL 完整推导
date: 2026-07-23 20:30:00
updated: 2026-07-23 20:30:00
mathjax: true
description: "从张量、概率、梯度和 Transformer 开始，以 17.66 涨停价工具调用为贯穿样例，逐步推导 SFT、DPO、PPO、GRPO、CISPO 与 Agent RL，并落到 MiniMind/LifeOS-Agent 的真实数据、张量维度、训练代码、3090 Ti 调参与发布验收。"
categories:
  - AI与大模型
  - 模型训练
tags:
  - LifeOS-Agent
  - MiniMind
  - Transformer
  - SFT
  - DPO
  - PPO
  - GRPO
  - Agent-RL
  - Tool-Calling
type: handbook
difficulty: progressive
review_status: published
---

> 这不是一篇只列名词的综述，而是一份可以从第一页连续学习到独立训练的主教材。我们从“标量、向量、概率和梯度是什么”开始，最终把一条多轮工具轨迹算到逐 token policy loss，并给出真实 MiniMind/LifeOS-Agent 代码位置、3090 Ti 训练步骤和面试验收标准。
>
> 全文只有一个主问题：**一条用户消息如何经过 tokenizer、Transformer、工具环境和训练目标，最终改变模型参数？**

<!-- more -->

![LifeOS-Agent 从数据到训练与评测的完整流水线](/images/lifeos-agent-zero-to-mastery/lifeos_training_pipeline.svg)

> 目标：不是“知道几个名词”，而是能够从一条 JSON 数据出发，独立说出它如何变成张量、如何经过 Transformer、如何产生工具调用、如何计算 loss、如何反向传播，以及如何完成训练和验收。

---

## 全书路线图

![从数学基础到 Agent RL 部署的学习地图](/images/lifeos-agent-zero-to-mastery/agent_rl_learning_map.svg)

章节依赖关系：

| 要掌握的目标 | 必须先掌握 |
| --- | --- |
| SFT loss | Token、logits、softmax、causal shift |
| DPO loss | SFT mask、序列 log probability、sigmoid |
| PPO | Policy gradient、value、advantage、importance ratio |
| GRPO | Policy gradient、组均值与标准差 |
| Agent RL | Tool Calling 外部循环、GRPO/PPO 风格 token loss |
| 独立训练 | 数据、shape、优化器、日志和独立测试集 |

不要跳过依赖：如果还不能解释 $[B,T,H]\to[B,T,V]$，直接背 GRPO 公式不会形成可迁移能力。

---

## 0. 怎么使用这份教材

这份教材采用一条固定主线：

```text
用户：17.66 涨停价是多少？
候选工具：calculate_math
模型动作：<tool_call>{...}</tool_call>
环境观察：{"result": 19.43}
模型最终动作：17.66 元的涨停价约为 19.43 元。
```

学习时始终追问四件事：

1. **当前数据是什么类型？** 是字符串、Python dict、token id，还是浮点张量？
2. **当前张量是什么形状？** 每一维分别代表什么？
3. **谁在执行？** Router、Tokenizer、Transformer、Parser 还是 Python 工具？
4. **当前 token 是否计算 loss？** 如果计算，监督信号来自哪里？

推荐学习三遍：

- 第一遍只看数据流和角色边界。
- 第二遍手算所有微型数值例子。
- 第三遍对照代码，闭卷讲完整流程。

---

## 0.1 数学预备：从标量到梯度

这一节不是额外的高等数学负担，而是后面所有 shape 和 loss 的语言。

### 0.1.1 标量、向量、矩阵、张量

| 名称 | 示例 | shape | 在模型中的例子 |
| --- | --- | --- | --- |
| 标量 | $3.2$ | `[]` | 最终 loss、learning rate |
| 向量 | $[0.2,0.7,0.1]$ | `[3]` | 一个位置的概率分布 |
| 矩阵 | $X\in\mathbb R^{T\times H}$ | `[T,H]` | 一条序列的 hidden states |
| 三维张量 | $X\in\mathbb R^{B\times T\times H}$ | `[B,T,H]` | 一个 batch 的 hidden states |
| 四维张量 | $A\in\mathbb R^{B\times N\times T\times T}$ | `[B,N,T,T]` | 多头 attention 权重 |

“维度”有两个常见含义：

1. 张量有几根轴，例如 `[B,T,H]` 是三维张量。
2. 某根轴的长度，例如 hidden dimension 是 $H=768$。

面试时要说清是哪一种，避免只说“维度是 3”。

### 0.1.2 矩阵乘法为什么改变最后一维

若：

{% raw %}
$$
X\in\mathbb R^{B\times T\times H},\qquad
W\in\mathbb R^{H\times V}
$$
{% endraw %}

最后两维执行矩阵乘法：

{% raw %}
$$
Z=XW\in\mathbb R^{B\times T\times V}
$$
{% endraw %}

中间的 $H$ 被求和消掉。元素形式：

{% raw %}
$$
Z_{b,t,v}=\sum_{h=1}^{H}X_{b,t,h}W_{h,v}
$$
{% endraw %}

这正是语言模型头把每个 token 的 768 维表示转换成 6400 个词表 logits 的过程。

### 0.1.3 概率、条件概率与期望

概率满足：

{% raw %}
$$
0\le p(x)\le1,\qquad \sum_xp(x)=1
$$
{% endraw %}

条件概率：

{% raw %}
$$
p(y\mid x)=\frac{p(x,y)}{p(x)}
$$
{% endraw %}

期望是按概率加权的平均：

{% raw %}
$$
\mathbb E[X]=\sum_xp(x)x
$$
{% endraw %}

强化学习目标中的：

{% raw %}
$$
\mathbb E_{\tau\sim\pi_\theta}[R(\tau)]
$$
{% endraw %}

表示“按当前策略产生轨迹，长期平均能拿到多少 reward”，不是某一条轨迹的分数。

### 0.1.4 为什么处处使用 log

若一条回答的 token 概率是：

{% raw %}
$$
0.1\times0.2\times0.05=0.001
$$
{% endraw %}

取 log 后乘法变加法：

{% raw %}
$$
\log0.1+\log0.2+\log0.05=\log0.001
$$
{% endraw %}

优点：

1. 避免大量小概率连乘下溢。
2. 序列分数可由 token log probability 求和。
3. 导数更容易计算。

几个必须熟悉的值：

```text
log(1)   = 0
log(0.5) ≈ -0.693
log(0.1) ≈ -2.303
```

概率越接近 1，负对数越接近 0；概率越接近 0，负对数越大。

### 0.1.5 导数、梯度与链式法则

导数表示输入变化一点，输出怎样变化：

{% raw %}
$$
f(x)=x^2,\qquad\frac{df}{dx}=2x
$$
{% endraw %}

模型有大量参数，所有偏导数组成梯度：

{% raw %}
$$
\nabla_\theta\mathcal L
=\left[
\frac{\partial\mathcal L}{\partial\theta_1},
\ldots,
\frac{\partial\mathcal L}{\partial\theta_n}
\right]
$$
{% endraw %}

链式法则把“loss 对 logits 的影响”一路传回每层参数：

{% raw %}
$$
\frac{\partial\mathcal L}{\partial W}
=\frac{\partial\mathcal L}{\partial Z}
\frac{\partial Z}{\partial H}
\frac{\partial H}{\partial W}
$$
{% endraw %}

`loss.backward()` 就是在计算这条反向链；`optimizer.step()` 才真正修改参数。

---

## 1. 最基础的概念：模型到底在做什么

### 1.1 大语言模型本质上是“下一个 token 预测器”

给定 token 序列：

{% raw %}
$$
x_1,x_2,\ldots,x_t
$$
{% endraw %}

模型估计下一个 token 的条件概率：

{% raw %}
$$
p_\theta(x_{t+1}\mid x_1,\ldots,x_t)
$$
{% endraw %}

这里：

- $x_t$ 是一个整数 token id。
- $\theta$ 是模型全部可训练参数。
- 输出不是一个词，而是词表中每个 token 的概率。

一句完整回答的概率用链式法则写成：

{% raw %}
$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{\lt t})
$$
{% endraw %}

因为很多小概率连乘容易数值下溢，训练时使用 log probability：

{% raw %}
$$
\log p_\theta(x_{1:T})=\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{\lt t})
$$
{% endraw %}

### 1.2 模型没有内置 Python 执行器

模型能生成：

```xml
<tool_call>{"name":"calculate_math","arguments":{"expression":"round(17.66*1.1,2)"}}</tool_call>
```

但它只是生成了一段文本。真正执行数学表达式的是外部 Python 程序。必须分清：

| 角色 | 做什么 | 不做什么 |
| --- | --- | --- |
| Router | 从注册工具中筛候选 schema | 不生成答案，不执行工具 |
| Tokenizer | 文本与 token id 互转 | 不理解数学，不选择工具 |
| LLM | 生成 tool-call 或自然语言 token | 不直接运行 Python |
| Parser | 从标签中提取并解析 JSON | 不决定答案是否正确 |
| Executor | 校验参数并调用 Python handler | 不训练模型 |
| Agent 外部循环 | 串起生成、执行、回填和再生成 | 不是 Transformer 的一层 |

### 1.3 三种“记忆”不要混淆

1. **参数记忆**：训练后写进权重的统计模式，例如工具调用格式。
2. **上下文记忆**：当前 `messages` 进入 prompt 后，模型本轮可以读取的信息。
3. **外部记忆**：Obsidian、数据库或文件，由工具读取，再回填给模型。

模型不会自动记住上一场已经结束的对话。若信息既不在权重、当前上下文，也不在可访问的外部存储中，模型就不知道它。

---

## 2. 从文本到 token：Tokenizer 与 Chat Template

### 2.1 Token 不是汉字，也不一定是完整单词

Tokenizer 将文本映射为整数序列：

{% raw %}
$$
\text{text}\xrightarrow{\text{tokenizer}}[x_1,x_2,\ldots,x_T]
$$
{% endraw %}

再把 token id 解码回文本：

{% raw %}
$$
[x_1,x_2,\ldots,x_T]\xrightarrow{\text{decode}}\text{text}
$$
{% endraw %}

`T` 是 token 数，不是字符数。相同字符长度的两段文本可能有不同 token 数。

### 2.2 Messages 是结构化对象，模型实际读取的是渲染后字符串

Python 中的初始消息：

```python
messages = [
    {"role": "system", "content": "你是 LifeOS-Agent..."},
    {"role": "user", "content": "17.66 涨停价是多少？"},
]
```

工具 schema：

```python
tools = [{
    "type": "function",
    "function": {
        "name": "calculate_math",
        "description": "计算简单数学表达式",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }
    }
}]
```

调用：

```python
input_text = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=True,
    open_thinking=False,
)
```

Chat template 的作用是把角色、内容、工具 schema 和 assistant 起始标记渲染成模型训练时熟悉的文本协议。之后才执行：

```python
inputs = tokenizer(input_text, return_tensors="pt")
```

得到：

```text
input_ids.shape      = [B, T]
attention_mask.shape = [B, T]
```

本项目推理时通常 $B=1$。

### 2.3 为什么只传候选工具

本项目先执行：

```python
candidate_tool_names = select_tool_names(user_input)
tools = get_tools_by_names(candidate_tool_names) or None
```

它有三个意义：

1. 减少 prompt token 和推理成本。
2. 减少相似工具之间的误选。
3. 普通聊天传 `tools=None`，避免诱导模型无端调用工具。

Router 是确定性的工程组件。LLM 仍然负责在候选集合中决定“调用哪个、参数是什么、是否调用”。如果训练 Agent RL 时 prompt 没有提供某个工具的 schema，模型无法仅凭工具注册表知道它当前可用。

### 2.4 Tokenizer、Template 与模型权重是三件不同的东西

常见误区是把“Tokenizer 会渲染工具”说成“模型已经会调用工具”。实际上：

```text
Chat template：规定工具 schema 和 role 如何排版
Tokenizer：规定字符串如何切成 token ids
Model weights：决定下一个 token 的概率
```

即使 template 能渲染 `<tools>`，若权重没见过相关模式，模型仍可能不会输出合法 `<tool_call>`。反过来，即使权重学过某种协议，推理时 template 换成不兼容格式，也会造成分布偏移。

训练与推理必须保持一致：

| 环节 | 训练 | 推理 |
| --- | --- | --- |
| system/user/assistant role | Chat template 渲染 | 同一模板渲染 |
| tools schema | 作为 `tools=` 传入 | 作为 `tools=` 传入 |
| tool call | assistant 监督文本 | 模型生成文本 |
| tool result | 已知 observation | Python 现场执行后回填 |
| final answer | assistant 监督文本 | 模型第二轮生成 |

### 2.5 `conversations[:-1]` 到底删除哪一条

Agent RL 数据常写成：

```python
conversations = [
    {"role": "system", "content": "...", "tools": "[...]"}, # 索引 0
    {"role": "user", "content": "17.66 涨停价是多少？"},       # 索引 1
    {"role": "assistant", "content": ""},                    # 索引 2
]
```

Python 切片：

```python
messages = conversations[:-1]
```

含义是“从开头取到最后一个元素之前”，结果只保留索引 0 和 1：

```python
messages = [
    {"role": "system", "content": "...", "tools": "[...]"},
    {"role": "user", "content": "17.66 涨停价是多少？"},
]
```

被删的是末尾空 assistant 占位，不是 user，也不是工具 schema。原因是 Agent RL 要让 current policy 在线生成 assistant 动作。如果保留空 assistant 再添加 generation prompt，容易产生协议重复。

MiniMind `AgentRLDataset.parse_conversations` 同时返回：

```python
return messages[:-1], tools
```

而 `__getitem__` 返回：

```python
{
    "messages": messages,
    "tools": tools,
    "gt": sample["gt"],
}
```

三者用途不同：

- `messages`：作为 rollout 初始状态。
- `tools`：渲染 schema，并限制环境可执行工具。
- `gt`：给 reward 检查结果或期望行为，不直接作为 assistant label 喂给模型。

这与 SFT 的根本区别是：SFT 保留标准 assistant 内容并构造 labels；Agent RL 删除空占位，在线采样动作，再通过 reward 学习。

---

## 3. Transformer 数据流与精确维度

![MiniMind Transformer 的核心张量维度](/images/lifeos-agent-zero-to-mastery/network_tensor_dimensions.svg)

### 3.1 本项目教学配置

本文采用仓库模型配置中的典型参数：

| 符号 | 含义 | 数值 |
| --- | --- | ---: |
| $B$ | batch size | 例：1 或 4 |
| $T$ | 序列长度 | 最大训练长度常用 768 |
| $H$ | hidden size | 768 |
| $L$ | Transformer 层数 | 8 |
| $V$ | vocabulary size | 6400 |
| $N_q$ | query heads | 8 |
| $N_{kv}$ | key/value heads | 4 |
| $D$ | head dimension | 96 |
| $I$ | MLP intermediate size | 2432 |

因为：

{% raw %}
$$
N_qD=8\times96=768=H
$$
{% endraw %}

### 3.2 Embedding

输入 token：

{% raw %}
$$
X_{ids}\in\mathbb{N}^{B\times T}
$$
{% endraw %}

Embedding 矩阵：

{% raw %}
$$
E\in\mathbb{R}^{V\times H}
$$
{% endraw %}

查表后：

{% raw %}
$$
X=E[X_{ids}]\in\mathbb{R}^{B\times T\times H}
$$
{% endraw %}

若 $B=4,T=768,H=768$，则激活张量有：

{% raw %}
$$
4\times768\times768=2,359,296
$$
{% endraw %}

个元素。FP16 每个元素约 2 字节，仅这个张量约 4.5 MiB；训练还要保存每层中间激活、梯度和优化器状态，所以总显存远大于这一项。

### 3.3 Self-Attention

先做线性投影：

{% raw %}
$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$
{% endraw %}

多头拆分后，query 为：

{% raw %}
$$
Q\in\mathbb{R}^{B\times N_q\times T\times D}
$$
{% endraw %}

GQA 中 key/value 头更少：

{% raw %}
$$
K,V\in\mathbb{R}^{B\times N_{kv}\times T\times D}
$$
{% endraw %}

注意力分数：

{% raw %}
$$
S=\frac{QK^\top}{\sqrt D}+M
$$
{% endraw %}

其中：

{% raw %}
$$
S\in\mathbb{R}^{B\times N_q\times T\times T}
$$
{% endraw %}

$M$ 是 causal mask。未来位置被加上负无穷，因此第 $t$ 个位置只能读取 $1\ldots t$：

{% raw %}
$$
M_{ij}=\begin{cases}
0,&j\le i\\
-\infty,&j\gt i
\end{cases}
$$
{% endraw %}

Softmax 后得到注意力权重：

{% raw %}
$$
A_{ij}=\frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}}
$$
{% endraw %}

再聚合 value：

{% raw %}
$$
O=AV
$$
{% endraw %}

合并多头后回到：

{% raw %}
$$
O\in\mathbb{R}^{B\times T\times H}
$$
{% endraw %}

### 3.4 MLP、残差与输出 logits

每层可简化理解为：

{% raw %}
$$
X'=X+\operatorname{Attention}(\operatorname{Norm}(X))
$$
{% endraw %}

{% raw %}
$$
X''=X'+\operatorname{MLP}(\operatorname{Norm}(X'))
$$
{% endraw %}

经过 $L=8$ 层后：

{% raw %}
$$
H_L\in\mathbb{R}^{B\times T\times H}
$$
{% endraw %}

语言模型头映射到词表：

{% raw %}
$$
Z=H_LW_{vocab}^\top
$$
{% endraw %}

{% raw %}
$$
Z\in\mathbb{R}^{B\times T\times V}
$$
{% endraw %}

`Z` 就是 logits。每个位置有 $V=6400$ 个未经归一化的分数。

### 3.5 RMSNorm、RoPE 与 SwiGLU 分别解决什么

#### RMSNorm

对 hidden vector $x\in\mathbb R^H$：

{% raw %}
$$
\operatorname{RMS}(x)
=\sqrt{\frac1H\sum_{i=1}^Hx_i^2+\epsilon}
$$
{% endraw %}

{% raw %}
$$
\operatorname{RMSNorm}(x)
=g\odot\frac{x}{\operatorname{RMS}(x)}
$$
{% endraw %}

它控制激活尺度，帮助深层训练稳定。$g\in\mathbb R^H$ 是可学习缩放参数。

#### RoPE

Attention 本身不知道 token 顺序。RoPE 根据位置旋转 query/key 的二维分量，使点积同时携带相对位置信息。对某一对分量可写为：

{% raw %}
$$
\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}
=
\begin{bmatrix}
\cos(m\omega_i)&-\sin(m\omega_i)\\
\sin(m\omega_i)&\cos(m\omega_i)
\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}
$$
{% endraw %}

$m$ 是 token 位置。RoPE 不改变 shape，只改变 Q/K 数值。

#### SwiGLU

现代 MLP 常使用门控结构：

{% raw %}
$$
\operatorname{SwiGLU}(x)
=\operatorname{SiLU}(xW_g)\odot(xW_u)W_d
$$
{% endraw %}

典型中间形状：

```text
x                 [B,T,H]
xW_g, xW_u        [B,T,I]
element-wise gate [B,T,I]
project down      [B,T,H]
```

Attention 负责 token 间信息混合，MLP 负责每个 token 位置内部的非线性特征变换。

---

## 4. Softmax、交叉熵与反向传播

### 4.1 从 logits 到概率

设某个位置只有三个候选 token，logits 为：

{% raw %}
$$
z=[2,1,0]
$$
{% endraw %}

Softmax：

{% raw %}
$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$
{% endraw %}

因为 $e^2\approx7.389,e^1\approx2.718,e^0=1$，所以：

{% raw %}
$$
p\approx[0.665,0.245,0.090]
$$
{% endraw %}

### 4.2 交叉熵

如果正确 token 是第二个，one-hot label 为：

{% raw %}
$$
y=[0,1,0]
$$
{% endraw %}

交叉熵：

{% raw %}
$$
\mathcal{L}_{CE}=-\sum_i y_i\log p_i=-\log(0.245)\approx1.407
$$
{% endraw %}

模型越相信正确 token，loss 越低。

### 4.3 为什么梯度是 $p-y$

Softmax 与交叉熵合并后，对 logit $z_k$ 的梯度为：

{% raw %}
$$
\frac{\partial\mathcal L}{\partial z_k}=p_k-y_k
$$
{% endraw %}

因此：

- 正确类别：$p_k-1\lt 0$，梯度下降会提高正确 logit。
- 错误类别：$p_k-0\gt 0$，梯度下降会降低错误 logit。

这条公式是理解“训练如何改变下一个 token 概率”的核心。

### 4.4 Causal LM shift

输入：

```text
[BOS, 我, 要, 调用, 工具, EOS]
```

位置关系：

```text
输入位置      BOS   我    要    调用   工具
监督目标       我    要   调用   工具   EOS
```

代码层面的 shape：

```text
logits        [B, T, V]
shift_logits  [B, T-1, V]
labels        [B, T]
shift_labels  [B, T-1]
loss          scalar
```

### 4.5 从一个 batch 到一次参数更新

一次 SFT step 的顺序：

```python
optimizer.zero_grad()
res = model(input_ids, labels=labels)
loss = res.loss / accumulation_steps
loss.backward()
clip_grad_norm_(model.parameters(), grad_clip)
optimizer.step()
```

更精确地说：

1. Forward 计算 logits 与 loss。
2. Backward 计算每个参数的 gradient。
3. Gradient clipping 限制梯度范数，防止异常大步。
4. Optimizer 根据梯度和历史状态更新参数。
5. 清空梯度，准备下一步。

### 4.6 SGD 与 AdamW

最简单的梯度下降：

{% raw %}
$$
\theta_{t+1}=\theta_t-\eta g_t
$$
{% endraw %}

其中 $g_t=\nabla_\theta\mathcal L_t$，$\eta$ 是 learning rate。

Adam 维护梯度的一阶、二阶动量：

{% raw %}
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
{% endraw %}

{% raw %}
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
{% endraw %}

偏差修正：

{% raw %}
$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$
{% endraw %}

AdamW 更新：

{% raw %}
$$
\theta_{t+1}
=\theta_t
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-\eta\lambda\theta_t
$$
{% endraw %}

最后一项是 decoupled weight decay。Learning rate 太大会破坏已有能力，太小则目标模式学不动；epoch 不是越多越好，重复小数据很容易过拟合。

### 4.7 梯度累积为什么要除以步数

若累积 $K$ 个 micro batches，希望得到它们平均 loss 的梯度：

{% raw %}
$$
\mathcal L_{avg}=\frac1K\sum_{k=1}^{K}\mathcal L_k
$$
{% endraw %}

所以每个 micro step 执行：

```python
loss = loss / accumulation_steps
loss.backward()
```

累积到第 $K$ 步再 `optimizer.step()`。如果不除以 $K$，梯度规模相当于扩大 $K$ 倍，等效学习率也会变化。

---

## 5. SFT：用标准答案教模型输出格式和内容

### 5.1 典型 Tool Calling SFT 样本

```json
{
  "conversations": [
    {"role":"system","content":"你是 LifeOS-Agent"},
    {"role":"user","content":"17.66 涨停价是多少？"},
    {"role":"assistant","content":"<tool_call>{\"name\":\"calculate_math\",\"arguments\":{\"expression\":\"round(17.66*1.1,2)\"}}</tool_call>"},
    {"role":"tool","content":"{\"result\":19.43}"},
    {"role":"assistant","content":"17.66 元的涨停价约为 19.43 元。"}
  ]
}
```

工具 schema 还必须进入训练 prompt，否则模型看不到工具描述与参数约束。

### 5.2 SFT mask

假设渲染后 token 分为：

```text
[system][user][assistant tool_call][tool result][assistant final]
```

常见监督策略：

```text
system             mask=0
user               mask=0
assistant toolcall mask=1
tool observation   mask=0
assistant final    mask=1
```

用 $m_t\in\{0,1\}$ 表示监督 mask：

{% raw %}
$$
\mathcal L_{SFT}
=-
\frac{1}{\sum_t m_t}
\sum_t m_t\log p_\theta(y_t\mid x_{\lt t})
$$
{% endraw %}

所以 `<tool_call>`、工具名和 JSON 参数都是 assistant 生成的 token，通常参与 SFT loss。工具返回值是外部环境提供的条件，不应伪装成模型动作。

### 5.3 SFT 学到了什么

- 标签格式：`<tool_call>...</tool_call>`。
- 工具名与参数键。
- 哪类问题通常需要工具。
- 看到 `role=tool` 后如何组织最终回答。

SFT 不保证自由探索，也不保证模型永远选对工具。它主要是模仿训练数据中的标准行为。

### 5.4 为什么少量身份数据会快速生效

“你叫 LifeOS-Agent”是低熵、重复且局部的映射。少量高重复样本就能明显提高相关 token 的概率。但这不等于泛化：改一种问法、增加上下文或换领域后可能失效。稳定能力需要覆盖表达多样性、负样本和真实分布。

### 5.5 MiniMind `SFTDataset` 逐行数据流

仓库实现可压缩为：

```text
JSONL 一行
-> sample['conversations']
-> pre_processing_chat
-> 解析 system.tools / assistant.tool_calls
-> apply_chat_template(add_generation_prompt=False)
-> prompt string
-> tokenizer(prompt).input_ids
-> 截断到 max_length
-> pad 到固定长度
-> generate_labels
-> input_ids [T], labels [T]
-> DataLoader 堆成 [B,T]
```

`generate_labels` 先创建：

```python
labels = [-100] * len(input_ids)
```

再定位每段 assistant 起止 token，只把 assistant 区间复制为真实 label。PyTorch CrossEntropy 默认忽略 `label=-100` 的位置。

一个微型例子：

```text
位置             0       1       2       3       4       5
内容          system   user   asst_bos   工具名   参数   asst_eos
input_ids       11      22       33       44      55      66
labels         -100    -100     -100       44      55      66
```

再经过 causal shift 后，位置 2 的 logit 预测位置 3 的 token 44，位置 3 预测 55，位置 4 预测 66。要区分：

- `labels=-100` 是监督 mask。
- causal attention mask 是禁止看未来。
- padding mask 是禁止关注补齐位置。

它们解决三个不同问题。

### 5.6 截断如何悄悄破坏工具样本

如果一条完整多轮样本长度是 900，但 `max_seq_len=768`，尾部 132 个 token 会消失。可能出现：

1. 保留 tool call，却删掉 tool result 和 final answer。
2. JSON 参数被截断，形成不闭合标签。
3. 最终回答监督比例下降。

所以训练前不仅要数 JSONL 行数，还要统计 token length 的 p50、p90、p95、max 和截断比例。

---

## 6. DPO：从 chosen/rejected 学习偏好

### 6.1 数据结构

同一个 prompt 有一个偏好答案和一个拒绝答案：

```json
{
  "prompt": "17.66 涨停价是多少？",
  "chosen": "<tool_call>{正确工具和参数}</tool_call>",
  "rejected": "大约是 20 元，不需要计算。"
}
```

### 6.2 序列 log probability

对回答 $y=(y_1,\ldots,y_T)$：

{% raw %}
$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{\lt t})
$$
{% endraw %}

定义当前策略对 chosen/rejected 的差：

{% raw %}
$$
\Delta_\theta
=\log\pi_\theta(y^+\mid x)-\log\pi_\theta(y^-\mid x)
$$
{% endraw %}

冻结 reference model 的对应差：

{% raw %}
$$
\Delta_{ref}
=\log\pi_{ref}(y^+\mid x)-\log\pi_{ref}(y^-\mid x)
$$
{% endraw %}

DPO loss：

{% raw %}
$$
\mathcal L_{DPO}
=-\log\sigma\left(\beta(\Delta_\theta-\Delta_{ref})\right)
$$
{% endraw %}

### 6.3 直觉

- 如果当前模型比 reference 更偏爱 chosen，括号变大，loss 变小。
- 如果当前模型更偏爱 rejected，loss 变大。
- $\beta$ 控制偏好优化相对 reference 的强度。
- reference 提供锚点，避免模型仅通过整体提高所有回答概率来“作弊”。

### 6.4 SFT 与 DPO 根本区别

| 方法 | 数据 | 问题 |
| --- | --- | --- |
| SFT | prompt + 标准 assistant token | “标准答案是什么？” |
| DPO | prompt + chosen + rejected | “两条答案哪条更好？” |

DPO 通常建立在已经会基本格式的 SFT 模型上。让一个从未学过工具协议的模型只靠少量偏好对，从零稳定学会 XML/JSON 工具格式，并不可靠。

### 6.5 DPO 数值算例：从四个序列分数到 loss

假设当前策略：

```text
log pi_theta(chosen|x)   = -2.0
log pi_theta(rejected|x) = -3.0
```

所以：

{% raw %}
$$
\Delta_\theta=-2-(-3)=1.0
$$
{% endraw %}

Reference：

```text
log pi_ref(chosen|x)   = -2.4
log pi_ref(rejected|x) = -2.8
```

所以：

{% raw %}
$$
\Delta_{ref}=-2.4-(-2.8)=0.4
$$
{% endraw %}

取 $\beta=0.1$：

{% raw %}
$$
z=\beta(\Delta_\theta-\Delta_{ref})
=0.1(1.0-0.4)=0.06
$$
{% endraw %}

{% raw %}
$$
\sigma(0.06)\approx0.515
$$
{% endraw %}

{% raw %}
$$
\mathcal L_{DPO}=-\log0.515\approx0.663
$$
{% endraw %}

若模型开始更偏爱 rejected，$\Delta_\theta$ 会下降，$z$ 下降，sigmoid 下降，loss 上升。反向传播因此推动 chosen 相对概率提高。

长度陷阱：序列 log probability 是 token logp 之和，长回答天然可能更负。具体实现是否按 token 平均会影响长度偏好，阅读代码时必须确认 reduction，而不能只背 DPO 公式。

---

## 7. 从最大化 Reward 到 Policy Gradient

### 7.1 目标函数

策略 $\pi_\theta$ 生成动作序列 $\tau$，环境返回奖励 $R(\tau)$：

{% raw %}
$$
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[R(\tau)]
$$
{% endraw %}

使用 log-derivative trick：

{% raw %}
$$
\nabla_\theta\pi_\theta(\tau)
=\pi_\theta(\tau)\nabla_\theta\log\pi_\theta(\tau)
$$
{% endraw %}

得到 REINFORCE：

{% raw %}
$$
\nabla_\theta J
=\mathbb E\left[R(\tau)\nabla_\theta\log\pi_\theta(\tau)\right]
$$
{% endraw %}

直觉：

- 高奖励轨迹：提高产生这些动作 token 的概率。
- 低奖励轨迹：降低产生这些动作 token 的概率。

### 7.2 Baseline 与 Advantage

直接使用 reward 方差很大，因此减去不依赖当前动作的 baseline：

{% raw %}
$$
A_t=R_t-b(s_t)
$$
{% endraw %}

若 $A_t\gt 0$，动作比预期好；若 $A_t\lt 0$，动作比预期差。

---

## 8. PPO：Actor、Critic、GAE 与 Clip

### 8.1 PPO 中的模型角色

- Actor/current policy：正在训练的语言模型。
- Old policy：产生本批 rollout 的行为策略快照。
- Critic/value model：预测状态价值 $V(s_t)$。
- Reference policy：约束模型不要偏离基础语言能力太远。
- Reward model 或规则：为回答评分。

### 8.2 TD error 与 GAE

一步 TD error：

{% raw %}
$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$
{% endraw %}

GAE：

{% raw %}
$$
\hat A_t
=\delta_t+(\gamma\lambda)\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}+\cdots
$$
{% endraw %}

$\gamma$ 控制未来奖励折扣，$\lambda$ 控制偏差与方差权衡。

### 8.3 Importance ratio

Rollout 来自 old policy，但更新的是 current policy：

{% raw %}
$$
r_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}
=\exp(\log\pi_\theta-\log\pi_{old})
$$
{% endraw %}

### 8.4 PPO clipped objective

{% raw %}
$$
L^{clip}_t
=\min\left(
r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
$$
{% endraw %}

训练最小化其负值。Clip 防止一次更新把策略推得过远。

Critic loss 可写为：

{% raw %}
$$
\mathcal L_V=(V_\phi(s_t)-\hat R_t)^2
$$
{% endraw %}

PPO 的代价是系统复杂：既要训练 actor，又要训练 critic，还要维护 old/reference 与 rollout。

### 8.5 PPO 单 token 数值算例

假设某 token：

```text
old probability     = 0.20
current probability = 0.26
advantage           = +2.0
epsilon             = 0.20
```

Ratio：

{% raw %}
$$
r=0.26/0.20=1.30
$$
{% endraw %}

允许上界是 $1+\epsilon=1.20$：

{% raw %}
$$
rA=1.30\times2=2.60
$$
{% endraw %}

{% raw %}
$$
\operatorname{clip}(r,0.8,1.2)A=1.2\times2=2.40
$$
{% endraw %}

取最小值 2.40，说明即使 current 已大幅提高该好动作概率，收益也被截住，避免继续激进更新。

若 $A=-2$，正负号会改变 min 的含义：策略不能通过把坏动作概率一次降得极低来获得无限收益。理解 PPO clip 时必须分别代入正、负 advantage，不能只看曲线。

### 8.6 GAE 三步手算

设：

```text
gamma  = 0.9
lambda = 0.8
delta0 = 1.0
delta1 = 0.5
delta2 = -0.2
```

则：

{% raw %}
$$
\hat A_0
=1.0+(0.9\times0.8)0.5+(0.9\times0.8)^2(-0.2)
$$
{% endraw %}

{% raw %}
$$
=1.0+0.36-0.10368=1.25632
$$
{% endraw %}

它把后续 TD error 按 $(\gamma\lambda)^k$ 衰减后传回当前动作。

---

## 9. GRPO：用同组相对成绩替代 Critic

### 9.1 同题生成一组回答

对同一个问题生成 $G=4$ 条轨迹，奖励假设为：

{% raw %}
$$
R=[3,1,-1,-3]
$$
{% endraw %}

组均值：

{% raw %}
$$
\mu=\frac{3+1-1-3}{4}=0
$$
{% endraw %}

若采用总体标准差：

{% raw %}
$$
\sigma
=\sqrt{\frac{(3-0)^2+(1-0)^2+(-1-0)^2+(-3-0)^2}{4}}
=\sqrt5
$$
{% endraw %}

标准化 advantage：

{% raw %}
$$
A_i=\frac{R_i-\mu}{\sigma+\varepsilon}
$$
{% endraw %}

约为：

{% raw %}
$$
A=[1.342,0.447,-0.447,-1.342]
$$
{% endraw %}

同一问题内部直接比较，省去单独训练 critic。

### 9.2 GRPO 仍然计算 token loss

“没有 critic”不等于“不计算 loss”。轨迹 advantage 会作用到该轨迹的有效动作 token：

{% raw %}
$$
\mathcal L_{policy}
=-\frac{1}{|\mathcal A|}
\sum_{t\in\mathcal A}
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A
$$
{% endraw %}

其中 $\mathcal A$ 是模型生成的 action-token 位置集合。

---

## 10. Tool Calling 外部循环完整数据流

![LifeOS-Agent 多轮 Tool Calling 数据流](/images/lifeos-agent-zero-to-mastery/agent_rl_multiturn_dataflow.svg)

### 10.1 第一轮

```text
user_input: str
    ↓ router
candidate_tool_names: list[str]
    ↓ schema lookup
tools: list[dict] | None
    ↓ apply_chat_template
input_text: str
    ↓ tokenizer
input_ids: LongTensor [1,T1]
    ↓ model.generate
generated_ids: LongTensor [1,T1+K1]
    ↓ slice + decode
response: str
```

模型第一轮输出：

```xml
<tool_call>{"name":"calculate_math","arguments":{"expression":"round(17.66*1.1,2)"}}</tool_call>
```

### 10.2 Parser 与 Executor

Parser 使用标签定位 JSON，再执行 `json.loads`：

```text
str -> dict
```

Executor 的边界检查：

1. 工具名是否注册。
2. `arguments` 是 dict、JSON string、`None` 还是非法类型。
3. JSON 能否解析。
4. handler 执行是否异常。
5. 数学表达式是否只包含 AST 白名单节点。

执行结果：

```python
{"result": 19.43}
```

### 10.3 回填与第二轮

第一轮模型文本作为 assistant message 保留：

```python
messages.append({"role": "assistant", "content": response})
```

工具结果以 tool role 回填：

```python
messages.append({
    "role": "tool",
    "content": json.dumps(result, ensure_ascii=False),
})
```

此时 `messages` 从 2 条变为 4 条。第二次重新执行 chat template 和 tokenizer：

```text
[system][user][assistant tool_call][tool result][assistant generation prompt]
```

模型据此生成最终回答。工具结果不是偷偷写进模型权重，而是作为本轮上下文被 attention 读取。

### 10.4 最大轮数

```python
for turn in range(1, args.max_turns + 1):
```

最多三轮是安全边界，不代表必须调用三次。模型一旦不再输出 `<tool_call>`，外部循环就把该文本作为最终回答；持续调用工具则在上限停止，避免死循环。

---

## 11. Agent RL：把工具环境放进在线 Rollout

![Agent RL 从 Reward 到逐 Token Loss](/images/lifeos-agent-zero-to-mastery/agent_rl_reward_to_loss.svg)

### 11.1 状态、动作、观察与轨迹

对 Agent：

- 状态 $s_t$：当前 prompt 和完整消息历史。
- 动作 $a_t$：模型生成的 token，包括 tool-call 和 final answer。
- 环境：parser、executor 和外部数据源。
- 观察 $o_t$：`role=tool` 返回结果。
- 轨迹 $\tau$：多轮状态、动作、观察的完整序列。

一条轨迹：

{% raw %}
$$
\tau=(s_0,a_0,o_0,s_1,a_1)
$$
{% endraw %}

对应：

```text
s0: system + tools + user
a0: <tool_call>...</tool_call>
o0: {"result":19.43}
s1: s0 + a0 + o0
a1: 最终自然语言回答
```

### 11.2 Action mask

将多轮内容打包成一条序列后，可概念性标记：

```text
prompt/system/user       action_mask=0
assistant tool_call      action_mask=1
tool observation         action_mask=0
assistant final answer   action_mask=1
padding                   action_mask=0
```

因此 tool-call 每个有效 token 都参与策略 loss，而 tool result 不参与。观察仍会进入模型输入，影响后续 final-answer token 的条件概率。

### 11.3 Reward 不是 loss

Reward 是环境对整条轨迹的评价，例如：

{% raw %}
$$
R=R_{format}+R_{name}+R_{args}+R_{result}+R_{final}-P_{invalid}
$$
{% endraw %}

可能检查：

- `<tool_call>` 标签是否完整。
- 工具名是否在候选 schema 中。
- 参数是否满足 JSON schema。
- 调用次数是否合理。
- 最终回答是否使用真实工具结果。
- 是否泄露思维、重复或出现非法工具。

Reward 本身通常不可微。训练先把 reward 转成 advantage，再乘到可微的 token log probability 上。

### 11.4 本项目 v1 的奖励漏洞

旧逻辑中，需要工具却没有有效调用的轨迹，仍可能获得长度、思考格式或 reward-model 奖励。结果是 reward 看似上升，但严格工具调用变成 `0/3`。

修复后的硬约束：

```text
标签不闭合                       -> reward = -3
需要工具但没有 tool call         -> reward = -3
调用候选集合之外的工具           -> reward = -3
参数校验失败                     -> reward = -3
```

这说明训练指标必须和任务验收同时看，不能用平均 reward 替代工具成功率。

### 11.5 从轨迹 reward 到逐 token loss

同题生成 $G$ 条轨迹，得到每条轨迹 advantage $A_i$。轨迹 $i$ 的所有有效 action token 共享轨迹级信号，但每个 token 有自己的 probability ratio：

{% raw %}
$$
r_{i,t}=\exp\left(
\log\pi_\theta(a_{i,t}\mid s_{i,t})
-\log\pi_{old}(a_{i,t}\mid s_{i,t})
\right)
$$
{% endraw %}

加入 reference KL：

{% raw %}
$$
D_{KL}(\pi_\theta\|\pi_{ref})
\approx
\frac{\pi_{ref}}{\pi_\theta}
-\log\frac{\pi_{ref}}{\pi_\theta}-1
$$
{% endraw %}

一个简化的 CISPO/GRPO 风格 token objective 可表示为：

{% raw %}
$$
\ell_{i,t}
=-
\left[
\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_i
-\beta D_{KL,i,t}
\right]
$$
{% endraw %}

只对 action mask 为 1 的位置求平均：

{% raw %}
$$
\mathcal L_i
=\frac{\sum_t m_{i,t}\ell_{i,t}}
{\sum_t m_{i,t}}
$$
{% endraw %}

再对有效轨迹或 batch 求平均：

{% raw %}
$$
\mathcal L
=\frac1N\sum_{i=1}^{N}\mathcal L_i
$$
{% endraw %}

最后：

```text
scalar loss
-> autograd backward
-> 每个参数的 gradient
-> optimizer.step
-> 更新 current policy
```

Old policy 与 reference policy 在这一更新步骤中不接收梯度。

### 11.6 为什么 Agent RL 慢

普通 SFT 对一个 batch 通常做一次训练前向和一次反向。Agent RL 还需要：

1. 每个 prompt 生成 $G$ 条候选。
2. 每条候选可能执行多轮生成。
3. CPU/Python 工具和 parser 介入。
4. current、old、reference 计算 token log probability。
5. 动态 padding、mask、reward 和组内统计。
6. 最后才进行反向传播。

因此“只有 380 条数据”也可能耗时很久，因为计算量取决于实际生成 token 和 rollout 次数，而不仅是 JSONL 行数。

### 11.7 一条 Agent RL 样本在代码中的精确对象变化

磁盘数据：

```json
{
  "conversations": [
    {"role":"system","content":"...","tools":"[...]"},
    {"role":"user","content":"17.66 涨停价是多少？"},
    {"role":"assistant","content":""}
  ],
  "gt": ["19.43"]
}
```

Dataset 返回：

```text
messages: list[dict]，最后空 assistant 已删除
tools:    list[dict]，候选 schema
gt:       list，reward 验证目标
```

假设一个 batch 有 $B=2$ 个 prompt，每题生成 $G=4$ 条：

```text
原始问题数                 B = 2
rollout 轨迹数              N = B*G = 8
每条轨迹长度               Li，各不相同
动态 padding 后最大长度     Lmax
input_ids                  [8,Lmax]
full_mask                  [8,Lmax]
response/action mask       [8,Lmax]
logits                     [8,Lmax,V]
shifted logits             [8,Lmax-1,V]
per_token_logps            [8,Lmax-1]
completion_mask            [8,Lmax-1]
rewards                    [8]
grouped_rewards            [2,4]
advantages                 [8]
per_token_loss             [8,Lmax-1]
policy_loss                scalar
```

关键代码对应：

```python
logits = res.logits[:, :-1, :]
per_token_logps = F.log_softmax(logits, dim=-1).gather(
    2, input_ids[:, 1:].unsqueeze(-1)
).squeeze(-1)
```

`gather` 不是保留 6400 个词的概率，而是在每个位置取“实际生成 token”的 log probability：

```text
[N,L-1,V] gather actual token ids -> [N,L-1]
```

组内统计：

```python
grouped_rewards = rewards.view(-1, G)  # [B,G]
mean_r = grouped_rewards.mean(dim=1)
std_r = grouped_rewards.std(dim=1, unbiased=False)
advantages = (rewards - repeated_mean) / (repeated_std + 1e-4)
```

最后先按每条轨迹的有效 action token 平均，再对轨迹平均：

```python
token_counts = completion_mask.sum(dim=1)        # [N]
sequence_loss = (per_token_loss * completion_mask).sum(1) / token_counts
policy_loss = sequence_loss[valid_rows].mean()   # scalar
```

这样可以避免长轨迹仅因为 token 多就在 batch loss 中占据不成比例的权重。

### 11.8 CISPO 与标准 PPO/GRPO clip 的代码差异

本项目 Agent RL 支持两种分支。标准 GRPO 风格：

{% raw %}
$$
\ell=-\left[\min(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A)-\beta KL\right]
$$
{% endraw %}

CISPO 分支对 ratio 做上界截断并 `detach`：

```python
clamped_ratio = torch.clamp(ratio, max=epsilon_high).detach()
per_token_loss = -(
    clamped_ratio * advantage * per_token_logps
    - beta * per_token_kl
)
```

`detach()` 表示 clamped ratio 在反向时被当作常数权重；梯度主要通过 `per_token_logps` 传播。不能把文档中的统一简化式当成每个实现完全相同，最终必须以训练脚本代码为准。

### 11.9 延迟奖励与信用分配限制

若整条轨迹只得到一个最终 reward，那么 tool-call 和 final-answer token 共享同一个轨迹 advantage。这是粗粒度信用分配：

- 最终答案正确，早期调用大概率被共同奖励。
- 最终答案错误，可能连正确的工具选择也一起受罚。

可改进方向包括步骤级 reward、过程验证器、工具调用局部 reward，但会增加奖励设计复杂度和被钻空子的机会。v0.1 应先保证轨迹级 reward 与严格测试一致。

---

## 12. 五种训练方法统一对照

![SFT、DPO、PPO、GRPO 与 Agent RL 的目标对照](/images/lifeos-agent-zero-to-mastery/five_stage_training_overview.svg)

![五种训练方法的 Loss 计算差异](/images/lifeos-agent-zero-to-mastery/loss_computation_comparison.svg)

| 方法 | 典型数据 | 在线生成 | 监督信号 | 是否有 Critic | 核心 loss |
| --- | --- | --- | --- | --- | --- |
| SFT | prompt + 标准回答 | 否 | 标准 token | 否 | Cross Entropy |
| DPO | prompt + chosen/rejected | 否 | 人类/规则偏好 | 否 | Pairwise preference |
| PPO | prompt | 是 | Reward | 是 | Clipped policy + value |
| GRPO | prompt，每题多条 rollout | 是 | 组内相对 reward | 否 | Group-relative policy |
| Agent RL | prompt + tools + 环境 | 是，多轮 | 轨迹 reward | 取决于算法 | 多轮 action-token policy loss |

共同点：最终都通过一个标量 loss 反向传播，改变模型参数，从而改变 token 概率。

不同点：监督信号从哪里来，以及哪些 token 被纳入优化。

---

## 13. 独立训练必须掌握的工程闭环

### 13.1 训练前必须回答 12 个问题

1. 要改善的具体能力是什么？
2. 用 SFT、DPO 还是 RL，为什么？
3. 起始 checkpoint 是哪个？
4. 数据 schema 是什么？
5. train/validation/test 是否隔离？
6. 样本数与 token 长度分布是什么？
7. `max_seq_len` 会截断什么？
8. micro batch 与 gradient accumulation 是多少？
9. learning rate 和 epoch 为什么这样选？
10. 输出路径是否会覆盖 production？
11. 记录哪些指标？
12. 什么条件停止、回滚或拒绝发布？

### 13.2 Effective batch size

若单卡训练：

{% raw %}
$$
B_{effective}=B_{micro}\times N_{accumulation}
$$
{% endraw %}

多卡时：

{% raw %}
$$
B_{effective}
=B_{micro}\times N_{accumulation}\times N_{gpu}
$$
{% endraw %}

梯度累积降低单步显存，但不会让总计算免费消失。

### 13.3 不能只看 loss

| 阶段 | 训练指标 | 必须搭配的任务指标 |
| --- | --- | --- |
| SFT | CE loss | 格式正确率、工具成功率、普通聊天保持率 |
| DPO | DPO loss、margin | chosen 胜率、能力回归 |
| PPO/GRPO | reward、KL、clip fraction | 独立测试集正确率、输出多样性 |
| Agent RL | reward、KL、长度 | tool-call、参数、执行、grounding、no-tool |

### 13.4 本项目发布闸门

至少分别验证：

1. Search 问题调用 `search_fake_obsidian`。
2. Task 问题调用 `list_today_tasks`。
3. Math 问题调用 `calculate_math` 并得到 19.43。
4. 普通聊天不传 tools、不误调用。
5. 最终回答引用真实工具结果，而不是自己编造。
6. 对照旧 checkpoint 没有明显回归。

当前 Agent RL v2 虽然严格调用达到 `3/3`，但原始最终 grounding 曾为 `2/3`，因此没有自动替换生产权重。这是正确的工程决策：训练完成不等于可以部署。

### 13.5 3090 Ti 独立训练的安全步骤

训练不是从复制一条命令开始，而是按以下顺序：

```text
1. 冻结目标：只改善一个可测能力
2. 校验 JSONL schema 和样本分布
3. 固定 baseline checkpoint 的测试结果
4. 使用新输出名，禁止覆盖 production
5. 先跑 1～5 step smoke test
6. 检查显存、shape、loss/reward、生成文本
7. 再启动完整训练
8. 保存日志、参数、git commit、数据 hash
9. 用固定 test set 对比 baseline/new
10. 达到发布闸门后手工切换 production symlink
```

显存不足时建议调整顺序：

1. 降低 `batch_size`。
2. 增加 `accumulation_steps` 保持 effective batch。
3. 降低 `max_seq_len` 或 RL 的 `max_total_len/max_gen_len`，但先检查截断风险。
4. RL 降低 `num_generations`，但组内统计质量也会下降。
5. 开启混合精度、gradient checkpointing 或更高效 rollout engine。
6. 最后才考虑缩小模型。

### 13.6 日志故障诊断树

#### Loss 变成 NaN

依次检查：

```text
数据是否有空 action mask
-> learning rate 是否过高
-> FP16 是否溢出
-> reward/advantage 是否异常大
-> denominator 是否为 0
-> gradient norm 是否爆炸
```

#### Reward 上升、工具成功率下降

依次检查：

```text
打印真实 rollout
-> 统计合法 XML/JSON 比例
-> 统计候选外工具名
-> 分解 reward 各项
-> 检查无调用是否仍能拿正奖励
-> 检查长度或 RM 是否压过工具奖励
```

#### Loss 降得很快、测试不提升

可能是：

- 训练集重复过高，模型背句式。
- 测试问法与训练问法差异太大。
- labels 大多是低价值固定文本。
- 训练数据泄漏进测试集。
- 只改善了格式，没有改善 grounding。

#### 模型会调用工具，但最终答案不用结果

要分别测试：

1. Tool result 是否真的回填。
2. 第二轮 prompt 是否包含 `role=tool`。
3. tool observation 是否被长度截断。
4. SFT 是否包含足够的“工具后二轮回答”。
5. Reward 是否单独验证 final grounding。

### 13.7 Checkpoint、Resume 与 Production 的区别

- **Weight file**：模型参数，适合推理。
- **Training checkpoint**：通常还包含 optimizer、scheduler、scaler、epoch、step，用于准确续训。
- **Production alias/symlink**：部署选择，不是新的模型文件。

只加载 weight 再训练，不等同于严格 resume，因为 optimizer 动量和 scheduler 进度可能丢失。面试中说“从某权重继续微调”与“断点续训”必须区分。

---

## 14. 高频面试问题与合格回答骨架

### 14.1 “Tool Calling 是模型执行工具吗？”

不是。模型生成结构化 action 文本，外部 parser 将其解析为对象，executor 校验后执行 Python 工具，再以 `role=tool` 把 observation 回填。模型只负责 token generation。

### 14.2 “工具选择是谁做的？”

两层选择。Router 根据用户输入筛候选 schema，控制哪些工具进入 prompt；LLM 在候选集合内判断是否调用、调用哪个以及参数是什么。Executor 最后再次做安全校验。

### 14.3 “Tool-call token 是否计算 loss？”

SFT 中，如果 tool call 位于 assistant label 区域，它参与交叉熵；Agent RL 中，它是策略生成的 action token，位于 response/action mask 内，参与策略 loss。`role=tool` 是环境 observation，通常不参与策略 loss。

### 14.4 “GRPO 与 PPO 最大区别是什么？”

PPO 用 critic/value function 估计 advantage；GRPO 对同一 prompt 采样一组回答，用组内 reward 标准化得到相对 advantage，通常不需要单独 critic。二者仍都要用生成 token 的 log probability 更新策略，并约束策略变化。

### 14.5 “Reward 上升为什么模型可能更差？”

模型可能利用奖励漏洞，例如输出长文本获得长度奖励，却没有合法工具调用。Reward 只是代理目标。必须同时查看严格格式、工具执行、最终 grounding 和 no-tool 回归测试。

### 14.6 “Agent RL 能从零学会写 Python 函数吗？”

通常不能把它理解为自动创造并注册可执行函数。模型可以探索生成字符串，但环境只能执行预先注册、可验证的工具。工具协议和基本格式通常先通过预训练/SFT 建立；Agent RL 更适合优化何时调用、选择哪个、参数如何填写以及多轮策略。

---

## 15. 最终闭卷复述模板

如果能不看文档完成下面这段话，并回答每一步的 shape，就达到了本项目的核心掌握标准：

> 用户问题先进入 Python router，router 只筛选候选工具 schema。`apply_chat_template` 把 system、user、候选 tools 和 assistant generation prompt 渲染成字符串；Tokenizer 把字符串变成 `[B,T]` 的 token ids。Transformer 经 embedding、8 层 attention/MLP 后输出 `[B,T,V]` logits，并自回归生成 `<tool_call>` action token。Parser 将 XML 包裹的 JSON 字符串变成 dict，Executor 校验工具名与参数并执行 Python，结果作为 `role=tool` observation 追加到 messages。第二轮重新 template 和 tokenize，模型读取 observation 生成最终回答。SFT 用 assistant token 的交叉熵教会格式；DPO用 chosen/rejected 的相对概率教偏好；PPO 用 critic 和 GAE；GRPO 用同组 reward 标准化 advantage；Agent RL 把多轮工具环境放进 rollout，对 tool-call 与 final-answer action token 计算策略 loss，对 tool observation 做 mask。最终 scalar loss 反向传播更新 current policy，训练后必须用独立测试集验证工具、参数、grounding 和 no-tool 能力，而不能只看 loss 或 reward。

---

## 16. 主教材与项目文件映射

| 学习主题 | 对应文件 |
| --- | --- |
| Agent 外部循环 | `lifeos_agent/main.py` |
| 候选工具路由 | `lifeos_agent/router.py` |
| Schema、参数校验与执行 | `lifeos_agent/tools.py` |
| SFT/DPO/PPO/GRPO/Agent RL 深入数据示例 | `docs/TRAINING_METHODS_COMPLETE_GUIDE.md` |
| Agent RL 361-token 完整 trace | `docs/AGENT_RL_COMPLETE_GUIDE.md` |
| 更完整数学证明 | `MATHEMATICAL_DERIVATIONS.md` |
| 真实远程训练指标和故障复盘 | `docs/REMOTE_AGENT_RL_TRAINING_REPORT_2026-07-15.md` |
| 7 天验收任务 | `docs/7_DAY_MASTERY_PLAN.md` |
| 固定评测集 | `eval/lifeos_eval.jsonl` |

## 17. 掌握检查题

不看答案完成以下题目：

1. 写出 `[B,T] -> [B,T,H] -> [B,T,V]` 的完整过程。
2. 手算 logits `[2,1,0]` 且 label 为第二类时的交叉熵。
3. 解释 causal mask 与 SFT label mask 的区别。
4. 为什么 schema 不进入 prompt，模型就不知道当前注册工具？
5. 为什么 router 不能替代 LLM 的参数生成？
6. 为什么 tool observation 进入 attention，却不进入 policy loss？
7. 从链式法则解释序列 log probability 为什么是 token log probability 之和。
8. 写出 DPO loss 并解释 reference model 和 $\beta$。
9. 从 $J(\theta)$ 推到 REINFORCE 的梯度形式。
10. 写出 TD error 和 GAE，并解释 $\gamma$、$\lambda$。
11. 写出 PPO ratio，说明 clip 对正负 advantage 的影响。
12. 手算 reward `[3,1,-1,-3]` 的 GRPO advantage。
13. 一条 Agent 轨迹有哪些 action token 和 observation token？
14. Reward 为什么不能直接 `backward()`？
15. 为什么训练 reward 上升却可能严格工具调用率下降？
16. 3090 Ti 出现 OOM 时，应按什么顺序调整参数？
17. 为什么不应该让新 checkpoint 自动覆盖 production？
18. 用三分钟完整讲解本项目端到端数据流。

建议判定：

- 15～18 题正确且能追问：核心熟练。
- 11～14 题正确：理解但不稳定，需要闭卷重做。
- 10 题及以下：仍处于阅读记忆阶段，应回到单一样本和手算。

---

## 18. 一次完整训练 Step 的统一伪代码

### 18.1 SFT

```python
for input_ids, labels in loader:                   # [B,T], [B,T]
    output = model(input_ids, labels=labels)
    loss = output.loss / accumulation_steps        # scalar
    loss.backward()
    if ready_to_update:
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

### 18.2 DPO

```python
chosen_logp  = sequence_logp(policy, chosen, chosen_mask)      # [B]
rejected_logp = sequence_logp(policy, rejected, rejected_mask) # [B]
with no_grad:
    ref_chosen_logp = sequence_logp(reference, chosen, chosen_mask)
    ref_rejected_logp = sequence_logp(reference, rejected, rejected_mask)

policy_margin = chosen_logp - rejected_logp
ref_margin = ref_chosen_logp - ref_rejected_logp
loss = -logsigmoid(beta * (policy_margin - ref_margin)).mean()
loss.backward()
optimizer.step()
```

### 18.3 PPO

```python
responses = old_policy.generate(prompts)
rewards = reward_model(prompts, responses)
values = critic(prompts, responses)
advantages, returns = gae(rewards, values)

ratio = exp(current_logp - old_logp)
actor_loss = -minimum(ratio * advantages, clipped_ratio * advantages)
critic_loss = mse(current_values, returns)
total_loss = actor_loss + value_coef * critic_loss + beta * kl
total_loss.backward()
optimizer.step()
```

### 18.4 GRPO / Agent RL

```python
trajectories = []
for prompt in prompts:
    for _ in range(G):
        trajectory = rollout_with_optional_tools(prompt, max_turns=3)
        trajectories.append(trajectory)

rewards = verify_trajectories(trajectories)          # [B*G]
advantages = normalize_within_each_prompt(rewards)   # [B*G]
input_ids, action_mask = pack_and_pad(trajectories)  # [B*G,L]

current_logp = token_logp(policy, input_ids)          # [B*G,L-1]
old_logp = stored_rollout_logp                         # [B*G,L-1]
ref_logp = token_logp(reference, input_ids)            # [B*G,L-1]

token_loss = policy_objective(current_logp, old_logp, ref_logp, advantages)
loss = masked_sequence_then_batch_mean(token_loss, action_mask)
loss.backward()
optimizer.step()
```

这四段伪代码必须能够闭卷写出。它们展示了所有训练方法的共同骨架：

```text
构造监督信号
-> 计算 token probability
-> 汇总为 scalar loss
-> backward
-> optimizer step
```

区别只在“监督信号怎么来”和“哪些 token 参与”。

---

## 19. 一条 Agent RL 轨迹从文本到 Loss 的总算例

![一条 Agent RL 工具轨迹的端到端 Trace](/images/lifeos-agent-zero-to-mastery/agent_rl_end_to_end_trace.svg)

这一节把前面所有概念合并，不引入新名词。

### 19.1 初始数据

用户：

```text
17.66 涨停价是多少？
```

Router 返回：

```python
["calculate_math"]
```

Schema lookup 返回一个工具定义。Chat template 生成第一轮 prompt，假设 tokenizer 后：

```text
prompt_ids 长度 P = 120
input_ids.shape = [1,120]
```

### 19.2 第一轮动作

模型自回归生成：

```xml
<tool_call>{"name":"calculate_math","arguments":{"expression":"round(17.66*1.1,2)"}}</tool_call>
```

假设共 $K_1=35$ 个 token：

```text
generated_ids.shape = [1,155]
response_ids.shape  = [1,35]
```

这 35 个 token 全部是模型动作；在 Agent RL 打包后，它们的 action mask 为 1。

### 19.3 环境观察

Parser：

```python
{
    "name": "calculate_math",
    "arguments": {"expression": "round(17.66*1.1,2)"},
}
```

Executor：

```python
{"result": 19.43}
```

假设 tool message 渲染后是 12 个 token。这 12 个 token 会进入第二轮上下文，但 action mask 为 0。

### 19.4 第二轮动作

第二轮 prompt 概念长度：

{% raw %}
$$
P_2=P+K_1+K_{tool}+K_{role}
$$
{% endraw %}

其中 $K_{role}$ 包含 assistant/tool 的协议标记。模型生成：

```text
17.66 元的涨停价约为 19.43 元。
```

假设是 $K_2=14$ 个 action token。

不考虑协议边界 token 的简化 action 数：

{% raw %}
$$
K_{action}=K_1+K_2=35+14=49
$$
{% endraw %}

### 19.5 同题四条轨迹

训练不会只生成这一条，假设四条 rollout：

| 轨迹 | 行为 | Reward |
| --- | --- | ---: |
| 1 | 正确调用、参数正确、回答包含 19.43 | 3 |
| 2 | 工具正确，但最终回答表达不完整 | 1 |
| 3 | 没调用工具，直接猜答案 | -1 |
| 4 | 非法标签或错误工具名 | -3 |

前面已经算得：

{% raw %}
$$
\mu=0,\qquad\sigma=\sqrt5\approx2.236
$$
{% endraw %}

{% raw %}
$$
A_1\approx1.342,\quad
A_2\approx0.447,\quad
A_3\approx-0.447,\quad
A_4\approx-1.342
$$
{% endraw %}

### 19.6 轨迹 1 某个工具名 token 的概率

选择 `calculate_math` 中的一个实际 token，假设：

```text
old_logp     = -0.50
current_logp = -0.40
ref_logp     = -0.45
advantage    = +1.342
beta         = 0.10
```

#### Ratio

{% raw %}
$$
r=\exp(-0.40-(-0.50))=e^{0.10}\approx1.1052
$$
{% endraw %}

说明 current policy 给这个 token 的概率相对 rollout 时提高了约 10.5%。

#### KL 近似项

代码使用：

{% raw %}
$$
d=\log\pi_{ref}-\log\pi_\theta=-0.45-(-0.40)=-0.05
$$
{% endraw %}

{% raw %}
$$
KL_{token}=e^d-d-1
$$
{% endraw %}

{% raw %}
$$
=e^{-0.05}+0.05-1
\approx0.95123+0.05-1
\approx0.00123
$$
{% endraw %}

它是非负且在两策略接近时很小。

#### CISPO token loss

假设 ratio 未超过 `epsilon_high`：

{% raw %}
$$
c=\operatorname{clamp}(r,\text{max}=\epsilon_{high})\approx1.1052
$$
{% endraw %}

{% raw %}
$$
\ell
=-\left[cA\log\pi_\theta-\beta KL_{token}\right]
$$
{% endraw %}

代入：

{% raw %}
$$
\ell
=-\left[1.1052\times1.342\times(-0.40)-0.1\times0.00123\right]
$$
{% endraw %}

{% raw %}
$$
\ell\approx0.5935
$$
{% endraw %}

不要根据“这个 token loss 是正数”判断更新方向。看导数：由于 $c$ 和 $A$ 在该分支被当成权重，$A\gt 0$ 时最小化 loss 会提高这个实际 token 的 log probability。

如果轨迹 4 的 $A\lt 0$，梯度方向相反，会降低其非法 action token 再次出现的概率。

### 19.7 从 49 个 token 到一个轨迹 loss

假设轨迹 1 有 49 个有效 action token：

{% raw %}
$$
\mathcal L_1
=\frac{\ell_1+\ell_2+\cdots+\ell_{49}}{49}
$$
{% endraw %}

Tool observation 的 12 个 token 因 mask 为 0，不进入分子，也不进入 token count。

四条轨迹分别求平均，再求 batch 平均：

{% raw %}
$$
\mathcal L
=\frac{\mathcal L_1+\mathcal L_2+\mathcal L_3+\mathcal L_4}{4}
$$
{% endraw %}

得到 scalar，执行：

```python
loss.backward()
optimizer.step()
```

参数更新后，下次遇到相似 prompt，正确工具格式和参数 token 的概率倾向上升，非法轨迹 token 的概率倾向下降。

### 19.8 这条算例的证据边界

- `35/12/14` token 是便于手算的示意长度；真实长度必须由具体 tokenizer 打印。
- Reward `3/1/-1/-3` 是教学轨迹组；真实值由训练脚本 reward 代码计算。
- Shape、mask、group normalization、ratio 和 KL 形式对应当前 MiniMind Agent RL 实现。
- 实际训练要以日志、debug rollout 和独立测试结果为准。

---

## 20. 术语表：必须能用自己的话解释

| 术语 | 精确定义 |
| --- | --- |
| Token | Tokenizer 词表中的离散单元，用整数 id 表示 |
| Vocabulary | 全部 token 的集合，大小记为 $V$ |
| Embedding | 把 token id 查表映射到 $H$ 维连续向量 |
| Hidden state | Transformer 某层对每个 token 的内部表示 |
| Logit | Softmax 前的未归一化词表分数 |
| Probability | Softmax 后归一化到和为 1 的数值 |
| Log probability | 概率的自然对数，序列中可相加 |
| Label | 期望模型在对应位置预测的 token id |
| Mask | 控制哪些位置可见或参与 loss 的 0/1 标记 |
| Causal mask | 禁止 token 读取未来位置 |
| Padding mask | 忽略为对齐长度而补的 pad token |
| Loss mask | 只让指定 token 参与训练目标 |
| Cross Entropy | 标准 token 的负 log probability |
| Gradient | Loss 对参数的偏导数组合 |
| Optimizer | 根据梯度更新参数的算法，如 AdamW |
| Checkpoint | 训练状态快照，可能含模型、优化器和 scheduler |
| SFT | 用标准 assistant token 做监督微调 |
| DPO | 用 chosen/rejected 相对偏好优化策略 |
| Policy | 给定状态时对下一动作 token 的概率分布 |
| State | Agent 当前可见的消息历史和环境信息 |
| Action | 模型实际生成的 token 序列 |
| Observation | 外部环境返回并进入下一轮上下文的信息 |
| Trajectory | 多轮 state、action、observation 的完整过程 |
| Reward | 环境对轨迹质量的标量评分 |
| Value | Critic 对状态未来回报的预测 |
| Advantage | 某动作相对基准表现好多少 |
| Old policy | 产生本批 rollout 的策略快照 |
| Reference policy | 冻结的能力锚点，用于 KL 约束 |
| Importance ratio | Current 与 old 对同一动作的概率比 |
| KL divergence | 衡量 current 与 reference 分布偏离程度 |
| PPO | 使用 critic、GAE 和 clipped objective 的策略优化 |
| GRPO | 用同题组内 reward 标准化构造 advantage 的策略优化 |
| CISPO | 当前实现使用的一种 importance-sampling policy objective |
| Router | 在模型前筛选候选工具 schema 的工程组件 |
| Parser | 把模型工具调用文本解析为结构化对象 |
| Executor | 校验并执行已注册工具的外部程序 |
| Grounding | 最终回答忠实使用工具结果或证据 |
| Reward hacking | 模型利用评分漏洞得高分，却没有完成真实目标 |

真正掌握的标准不是能认出这些词，而是能把任意一个词放回完整数据流，并说出它的输入、输出、shape、是否可训练以及失败时如何观察。
