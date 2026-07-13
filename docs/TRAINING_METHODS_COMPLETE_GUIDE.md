# LifeOS-Agent 训练全解：SFT、DPO、PPO、GRPO 与 Agent RL

> 适用项目：MiniMind 63.91M（hidden size 768、8 层、8 个 Query heads、4 个 KV heads、词表 6400）  
> 适用硬件：单卡 RTX 3090 Ti 24GB  
> 目标：从一条 JSON 数据开始，追踪它如何变成张量、经过网络、形成 loss，直到参数更新。  
> 术语说明：本文将用户提到的“PRO”按 **PPO（Proximal Policy Optimization）** 解释。

![五类训练方法总览](assets/five_stage_training_overview.svg)

## 0. 先用一句话分清五种训练

| 方法 | 数据告诉模型什么 | 模型在训练时要做什么 | 最终优化目标 |
|---|---|---|---|
| SFT | “这个问题的标准答案是它” | 预测标准答案的下一个 token | 提高标准答案概率 |
| DPO | “答案 A 比答案 B 好” | 比较 chosen/rejected 的序列概率 | 更偏好 chosen，同时不偏离参考模型太多 |
| PPO | “你先自己回答，奖励模型再打分” | 在线生成、估值、算 advantage、更新 actor/critic | 提高高回报动作概率，限制策略突变 |
| GRPO | “同一道题生成多份答案，组内比高低” | 一次生成 G 个回答，用组内均值作 baseline | 提高组内相对高分回答概率 |
| Agent RL | “你可以调用工具，最终结果必须正确” | 多轮生成 tool call、执行工具、读 observation、继续回答 | 学会正确选工具、填参数、利用结果并完成回答 |

通俗类比：

- SFT 是看老师的标准答案临摹。
- DPO 是老师给两篇作文，让你学习哪篇更好。
- PPO 是自己写作文，老师打分，同时再训练一个“自我估分器”。
- GRPO 是一次写四篇，比较同一组中谁最好，不再单独训练估分器。
- Agent RL 是参加开卷实操考试：不仅要回答，还要决定何时查资料、按什么格式调用工具、如何使用查询结果。

## 1. 所有方法共享的 Transformer 主干

### 1.1 符号和本项目固定维度

| 符号 | 含义 | 本项目值或范围 |
|---|---|---|
| $B$ | batch 中 prompt 数 | SFT 计划值 4；RL 实际值 1 |
| $G$ | 每个 prompt 的生成数 | PPO 1；GRPO/Agent RL 4 |
| $T$ | 当前完整序列长度 | SFT/DPO 最多 768；Agent 轨迹最多 1600 |
| $P$ | prompt token 数 | 最多 768 |
| $R$ | response token 数 | 最多 256 |
| $V$ | tokenizer 词表大小 | 6400 |
| $H$ | hidden size | 768 |
| $L$ | Transformer 层数 | 8 |
| $N_q$ | Query attention heads | 8 |
| $N_{kv}$ | Key/Value heads | 4 |
| $D_h$ | 每个 head 的维度 | $768/8=96$ |

一条文本不会直接进入神经网络。它先变成 token id：

```text
"计算 2+3" -> [1, 1042, 87, 19, 14, 20, 2]
```

数字只是示意，真实 ID 由 tokenizer 决定。批处理后：

```text
input_ids       [B, T]      int64
attention_mask  [B, T]      0 表示 padding，1 表示有效 token
labels          [B, T]      SFT 中 -100 表示不计算 loss
```

### 1.2 Embedding

词嵌入矩阵为：

$$
E\in\mathbb{R}^{V\times H}=\mathbb{R}^{6400\times768}
$$

查表后：

$$
X=E[input\_ids]\in\mathbb{R}^{B\times T\times768}
$$

例如 SFT 使用 $B=4,T=768$：

```text
input_ids       [4, 768]
hidden_states   [4, 768, 768]
```

### 1.3 一层 Grouped-Query Attention 的真实维度

MiniMind 使用 8 个 Q heads、4 个 KV heads：

```text
输入 X                 [B, T, 768]
Q projection           [B, T, 8*96] -> [B, T, 8, 96]
K projection           [B, T, 4*96] -> [B, T, 4, 96]
V projection           [B, T, 4*96] -> [B, T, 4, 96]
K/V repeat 两次        [B, T, 8, 96]
transpose 后 Q/K/V     [B, 8, T, 96]
attention scores       [B, 8, T, T]
attention output       [B, 8, T, 96]
拼回                   [B, T, 768]
```

注意力分数：

$$
S=\frac{QK^\top}{\sqrt{D_h}}+M_{causal}+M_{padding}
$$

$$
A=\operatorname{softmax}(S),\qquad O=AV
$$

除以 $\sqrt{96}$ 是为了避免点积方差随 head dimension 增大，导致 softmax 过早饱和。Causal mask 保证第 $t$ 个 token 只能看到自己和之前的 token。

### 1.4 MLP、残差和最终 logits

每层都执行：

$$
X'=X+\operatorname{Attention}(\operatorname{RMSNorm}(X))
$$

$$
X''=X'+\operatorname{MLP}(\operatorname{RMSNorm}(X'))
$$

8 层之后仍是 `[B,T,768]`。语言模型头：

$$
W_{lm}\in\mathbb{R}^{768\times6400}
$$

$$
Z=HW_{lm}\in\mathbb{R}^{B\times T\times6400}
$$

`logits[b,t,k]` 表示：看到第 `t` 个位置以前的上下文后，下一个 token 为词表第 `k` 项的未归一化分数。

![共享网络张量维度](assets/network_tensor_dimensions.svg)

## 2. SFT：从标准答案学习

### 2.1 典型数据

真实数据是 `conversations`：

```json
{
  "conversations": [
    {"role": "user", "content": "你模型的训练数据如何保障准确性？"},
    {"role": "assistant", "content": "通过多轮验证和质量控制流程保障准确性。"}
  ]
}
```

LifeOS Tool Calling 样本更接近：

```json
{
  "conversations": [
    {"role": "system", "content": "你是 LifeOS-Agent", "tools": "[...]"},
    {"role": "user", "content": "17.66 涨停价是多少？"},
    {"role": "assistant", "content": "", "tool_calls": "[{...}]"},
    {"role": "tool", "content": "{\"result\":19.43}"},
    {"role": "assistant", "content": "按 10% 计算，涨停价约为 19.43 元。"}
  ]
}
```

`apply_chat_template` 会把 tools schema、角色标签、`<tool_call>`、tool response 和 assistant 内容统一渲染成一段文本。

### 2.2 数据如何变成 labels

MiniMind `SFTDataset`：

1. 调用 `apply_chat_template(..., tools=tools)`。
2. tokenizer 得到最多 768 个 token。
3. 不足 768 的部分用 pad 填满。
4. `labels` 初始全部为 `-100`。
5. 只把 `assistant` 段对应 token 复制到 labels。

示意：

```text
token:   <system> tools <user> 17.66... <assistant> <tool_call> ... </tool_call>
label:      -100   -100   -100    -100       -100      id1     id2       id3
loss?:        否     否     否      否          否       是      是        是
```

用户和工具内容虽然不直接算 loss，却通过 attention 影响 assistant token 的 hidden state，因此仍然决定模型“在什么条件下”输出答案。

### 2.3 Shift 后的维度

模型内部执行：

```python
x = logits[..., :-1, :]   # [B, T-1, V]
y = labels[..., 1:]       # [B, T-1]
```

本项目计划 SFT `B=4,T=768,V=6400`：

```text
input_ids       [4, 768]
labels          [4, 768]
hidden_states   [4, 768, 768]
logits          [4, 768, 6400]
shift_logits    [4, 767, 6400]
shift_labels    [4, 767]
loss            []  scalar
```

### 2.4 Cross Entropy 逐步计算

位置 $t$ 的概率：

$$
p_{t,k}=\frac{e^{z_{t,k}}}{\sum_{j=1}^{V}e^{z_{t,j}}}
$$

若真实 token 为 $y_t$：

$$
\ell_t=-\log p_{t,y_t}
$$

只平均有效 assistant token：

$$
L_{SFT}=\frac{\sum_{b,t}m_{b,t}(-\log p_\theta(y_{b,t}|x_{b,<t}))}{\sum_{b,t}m_{b,t}}
$$

其中 $m=1$ 表示 assistant token，`labels=-100` 的位置等价于 $m=0$。

对 logit 的梯度：

$$
\frac{\partial\ell}{\partial z_k}=p_k-\mathbb{1}[k=y]
$$

所以梯度下降会提高正确 token 的 logit，降低错误 token 的 logit。

### 2.5 SFT 需要多少数据

| 目标 | 建议高质量样本量 | 原因 |
|---|---:|---|
| 改名字、固定口吻 | 20-200 | 输出模式集中，少量重复就能改变 greedy decoding |
| 新增 3-10 个工具格式 | 500-5,000 | 需要覆盖工具选择、参数、错误、多轮和无工具负例 |
| 稳定领域助手 | 10,000-100,000 | 要覆盖不同问法、长度、主题和边界情况 |
| 通用聊天能力 | 100,000-1,000,000+ | 语言现象和知识分布极长尾 |

本项目通用 SFT 有 905,718 条，其中约 9.37% 含工具；LifeOS 增量 seed 只有 26 条，其中 18 条含工具。26 条能让名字立刻生效，因为“你是谁”对应的目标 token 非常固定；但它不足以保证开放场景下稳定工具调用。

## 3. DPO：从 chosen/rejected 学偏好

### 3.1 典型数据

```json
{
  "chosen": [
    {"role":"user", "content":"如何扩展女性疗愈营业务？"},
    {"role":"assistant", "content":"先扩充团队，再增加主题和合作场地……"}
  ],
  "rejected": [
    {"role":"user", "content":"如何扩展女性疗愈营业务？"},
    {"role":"assistant", "content":"直接做加盟、贷款和国际扩张……"}
  ]
}
```

DPO 不要求一个绝对完美答案，只要求同一 prompt 下 `chosen` 优于 `rejected`。

### 3.2 数据张量

`DPODataset` 分别渲染并 tokenize，两条都 pad 到 $T=768$，再提前 shift：

```text
x_chosen       [B, 767]
y_chosen       [B, 767]
mask_chosen    [B, 767]
x_rejected     [B, 767]
y_rejected     [B, 767]
mask_rejected  [B, 767]
```

训练时在 batch 维拼接：

```text
x       [2B, 767]
y       [2B, 767]
mask    [2B, 767]
logits  [2B, 767, 6400]
logp    [2B, 767]
```

这里同时有两个模型：可训练 policy $\pi_\theta$ 和冻结 reference $\pi_{ref}$。两者都前向，但只有 policy 反向传播。

### 3.3 从 token log probability 到序列分数

先从 `[2B,767,6400]` gather 正确 token：

$$
\log p_{b,t}=\log\operatorname{softmax}(z_{b,t})_{y_{b,t}}
$$

再仅对 assistant mask 求和：

$$
\log\pi(y|x)=\sum_t m_t\log\pi(y_t|x,y_{<t})
$$

结果从 `[2B,767]` 变为 `[2B]`，然后前半拆为 chosen、后半拆为 rejected，各为 `[B]`。

### 3.4 DPO loss

策略偏好差：

$$
\Delta_\pi=\log\pi_\theta(y^+|x)-\log\pi_\theta(y^-|x)
$$

参考模型偏好差：

$$
\Delta_{ref}=\log\pi_{ref}(y^+|x)-\log\pi_{ref}(y^-|x)
$$

最终：

$$
L_{DPO}=-\frac1B\sum_i\log\sigma\left(\beta(\Delta_{\pi,i}-\Delta_{ref,i})\right)
$$

直觉：不是要求 chosen 概率绝对大，而是要求 policy 相对 reference 更偏向 chosen。$\beta$ 控制偏好更新强度。本项目使用 17,166 对、1 epoch、初始学习率约 $4\times10^{-8}$，最终记录 loss 0.3030。

### 3.5 DPO 数据量建议

| 目标 | 建议偏好对数 |
|---|---:|
| 调整回答长度/语气 | 500-3,000 |
| 纠正工具格式和拒答边界 | 2,000-20,000 |
| 稳定领域偏好 | 10,000-100,000 |

DPO 的关键不是数量，而是 chosen/rejected 差异必须可解释。若两者质量相近，信号弱；若 prompt 不同，偏好比较失去意义；若 rejected 明显包含格式噪声，模型可能只学会表面特征。

## 4. PPO：Actor、Critic 与 Reward Model 协作

### 4.1 PPO 数据只有 prompt

PPO 使用 `rlaif.jsonl`。文件中虽然保留最后一个空 assistant，但 `RLAIFDataset` 实际只渲染 `conversations[:-1]`：

```json
{
  "conversations": [
    {"role":"user", "content":"基于以上对话提出问题……"},
    {"role":"assistant", "content":"这些产品需要哪些前提？"},
    {"role":"user", "content":"请回答这个问题。"},
    {"role":"assistant", "content":""}
  ]
}
```

训练时没有标准答案；actor 在线生成答案，1.8B InternLM2 reward model 打分。

### 4.2 本项目真实维度

参数：`B=1, P<=768, R<=256, G=1`。

```text
prompt input_ids       [B, P]
completion_ids         [B, R]
gen_out                [B, P+R]
actor logits           [B, P+R, 6400]
old_resp_logp          [B, R]
new_resp_logp          [B, R]
ref_resp_logp          [B, R]
critic values          [B, P+R]
old_resp_values        [B, R]
token_rewards          [B, R]
advantages             [B, R]
returns                [B, R]
reward model score     [B]
```

### 4.3 Reward 如何放到 token 上

外部 reward 是完整回答的一个标量 $r$。代码把它加到最后一个有效 response token：

```text
token_rewards = [0, 0, 0, ..., r]
```

随后 Critic 为每个 response token 预测状态价值 $V(s_t)$。

### 4.4 GAE

TD 残差：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

广义优势估计：

$$
A_t=\delta_t+\gamma\lambda A_{t+1}
$$

return：

$$
R_t=A_t+V_{old}(s_t)
$$

优势经过有效 token 上的均值方差归一化，维度仍为 `[B,R]`。

### 4.5 Actor loss

新旧策略概率比：

$$
r_t(\theta)=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t))
$$

PPO clipped objective：

$$
L_{actor}=-\mathbb{E}_t\left[\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\right]
$$

本实现还加入相对 reference 的 KL penalty：

$$
L_{policy}=L_{actor}+c_{KL}D_{KL}(\pi_\theta\|\pi_{ref})
$$

### 4.6 Critic loss

Critic 预测 `[B,R]`，目标 return 也是 `[B,R]`：

$$
L_V=\frac12\mathbb{E}_t\left[\max((V_\phi-R_t)^2,(\operatorname{clip}(V_\phi,V_{old}-\epsilon_v,V_{old}+\epsilon_v)-R_t)^2)\right]
$$

总 loss：

$$
L_{PPO}=L_{policy}+c_vL_V+L_{aux}
$$

PPO 比其他方法占显存，因为同时驻留 actor、critic、reference 和 1.8B reward model。

### 4.7 PPO 数据量建议

| 场景 | 建议 prompt 数 |
|---|---:|
| 验证 reward/代码 | 500-2,000 |
| 单一能力优化 | 5,000-20,000 |
| 多领域稳定优化 | 20,000-100,000+ |

在线 RL 的一个 prompt 可能被重复更新多次，数据量不能照搬 SFT。奖励模型偏差会被策略主动利用，通常应先用小规模 rollout 验证 reward 与人工评价相关，再扩大。

## 5. GRPO：不训练 Critic 的组内相对优化

### 5.1 数据与 PPO 相同，计算方式不同

GRPO 同样用 19,502 条 RLAIF prompt，但每个 prompt 生成 $G=4$ 条答案：

```text
B=1 个 prompt
-> G=4 条 completion
-> 有效 rollout batch = B*G = 4
```

维度：

```text
prompt_ids             [B, P]
outputs                [B*G, P+R]
completion_ids         [B*G, R]
rewards                [B*G]
grouped_rewards        [B, G]
advantages             [B*G]
new/old/ref logps      [B*G, R]
completion_mask        [B*G, R]
per_token_loss         [B*G, R]
policy_loss            []
```

### 5.2 组内 advantage

对同一个 prompt 的四个 reward：

$$
R=[1.2,-0.4,0.8,0.0]
$$

$$
\mu=0.4,\qquad \sigma\approx0.632
$$

$$
A_i=\frac{R_i-\mu}{\sigma+10^{-4}}
$$

大约得到：

```text
[1.265, -1.265, 0.633, -0.633]
```

高于组均值的回答正 advantage，低于均值的回答负 advantage。它不需要 Critic，因此比 PPO 简单；代价是每题必须生成多条答案。

若四条答案 reward 完全相等，$A_i\approx0$，这一组几乎没有策略梯度。这不是 bug，而是组内没有可比较信息。

### 5.3 GRPO/CISPO loss

标准 GRPO 分支：

$$
L=-\frac1{BG}\sum_i\frac1{|R_i|}\sum_t m_{i,t}left[\min(r_{i,t}A_i,\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_i)-\beta KL_{i,t}\right]
$$

本项目命令未显式覆盖 `loss_type`，因此使用源码默认 `cispo`：

$$
\tilde r_{i,t}=\min(r_{i,t},\epsilon_{high})\quad\text{并 detach}
$$

$$
L_{CISPO}=-\mathbb{E}_{i,t}[\tilde r_{i,t}A_i\log\pi_\theta(a_{i,t}|s_{i,t})-\beta KL_{i,t}]
$$

### 5.4 GRPO 数据量建议

| 每 prompt 生成数 $G$ | 建议 prompt 数 | 实际候选轨迹数 |
|---:|---:|---:|
| 2 | 5,000-20,000 | 10,000-40,000 |
| 4 | 5,000-20,000 | 20,000-80,000 |
| 8 | 2,000-10,000 | 16,000-80,000 |

应按“生成轨迹数”而不是 JSON 行数评估成本。本项目 19,502 prompt、$G=4$，约生成 78,008 条候选回答。

## 6. Agent RL：把工具环境放进 rollout

### 6.1 两类 Agent 数据

`agent_rl.jsonl` 共 39,988 条，其中约 20,000 条带 tools 和 gt，另一些是普通对话 prompt，用于防止模型看到任何问题都调用工具。

典型工具样本（精简自真实结构）：

```json
{
  "conversations": [
    {
      "role":"system",
      "content":"",
      "tools":"[{\"type\":\"function\",\"function\":{\"name\":\"calculate_math\",...}}]"
    },
    {"role":"user", "content":"Compute 2045*6994 for me"},
    {"role":"assistant", "content":""}
  ],
  "gt":["14302730"]
}
```

`AgentRLDataset` 返回：

```text
messages = conversations[:-1]
tools    = system.tools 解析后的 list[dict]
gt       = ["14302730"]
```

注意：`gt` 不是拿来做交叉熵的标准答案，而是用于验证最终回答是否包含正确结果。

### 6.2 一条多轮轨迹

```text
Turn 1 prompt:
  system + tools schema + user

模型生成（mask=1）:
  <tool_call>{"name":"calculate_math","arguments":{"expression":"2045*6994"}}</tool_call>

环境执行并追加（mask=0）:
  role=tool, {"result":14302730}

Turn 2 模型生成（mask=1）:
  计算结果是 14302730。
```

为什么 observation mask 为 0？工具返回不是模型选择的动作，不应奖励模型“背诵环境输出”；但 observation 仍位于上下文中，后续答案可以 attention 到它。

### 6.3 本项目真实维度

参数：`B=1,G=4,P<=768,R_each<=256,max_turns=3,max_total_len=1600`。

因为四条轨迹长度可能不同，先动态 pad 到该组最大长度 $T_{batch}$：

```text
input_ids             [B*G, T_batch] = [4, T_batch], T_batch<=1600
full_mask             [4, T_batch]
full_response_masks   [4, T_batch]
old_per_token_logps   [4, T_batch-1]
logits                [4, T_batch, 6400]
per_token_logps       [4, T_batch-1]
ref_per_token_logps   [4, T_batch-1]
completion_mask       [4, T_batch-1]
rewards               [4]
advantages            [4]
per_token_loss        [4, T_batch-1]
policy_loss           []
```

Agent RL 的初始 prompt 截断到 768，但加入工具调用、observation 和后续回答后，整条训练轨迹最多保留 1600 token。超过时从左侧截断，这是一个风险：可能丢掉最初用户要求或 tools schema。

### 6.4 Agent reward 的具体组成

无工具调用时：

- 回答长度合理：`+0.5`，否则 `-0.5`。
- thinking 长度合理：`+1.0`，否则 `-0.5`。
- thinking 标签正确闭合：`+0.25`，否则扣分。
- 1.8B reward model 质量分。
- 重复文本 penalty。

有工具调用时：

- `<tool_call>` 标签不配对：每次扣 `0.5`。
- 工具名存在、arguments 通过对应 checker，并且调用数量正确：`+0.5`。
- 工具数量或参数不正确：按 gap 扣分。
- 最终回答包含 gt：最多 `+2.5`。
- 达到最大轮数仍未完成：`-0.5`。
- 重复回答扣分。
- 总 reward clip 到 `[-3,3]`。

### 6.5 Agent RL loss

reward `[B*G]` 先像 GRPO 一样变成组内 advantage `[B*G]`，再广播到每个可学习 response token：

$$
L_{Agent}=-\frac1{BG}\sum_i\frac{\sum_t m^{action}_{i,t}(\tilde r_{i,t}A_i\log\pi_\theta(a_{i,t}|s_{i,t})-\beta KL_{i,t})}{\sum_t m^{action}_{i,t}}
$$

`m_action=1` 的位置包括模型生成的 tool call 和最终回答；工具 observation 为 0。

### 6.6 为什么 Agent RL 特别慢

本项目：

$$
39988\text{ prompts}\times4\text{ generations}=159952\text{ trajectories}
$$

每条轨迹最多 3 轮，每轮最多生成 256 token，还要：

1. 反复运行自回归生成。
2. Python 解析 XML/JSON tool call。
3. 执行工具并重新渲染 chat template。
4. 运行 1.8B reward model 或规则 reward。
5. 再运行 policy 和 reference 前向。
6. policy 反向传播。

所以“一条 Agent RL 数据”远重于“一条 SFT 数据”。

### 6.7 Agent RL 数据量建议

| 能力范围 | 建议 prompt 数 | 关键覆盖 |
|---|---:|---|
| 单工具概念验证 | 500-2,000 | 正常参数、错误参数、无工具负例 |
| 3-10 个稳定工具 | 5,000-20,000 | 工具选择、并列意图、失败恢复、多轮 |
| 多领域个人 Agent | 20,000-100,000 | 长尾表达、权限、安全、组合工具、真实环境反馈 |

工具样本与无工具样本应保持合理比例。本项目约 50% 带工具，是为了同时训练“该调用”和“不该调用”。若 100% 都有工具，模型容易形成工具滥用。

## 7. 五种 loss 的同与不同

![五种损失函数对比](assets/loss_computation_comparison.svg)

### 7.1 相同点

1. 最终都通过同一个 policy Transformer 得到 `[*,T,6400]` logits。
2. 都需要从 logits 得到真实或生成 token 的 log probability。
3. 最终 loss 都是 scalar，调用 `backward()` 后更新 policy 参数。
4. prompt/pad/observation 等不应学习的位置都依赖 mask 排除。
5. 都需要控制学习率和梯度，否则会遗忘 SFT 基础能力。

### 7.2 核心不同点

| 对比项 | SFT | DPO | PPO | GRPO | Agent RL |
|---|---|---|---|---|---|
| 答案来源 | 数据集 | chosen/rejected | 在线生成 | 在线生成 G 条 | 多轮环境生成 G 条 |
| 学习信号 | token 标签 | 成对偏好 | 标量 reward + critic | 组内 reward | 工具规则 + gt + RM |
| advantage | 无 | 无 | 每 token GAE | 每序列组内标准化 | 每轨迹组内标准化 |
| reference model | 无 | 有 | 有 | 有 | 有 |
| critic | 无 | 无 | 有 | 无 | 无 |
| reward model | 无 | 无 | 有 | 有 | 无工具分支可用 |
| policy loss 粒度 | assistant token | chosen/rejected 序列 | response token | response token | action token |
| 主要成本 | 1 次 policy | 2 模型×2序列 | rollout+4模型 | G rollout+3模型 | 多轮 G rollout+工具环境 |

## 8. 一个可手算的微型例子

假设词表只有 4 个 token，某位置 logits：

$$
z=[2.0,1.0,0.0,-1.0]
$$

softmax 约为：

$$
p=[0.644,0.237,0.087,0.032]
$$

若正确 token 是第二项：

$$
L_{SFT}=-\log0.237\approx1.44
$$

若正确 token 是第一项，loss 只有：

$$
-\log0.644\approx0.44
$$

DPO 不直接比较这一个 token，而是把 chosen 所有 assistant token 的 log probability 相加，再减 rejected 的序列和。

PPO/GRPO 则假设这个 token 是模型自己采样出来的。如果整条回答 reward 较高，advantage 为正，就提高该 token 在相同状态下的概率；如果 reward 较低，advantage 为负，就降低概率。

## 9. 为什么数据不能只追求数量

有效数据量近似取决于：

$$
N_{effective}=N\times q\times d\times c
$$

其中：

- $N$：原始样本数。
- $q$：质量比例。
- $d$：去重后的多样性比例。
- $c$：与目标任务相关的覆盖比例。

100 万条高度重复或错误数据，可能不如 2 万条覆盖充分、答案可靠的数据。RL 中还应乘 rollout 数 $G$，但生成轨迹来自同一策略，并不等价于独立人工数据。

### 9.1 数据多的必要性

自然语言存在长尾：同一个“查今天任务”可能写成“今天做啥”“列一下待办”“下一步安排什么”。工具参数还有数字、日期、中英文、缺省值、非法 JSON 等组合。数据越少，模型越可能只记住模板，而不是学到决策边界。

### 9.2 数据少也能立即生效的原因

身份名称属于低熵目标。对于“你是谁”，目标答案几乎固定，少量样本持续提高 `LifeOS-Agent` 对应 token 的 logit，就能改变 greedy decoding。公司场景没生效常见原因不是数据不够，而是：

- 实际加载的不是微调 checkpoint。
- system prompt 覆盖了身份。
- tokenizer/chat template 不一致。
- 推理参数或模型路径不同。
- 训练样本格式没有让 assistant token 进入 loss mask。

## 10. 本项目数据规模与风险

| 数据 | 行数 | P95 token | 当前上限 | 估计截断率 | 主要风险 |
|---|---:|---:|---:|---:|---|
| Pretrain | 1,270,238 | 563 | 512 | 6.10% | 长文本尾部丢失 |
| SFT | 905,718 | 733 | 768 | 3.15% | 少量长多轮截断 |
| DPO | 17,166 对 | 899 | 768 | 14.68% | chosen/rejected 可能截断不对称 |
| RLAIF | 19,502 | 579 | 768 | 0% | reward model 偏差 |
| Agent RL | 39,988 | 1,149 | prompt 768 / total 1600 | 初始估计 22.05% | tools/schema 或早期上下文丢失 |
| LifeOS seed | 26 | 339 | 768 | 0% | 只足够身份和最小工具闭环 |

这些截断率来自抽样 token 估计，Agent RL 实际训练采用动态多轮轨迹，不能把 22.05% 直接解释为最终轨迹截断率。

## 11. 3090 Ti 上的合理训练路线

### 11.1 不要同时跑五种训练

建议链路：

```text
Pretrain -> SFT -> LifeOS 增量 SFT -> DPO
                                  |-> PPO
                                  |-> GRPO
                                  |-> Agent RL
```

PPO、GRPO、Agent RL 都从同一个 DPO checkpoint 分叉，便于公平比较，而不是把 PPO 再接 GRPO 导致无法判断提升来自哪里。

### 11.2 数据优先级

1. 先建立 100 条固定评测集。
2. 检查 SFT 工具格式成功率。
3. 用 DPO 修正明确偏好和拒答边界。
4. 用 1,000-2,000 prompt 做 RL smoke test。
5. reward 与人工评价相关后才跑全量。
6. 每个 checkpoint 用同一评测集比较，不能只看训练 loss。

### 11.3 应报告的指标

- Tool Selection Accuracy。
- Arguments JSON Valid Rate。
- Tool Execution Success Rate。
- Final Answer Accuracy。
- No-tool False Positive Rate。
- Reward mean/std。
- KL、clip fraction、group reward std。
- 平均 response 长度和 P95 延迟。
- 与 SFT 基线相比的普通聊天退化率。

## 12. 阅读源码时的定位表

| 内容 | MiniMind 源码 |
|---|---|
| SFT/DPO/RL Dataset | `dataset/lm_dataset.py` |
| 网络、Attention、CE | `model/model_minimind.py` |
| SFT 训练循环 | `trainer/train_full_sft.py` |
| DPO log-ratio | `trainer/train_dpo.py` |
| PPO GAE/actor/critic | `trainer/train_ppo.py` |
| GRPO/CISPO | `trainer/train_grpo.py` |
| 多轮工具 rollout/reward | `trainer/train_agent.py` |
| reward model wrapper | `trainer/trainer_utils.py` |
| 本项目串行训练配置 | `scripts/remote_rl_worker.sh` |

## 13. 最终记忆框架

先问数据提供了哪一种监督：

```text
给标准答案        -> SFT -> token CE
给好坏答案对      -> DPO -> preference log-ratio
给 prompt+打分器  -> PPO -> actor + critic + GAE
给 prompt+组内比较-> GRPO -> group advantage
给工具环境+结果   -> Agent RL -> trajectory reward + action mask
```

再追踪固定的维度链：

```text
[B,T] token ids
-> [B,T,768] hidden states
-> [B,T,6400] logits
-> gather 为 [B,T] token logps
-> mask / sum / advantage
-> [] scalar loss
-> backward
```

真正的差别不在 Transformer 主干，而在于：**答案从哪里来、哪些 token 参与学习、每条轨迹得到什么分、这个分怎样变成梯度。**
