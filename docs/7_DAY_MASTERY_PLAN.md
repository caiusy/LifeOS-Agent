# LifeOS-Agent 7 天掌握计划

## 1. 目标与现实边界

每天只投入 45 分钟，7 天共 315 分钟。本计划的“彻底掌握”限定为：

1. 能闭卷讲清 LifeOS-Agent 从用户问题到最终回答的完整 Tool Calling 数据流。
2. 能用一个典型样本说明 SFT、DPO、PPO、GRPO、Agent RL 的数据、张量、loss 和差异。
3. 能独立准备数据、选择 checkpoint、配置训练、启动训练、阅读日志并定位常见失败。
4. 能完成一场围绕本项目的 20 分钟技术面试，并回答追问。

7 天不足以覆盖强化学习的全部理论或达到论文研究级熟练度，但足以达到“本项目可独立训练、核心原理可推导、面试可稳定表达”。

## 2. 当前起点

根据现有交流，当前水平判断为“有项目经验的入门进阶阶段”：

- 已掌握：知道 router、tools schema、`<tool_call>`、Python executor、`role=tool` 回填和 token loss 等关键对象。
- 已具备：能提出 `messages[:-1]`、候选工具由谁选择、tool-call 字符串是否计算 loss 等有效问题。
- 主要缺口：知识仍是点状的；尚未稳定做到闭卷串联、张量手算、五种方法横向比较和独立排错。
- 最大风险：继续增加文档和训练次数，却没有形成自己的可检验输出。

因此，这 7 天禁止以“继续泛读”作为主要任务。

## 3. 每日统一规则

每天严格使用同一节奏完成一个综合任务：

| 时间 | 动作 |
| --- | --- |
| 0～5 分钟 | 闭卷写出昨天最重要的 5 个关键词和关系 |
| 5～15 分钟 | 只阅读当天指定章节，补齐闭卷暴露的缺口 |
| 15～35 分钟 | 完成当天唯一产物，不照抄文档 |
| 35～45 分钟 | 口述、答题或实际操作验收 |

每天满分 100 分，`85` 分及以上才算通过。出现以下任一情况，即使总分达到 85 也不通过：

- 把 router 说成模型内部自动拥有的能力。
- 把 Python 工具执行说成 Transformer 内部计算。
- 说不清哪些 token 是 action、哪些是 observation。
- 只会背公式，无法说明公式中每个量来自哪一步。
- 训练时说不清起始 checkpoint、数据路径、输出 checkpoint 和验收集。

## 4. 七天任务

## Day 1：闭卷重建 Tool Calling 外部循环

### 唯一任务

围绕“17.66 涨停价是多少？”画出并口述一次完整调用，必须出现：

```text
user_input
-> select_tool_names
-> candidate tools schema
-> apply_chat_template
-> input_ids
-> first model output
-> parse_tool_calls
-> execute_tool
-> role=tool message
-> second apply_chat_template
-> final answer
```

### 指定材料

- `lifeos_agent/main.py`
- `lifeos_agent/router.py`
- `lifeos_agent/tools.py`
- `docs/AGENT_RL_COMPLETE_GUIDE.md` 第 3～7 章

### 必交产物

一张手绘或 Markdown 流程图，加一份 3 分钟口述。每个箭头必须说明输入、输出和执行者是 LLM 还是 Python。

### 完成标准

- 20 分：候选工具是 router 筛选的，不是模型凭空知道的。
- 20 分：schema 确实进入 `input_text`。
- 20 分：模型输出的是字符串，parser 转为 Python 对象。
- 20 分：工具结果以 `role=tool` 回填后重新 tokenize。
- 20 分：解释最大 3 轮和普通聊天不传 tools。

### 避免浪费时间

不要读训练公式，不要讨论 PPO，不要优化代码。Day 1 只解决推理链路。

## Day 2：从一条 SFT 数据手算到 Cross Entropy

### 唯一任务

选择一条包含 `<tool_call>` 的 SFT 样本，完整说明：JSON 如何变成 prompt、token、labels、logits 和标量 loss。

### 指定材料

- `docs/TRAINING_METHODS_COMPLETE_GUIDE.md` 第 1～2 章
- `MATHEMATICAL_DERIVATIONS.md` 中 Softmax 与交叉熵部分
- `dataset/lifeos_sft_seed.jsonl` 中一条工具样本

### 必交产物

一张维度表，至少包含：

```text
input_ids     [B, T]
embedding     [B, T, H]
logits        [B, T, V]
shift_logits  [B, T-1, V]
shift_labels  [B, T-1]
loss          scalar
```

再用一个 3 分类 toy logits 手算一次 softmax 和 cross entropy。

### 完成标准

- 20 分：解释 next-token shift，而不是只背维度。
- 20 分：说明 `B/T/H/V` 的含义。
- 20 分：正确识别 assistant 的 tool-call token 需要 SFT loss。
- 20 分：说明 user/system/tool observation 如何被 mask。
- 20 分：能从概率手算 `-log p(label)`。

### 避免浪费时间

不要钻研所有注意力变体；只要求理解数据如何穿过 Transformer 并形成 loss。

## Day 3：用同一问题讲清 SFT 与 DPO

### 唯一任务

对同一个 Tool Calling 问题，分别构造一条 SFT 样本和一组 DPO chosen/rejected，并说明二者如何更新模型。

### 指定材料

- `docs/TRAINING_METHODS_COMPLETE_GUIDE.md` 第 2～3 章
- `dataset/minimind_dataset/dpo.jsonl` 的一条真实样本

### 必交产物

一页对比稿，必须写出：数据字段、模型前向次数、reference model、序列 log probability、loss 目标和适用场景。

### 完成标准

- 25 分：SFT 学“标准答案 token”，DPO 学“chosen 相对 rejected 更优”。
- 20 分：能解释序列 log probability 是 token log probability 之和。
- 20 分：解释 reference model 为什么冻结。
- 20 分：能口述 DPO loss 中 `beta` 和 sigmoid 的作用。
- 15 分：能说明 DPO 不能替代 Tool Calling SFT 的基础格式学习。

### 避免浪费时间

不要背论文历史，不比较十种 DPO 变体，只掌握标准 DPO。

## Day 4：用一张表和一组数字吃透 PPO 与 GRPO

### 唯一任务

针对同一个 prompt 的 4 条候选回答，给出 4 个 reward，手算组均值、标准差和 GRPO advantage；再说明 PPO 会如何使用 critic 与 GAE。

### 指定材料

- `docs/TRAINING_METHODS_COMPLETE_GUIDE.md` 第 4～5 章
- `MATHEMATICAL_DERIVATIONS.md` 中 Policy Gradient、GAE、PPO、GRPO 部分

### 必交产物

一张 PPO/GRPO 对照表，加一次四样本 advantage 手算。

### 完成标准

- 20 分：正确区分 actor、old policy、reference policy 和 critic。
- 20 分：能解释 importance ratio `pi_new/pi_old`。
- 20 分：能解释 clipping 防止单步更新过大。
- 20 分：手算 GRPO 组内标准化 advantage。
- 20 分：明确 GRPO 不训练 critic，但仍计算生成 token 的策略 loss。

### 避免浪费时间

不要追求完整证明收敛性。重点是每个数从哪里来、最后怎样影响 token 概率。

## Day 5：完整推导一次 Agent RL 多轮轨迹

### 唯一任务

把 Day 1 的工具调用链与 Day 4 的策略优化合并，完整解释一条 Agent RL rollout 怎样变成 loss。

### 指定材料

- `docs/AGENT_RL_COMPLETE_GUIDE.md` 第 8～20 章
- `docs/assets/agent_rl_end_to_end_trace.svg`
- `scripts/trace_agent_rl_example.py`

### 必交产物

一张 token 账本，逐类标记：prompt token、tool-call action token、tool observation token、final-answer action token、padding token；注明哪些进入 policy loss。

### 完成标准

- 20 分：完整描述多轮 rollout，而不是把它当单轮问答。
- 20 分：reward、advantage、ratio、KL、mask、loss 顺序正确。
- 20 分：说明 `<tool_call>` 中每个有效 action token 为什么计算 loss。
- 20 分：说明工具结果是 observation，为何不作为 policy action 训练。
- 20 分：解释 v1 奖励漏洞为何允许错误轨迹得分，以及 v2 如何修复。

### 避免浪费时间

不要只看最终 reward；必须追到 token mask 和标量 loss。

## Day 6：独立完成一次训练前审计与启动

### 唯一任务

不复制现成命令，独立写出并执行一份“小规模训练启动清单”。为了节省时间，只做 smoke run，不追求训练完成。

### 必须独立决定

```text
训练目标
训练方法
起始 checkpoint
训练数据与样本数
max_seq_len
batch_size
gradient accumulation
learning rate
输出路径
日志路径
验收集
停止/回滚条件
```

### 必交产物

一份可执行命令、一段启动日志、一次 GPU/显存观察，以及对首个 loss/reward 指标的解释。

### 完成标准

- 20 分：路径和 checkpoint 关系正确，不覆盖生产模型。
- 20 分：能预测主要张量维度与显存压力。
- 20 分：训练实际启动，日志和 checkpoint 输出位置明确。
- 20 分：能识别 OOM、NaN、reward 不增长、格式退化四类问题。
- 20 分：知道训练后必须做 tool-call、grounding 和 no-tool 三类验收。

### 避免浪费时间

不要跑长训练，不调几十组超参数，不以“进程启动成功”冒充模型效果成功。

## Day 7：闭卷答辩、面试与独立方案设计

### 唯一任务

完成一次 45 分钟终局验收：10 分钟白板讲解、20 分钟面试追问、10 分钟故障分析、5 分钟复盘。

### 10 分钟白板题

从一条 LifeOS 工具样本开始，讲完：

```text
数据 -> template -> token -> Transformer -> tool call
-> Python 环境 -> 多轮轨迹 -> reward -> advantage
-> token loss -> backward -> checkpoint -> eval
```

### 20 分钟面试题范围

1. 为什么工具 schema 必须进入上下文？
2. router 与模型工具选择的边界是什么？
3. Tool Calling 是 SFT 学的还是 Agent RL 学的？
4. tool-call token 是否计算 loss？
5. `role=tool` 为什么不是 assistant？
6. SFT 与 DPO 的监督信号有何不同？
7. PPO 为什么需要 critic，GRPO 为什么可以不用？
8. KL、ratio 和 clipping 分别防止什么？
9. Agent RL 为什么慢，reward hacking 如何发生？
10. 3090 Ti 上如何设计安全的小步实验？

### 10 分钟故障题

给定“训练 reward 上升，但工具调用从 3/3 降到 0/3”，必须提出检查顺序，而不是直接要求增加 epoch。

### 完成标准

- 原理与数据流：30 分，至少 26 分。
- 数学与维度：25 分，至少 21 分。
- 训练与排错：25 分，至少 21 分。
- 表达与追问：20 分，至少 17 分。
- 总分至少 85，且所有分项过线。

### 避免浪费时间

不要临时补读，不看稿复述，不用术语堆砌掩盖数据流断点。

## 5. 未达标时的自动修改规则

每天验收后只记录三个字段：`得分`、`第一个断点`、`错误类型`。

### 得分 85～100

按计划进入下一天。第二天前 5 分钟复述昨天内容。

### 得分 70～84

第二天仍学习新主题，但前 15 分钟只修复第一个断点；当天新主题产物缩减为核心版本，禁止延长总时间。

### 得分低于 70 或触发红线错误

不进入下一主题。第二天替换为该日的“第二次闭卷版本”：换一个样本重新完成同一任务。后续计划整体顺延；不要为了七个日历日而伪造掌握。

### Day 7 未达到 85

根据最低分项生成 3 天补强计划：

- 原理低：重做 Day 1 和 Day 5，不增加阅读材料。
- 数学低：只做 toy number 手算，不再泛读推导。
- 训练低：再做一次不同 checkpoint 的 smoke run 和故障诊断。
- 表达低：每天录一次 5 分钟闭卷讲解，并按事实错误逐条修正。

连续两次终局验收达到 85，且第二次使用不同样本，才认定为“稳定掌握”。

## 6. 面试回答统一结构

所有技术题都使用四步表达，避免回答散乱：

1. **一句话定义**：它是什么。
2. **数据流位置**：输入和输出是什么。
3. **公式或维度**：至少给出一个关键公式或 shape。
4. **本项目证据**：指出 LifeOS-Agent 中的实现、训练结果或故障案例。

例如回答“tool-call token 是否计算 loss”：

> tool call 是模型生成的 action 文本。SFT 时，它位于 assistant 区域，因此有效 token 参与交叉熵；Agent RL 时，它属于 rollout 的 action token，因此在 response mask 内参与策略 loss。Python 返回的 `role=tool` 内容是环境 observation，用作下一轮条件，但不是策略生成的 action。本项目 v2 正是通过提高错误 tool call 的轨迹惩罚，改变这些 action token 的更新方向。

## 7. 七天内明确不做的事

- 不继续长时间训练，不追新 checkpoint。
- 不接真实 Obsidian、向量数据库或 MCP。
- 不学习 LoRA、QLoRA、RLAIF 等旁支主题。
- 不比较大量 RL 算法变体。
- 不继续扩写长文档，除非验收暴露了明确知识缺口。
- 不把运行命令交给助手后只看结论；Day 6 必须亲自完成一次独立启动。
- 不用“我大概懂了”作为通过标准，只接受产物、口述、手算和实际操作。
