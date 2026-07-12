# LifeOS-Agent v0.1 评测集

`lifeos_eval.jsonl` 是固定的 100 条能力评测集，用于公平比较 SFT、DPO、PPO、GRPO 和 Agent RL checkpoint。

## 类别

| 类别 | 数量 | 目标 |
|---|---:|---|
| `note_search` | 20 | 笔记工具选择与 query 提取 |
| `today_tasks` | 10 | 无参数工具调用与结果利用 |
| `math` | 15 | 表达式提取与数值正确性 |
| `no_tool_chat` | 15 | 普通对话不得误调用工具 |
| `multi_candidate` | 10 | 路由多个候选 schema |
| `tool_error` | 5 | 工具安全失败与错误回答 |
| `parser_robustness` | 10 | `<tool_call>` 边界与 JSON 解析 |
| `safety` | 15 | 危险请求、伪造结果和越权操作 |

## 字段

- `expected_tool`：`null` 表示不应传 tools；字符串表示唯一工具；数组表示路由后的有序候选工具。
- `expected_arguments`：语义上的标准参数。模型评测时数学表达式应按执行结果等价判断，不要求逐字符相同。
- `expected_tool_result`：确定性工具的标准输出；`error_contains` 表示错误文本必须包含的片段。
- `final_answer_contains`：最终自然语言回答应覆盖的关键内容。
- `model_output_fixture`：仅用于解析器边界测试，不发送给模型。

## 验证

```bash
python scripts/build_eval_dataset.py
python scripts/validate_eval_dataset.py
```

验证器不加载模型，检查样本数量、ID、工具名称、路由结果、解析器和工具执行器。模型 checkpoint 横向对比应固定 tokenizer、解码参数和本文件版本，并分别记录工具选择率、参数执行成功率、最终答案正确率、误调用率与延迟。
