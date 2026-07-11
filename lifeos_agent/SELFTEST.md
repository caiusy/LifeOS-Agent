# LifeOS-Agent Self Test

## 核心检查项

1. 只传候选工具，不传全量工具
2. `apply_chat_template(..., tools=tools)` 正确注入 schema
3. `<tool_call>...</tool_call>` 可被解析
4. `execute_tool()` 支持：
   - 工具名不存在
   - `arguments` 是 dict
   - `arguments` 是 JSON string
   - JSON 格式错误
5. tool result 以 `role="tool"` 回填
6. 最多 `3` 轮，避免死循环
7. 普通聊天默认 `tools=None`
8. README 命令可执行

## 验收问题

### 1. 检索类

输入：

- `我之前学 SFTDataset 学到哪了？`

期望：

- 候选工具：`search_fake_obsidian`
- 模型输出 `<tool_call>`
- 工具结果中包含 `SFTDataset` 相关说明

### 2. 任务类

输入：

- `我今天应该做什么？`

期望：

- 候选工具：`list_today_tasks`
- 工具结果：
  - `整理 Tool Calling 笔记`
  - `复习 SFTDataset`
  - `跑通 LifeOS-Agent v0.1`

### 3. 计算类

输入：

- `17.66 涨停价是多少？`

期望：

- 候选工具：`calculate_math`
- 结果接近 `19.43`

### 4. 普通闲聊

输入：

- `你好，简单介绍一下你自己`

期望：

- 不传工具
- 不调用工具
- 直接回答

## 结果判定

如果 4 个 case 都满足下面条件，则认为 `v0.1` 验收通过：

1. 命中的 case 能正确选择工具
2. 工具结果被回填后模型能继续回答
3. 非工具 case 不误调工具
4. 最多 `3` 轮后强制停止
