# LifeOS-Agent User Guide

这份文档面向实际使用，少讲训练，多讲怎么对话、怎么验证、怎么判断是否正常。

## 1. 最快开始

在本地项目目录运行：

```bash
cd /Users/caius/Documents/LifeOS-Agent
bash scripts/run_remote_lifeos_best.sh "我今天应该做什么？"
```

这条命令会连接远程 `wsl-dev`，加载：

```text
/home/caius/minimind/out/lifeos_agent_best_768.pth
```

## 2. 连续对话

如果想在远程机器上连续输入：

```bash
ssh wsl-dev
source /home/caius/lead-3d/venv/bin/activate
cd /home/caius/projects/LifeOS-Agent
python lifeos_agent/main.py \
  --minimind_repo /home/caius/minimind \
  --tokenizer_path /home/caius/minimind/model \
  --checkpoint_path /home/caius/minimind/out/lifeos_agent_best_768.pth
```

看到 `User>` 后就可以输入问题。

## 3. 可以问什么

### 笔记类

```text
我之前学 SFTDataset 学到哪了？
帮我查一下 Tool Calling 的笔记
把 Agentic RL 那条笔记找出来
```

预期工具：

```text
search_fake_obsidian
```

### 任务类

```text
我今天应该做什么？
今天的任务清单是什么？
我今天的计划是什么？
```

预期工具：

```text
list_today_tasks
```

### 计算类

```text
17.66 涨停价是多少？
帮我算一下 256 乘以 37
1024 除以 16 等于多少？
```

预期工具：

```text
calculate_math
```

### 普通聊天

```text
你好，简单介绍一下你自己
你能做哪些事？
我们现在在做什么？
```

预期行为：

```text
不传 tools，不调用工具，直接回答。
```

## 4. 如何看输出

程序会打印几段信息。

### Selected tools

```text
Selected tools: ['list_today_tasks']
```

说明 router 命中了候选工具。

如果是普通聊天，应该看到：

```text
Selected tools: []
```

### Prompt / input_text

这里能看到 `apply_chat_template` 生成的完整 prompt。命中工具时，里面应该包含 `<tools>...</tools>`。

### Model output

第一轮如果需要工具，应该看到：

```xml
<tool_call>
{"name": "...", "arguments": {...}}
</tool_call>
```

### Tool result

说明 Python 已经执行了工具。

### Final answer

最终给用户看的答案。若模型第二轮回答退化，程序会打印 `Fallback answer`，再把 fallback 作为最终答案。

## 5. 批量自测

```bash
bash scripts/remote_lifeos_selftest.sh
```

它会一次跑 4 个验收问题：

1. 检索类
2. 任务类
3. 计算类
4. 普通聊天

## 6. 常见问题

### 模型没有调用工具

先看 `Selected tools` 是否为空。如果为空，说明 router 没命中关键词，需要补路由词或改问法。

### 模型调用了工具，但最终回答重复

当前小模型可能出现这种情况。运行时 fallback 会在明显退化时给出保底答案。

### 普通聊天出现工具相关提示

这通常是小模型复读 system prompt。当前版本已经加了 no-tool fallback，后续可以继续补 no-tool seed 数据来改善。

## 7. 当前最佳使用姿势

现在最适合把它当作一个“训练完成的 Tool Calling 原型”来使用：

1. 用它验证工具调用链路
2. 用它测试 fake tools
3. 逐步把 fake notes 替换为真实 Obsidian 检索
4. 继续积累真实问法，扩充 seed 数据
