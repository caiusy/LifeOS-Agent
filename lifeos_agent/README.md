# LifeOS-Agent v0.1

这是一个基于 MiniMind Tool Calling 外部循环做的最小可运行 demo，用来验证下面这条链路：

1. 用户提问
2. 模型输出 `<tool_call>`
3. 外部 Python 程序执行工具
4. 工具结果以 `role=tool` 回填
5. 模型第二轮继续生成最终回答

## 文件说明

- `lifeos_agent/main.py`
  - 主入口，负责加载模型、调用 `apply_chat_template`、解析 `<tool_call>`、执行工具、回填 `role=tool`。
- `lifeos_agent/tools.py`
  - fake tools 定义与执行器。
- `lifeos_agent/router.py`
  - 根据用户输入挑选候选工具。
- `lifeos_agent/fake_notes.py`
  - 内置 fake Obsidian 笔记和简单搜索逻辑。

## 当前工具

- `calculate_math`
- `list_today_tasks`
- `search_fake_obsidian`

## 运行方式

### 方式一：使用 Transformers 格式本地模型

```bash
python lifeos_agent/main.py \
  --hf_model_path /path/to/your/minimind-hf \
  --tokenizer_path /path/to/minimind-master/model \
  --prompt "我之前学 SFTDataset 学到哪了？"
```

### 方式二：使用 MiniMind 原生 `.pth` 权重

```bash
python lifeos_agent/main.py \
  --minimind_repo /path/to/minimind-master \
  --tokenizer_path /path/to/minimind-master/model \
  --checkpoint_path /path/to/full_sft_768.pth \
  --prompt "我今天应该做什么？"
```

### 方式三：直接在你的远程 3090 Ti 上跑 HF 模型

如果你已经把项目同步到：

- `/home/caius/projects/LifeOS-Agent`
- `/home/caius/models/minimind-3`

可以在本地执行：

```bash
bash scripts/run_remote_3090_demo.sh "我今天应该做什么？"
```

### 方式四：直接运行远程最佳 LifeOS 权重

如果你已经完成了远程增量 SFT，并且存在：

- `/home/caius/minimind/out/lifeos_agent_best_768.pth`

可以在本地执行：

```bash
bash scripts/run_remote_lifeos_best.sh "我今天应该做什么？"
```

批量跑 4 个验收问题：

```bash
bash scripts/remote_lifeos_selftest.sh
```

## 你会看到的输出

程序会打印：

- 选中了哪些工具
- `input_text`
- 第一轮模型输出
- 解析出的 `tool_call`
- fake tool 执行结果
- 第二轮最终回答

## 验收用例

### 1. 学习进度

```bash
python lifeos_agent/main.py \
  --minimind_repo /path/to/minimind-master \
  --tokenizer_path /path/to/minimind-master/model \
  --checkpoint_path /path/to/full_sft_768.pth \
  --prompt "我之前学 SFTDataset 学到哪了？"
```

预期：

- 候选工具包含 `search_fake_obsidian`
- 模型优先调用 `search_fake_obsidian`

### 2. 今日任务

```bash
python lifeos_agent/main.py \
  --minimind_repo /path/to/minimind-master \
  --tokenizer_path /path/to/minimind-master/model \
  --checkpoint_path /path/to/full_sft_768.pth \
  --prompt "我今天应该做什么？"
```

预期：

- 候选工具包含 `list_today_tasks`
- 模型优先调用 `list_today_tasks`

### 3. 涨停价

```bash
python lifeos_agent/main.py \
  --minimind_repo /path/to/minimind-master \
  --tokenizer_path /path/to/minimind-master/model \
  --checkpoint_path /path/to/full_sft_768.pth \
  --prompt "17.66 涨停价是多少？"
```

预期：

- 候选工具包含 `calculate_math`
- 模型调用 `calculate_math`
- 表达式接近 `round(17.66 * 1.1, 2)` 或 `17.66 * 1.1`
- 最终结果接近 `19.43`

## 后续接真实 Obsidian 的方向

第一步建议只替换 `search_fake_obsidian`：

1. 把 `fake_notes.py` 换成真实笔记读取器
2. 先支持扫描 Markdown 文件和标题匹配
3. 再补 BM25 / embedding 检索
4. 最后再接文件写入、任务更新、双向同步

这样可以保留当前的 Tool Calling 主循环，不需要重写 agent 框架。
