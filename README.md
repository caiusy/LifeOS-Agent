# LifeOS-Agent

`LifeOS-Agent` 是一个基于 MiniMind Tool Calling 外部循环实现的最小 Agent Demo。

当前版本 `v0.1` 主要验证这条链路：

1. 用户提问
2. 模型输出 `<tool_call>`
3. 外部程序执行工具
4. 工具结果以 `role=tool` 回填
5. 模型第二轮继续生成最终回答

当前实现位于 [lifeos_agent](lifeos_agent/)。

详细说明见：

- [lifeos_agent/README.md](lifeos_agent/README.md)

## 当前工具

- `calculate_math`
- `list_today_tasks`
- `search_fake_obsidian`

## 运行示例

```bash
python lifeos_agent/main.py \
  --checkpoint_path /path/to/full_sft_768.pth \
  --prompt "我之前学 SFTDataset 学到哪了？"
```
