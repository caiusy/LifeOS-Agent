def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def select_tool_names(user_input: str) -> list[str]:
    """用轻量关键词路由候选工具。

    返回空列表是有意设计：普通聊天不传 tools，让模型走纯对话路径；
    返回多个工具时，chat template 会把这些候选函数的 schema 一起渲染。
    """
    text = user_input.lower()
    selected = []

    if _contains_any(
        text,
        [
            "sft",
            "sftdataset",
            "dpo",
            "tool calling",
            "toolcalling",
            "agentic rl",
            "之前学到哪",
            "学到哪",
            "笔记",
        ],
    ):
        selected.append("search_fake_obsidian")

    if _contains_any(
        text,
        [
            "今天",
            "任务",
            "做什么",
            "计划",
        ],
    ):
        selected.append("list_today_tasks")

    if _contains_any(
        text,
        [
            "算",
            "计算",
            "多少",
            "乘以",
            "涨停价",
        ],
    ):
        selected.append("calculate_math")

    return selected
