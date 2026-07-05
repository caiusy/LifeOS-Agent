def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def select_tool_names(user_input: str) -> list[str]:
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

    if not selected:
        selected = ["search_fake_obsidian", "list_today_tasks", "calculate_math"]

    return selected
