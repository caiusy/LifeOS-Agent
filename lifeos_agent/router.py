def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def select_tool_names(user_input: str) -> list[str]:
    """用轻量关键词路由候选工具。

    返回空列表是有意设计：普通聊天不传 tools，让模型走纯对话路径；
    返回多个工具时，chat template 会把这些候选函数的 schema 一起渲染。
    """
    text = user_input.lower()

    # v0.1 的工具都是只读/计算工具。危险、越权或要求伪造执行结果的请求
    # 不应仅因命中“笔记/任务”等普通关键词就获得工具 schema。
    blocked_intents = [
        "删除",
        "rm -rf",
        "密码",
        "关闭服务器",
        "绕过确认",
        "伪造",
        "系统提示词",
        "不存在的",
        "无限循环",
        "编一个答案",
        "上传到公开",
        "未经确认",
        "忽略工具返回",
        "api 密钥",
        "api key",
    ]
    if _contains_any(text, blocked_intents):
        return []

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
