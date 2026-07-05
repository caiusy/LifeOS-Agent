import re


FAKE_NOTES = [
    {
        "title": "SFTDataset",
        "content": (
            "SFTDataset 负责把 conversations 样本展开成 chat template 文本，再做 tokenize。"
            "其中 system.tools 会被解析成 tools 参数，assistant.tool_calls 会被渲染成 "
            "<tool_call>，tool role 会渲染成 <tool_response>。"
        ),
    },
    {
        "title": "Tool Calling",
        "content": (
            "MiniMind 的 Tool Calling 采用外部循环：模型先输出 <tool_call>，"
            "宿主程序解析后执行工具，把结果作为 role=tool 回填，再让模型继续生成最终回答。"
        ),
    },
    {
        "title": "DPO",
        "content": (
            "DPO 使用 chosen / rejected 成对样本做偏好优化。MiniMind 里 DPODataset "
            "会分别渲染 chosen_prompt 和 rejected_prompt，再构造 loss mask。"
        ),
    },
    {
        "title": "Agentic RL",
        "content": (
            "Agentic RL 关注多轮 rollout。模型生成 tool call 后，环境执行工具并追加 tool "
            "observation，再继续 rollout，并把 observation token 标成非学习部分。"
        ),
    },
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokens(text: str) -> list[str]:
    text = _normalize(text)
    return re.findall(r"[a-z0-9_+.-]+|[\u4e00-\u9fff]+", text)


def search_notes(query: str, limit: int = 3) -> list[dict]:
    query_norm = _normalize(query)
    query_tokens = _tokens(query)
    scored = []
    for idx, note in enumerate(FAKE_NOTES):
        haystack = _normalize(f"{note['title']} {note['content']}")
        score = 0
        for token in query_tokens:
            if token and token in haystack:
                score += max(1, len(token))
        if note["title"].lower() in query_norm:
            score += 10
        scored.append((score, idx, note))

    scored.sort(key=lambda item: (-item[0], item[1]))
    top_notes = [note for score, _, note in scored if score > 0][:limit]
    if not top_notes:
        top_notes = FAKE_NOTES[:limit]

    return [{"title": note["title"], "content": note["content"]} for note in top_notes]
