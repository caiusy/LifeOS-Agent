"""Build the standalone Agent RL guide as a Hexo-compatible article."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "AGENT_RL_COMPLETE_GUIDE.md"
OUTPUT = ROOT / "build" / "Agent-RL-Multiturn-Tool-Use-Complete-Guide.md"


FRONT_MATTER = """---
title: Agent RL 全链路教程：从工具选择、多轮对话到 Reward、Advantage 与 Token Loss
date: 2026-07-13 23:40:00
updated: 2026-07-13 23:40:00
mathjax: true
description: "以 MiniMind 和 LifeOS-Agent 的真实代码为主线，用一条数学工具调用和四条 rollout，完整解释 schema、router、多轮 observation、张量维度、reward、group advantage、KL、CISPO token loss 与反向传播。"
categories:
  - AI与大模型
  - Agent
tags:
  - Agent-RL
  - Tool-Calling
  - MiniMind
  - LifeOS-Agent
  - GRPO
  - CISPO
type: deep-dive
difficulty: progressive
review_status: published
---

> 这是一篇从小学生直觉逐层走到高中数学与工程代码的独立教材。所有工程结论均映射到当前 MiniMind/LifeOS-Agent 实现，所有教学数字均明确标注，不把假设当实测。

<!-- more -->

"""


def wrap_display_math(markdown: str) -> str:
    """Keep display math untouched by Hexo's Markdown renderer."""
    pattern = re.compile(r"(^\$\$\n.*?^\$\$)", re.MULTILINE | re.DOTALL)
    return pattern.sub(lambda match: "{% raw %}\n" + match.group(1) + "\n{% endraw %}", markdown)


def main() -> None:
    body = SOURCE.read_text(encoding="utf-8")
    body = re.sub(r"^# .+?\n", "", body, count=1)
    body = body.replace("assets/", "/images/lifeos-agent-training/")
    body = wrap_display_math(body)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    article = FRONT_MATTER + body
    OUTPUT.write_text(article, encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(article.splitlines())} lines)")


if __name__ == "__main__":
    main()
