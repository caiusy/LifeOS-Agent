"""Build a Hexo-compatible copy of the complete LifeOS training guide."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "TRAINING_METHODS_COMPLETE_GUIDE.md"
OUTPUT = ROOT / "build" / "LifeOS-Agent-SFT-DPO-PPO-GRPO-Agent-RL.md"


FRONT_MATTER = """---
title: LifeOS-Agent 训练全解：从 SFT、DPO、PPO、GRPO 到多轮 Agent RL
date: 2026-07-13 23:00:00
updated: 2026-07-13 23:00:00
mathjax: true
description: "基于 MiniMind 与 RTX 3090 Ti 的完整训练实战：真实数据样本、Transformer 张量维度、五类损失函数，以及 Agent RL 多轮工具调用从 rollout、observation mask、reward、advantage 到 CISPO loss 的逐步推导。"
categories:
  - AI与大模型
  - 深度学习
tags:
  - LifeOS-Agent
  - MiniMind
  - SFT
  - DPO
  - PPO
  - GRPO
  - Agent-RL
type: deep-dive
difficulty: advanced
review_status: published
---

> 这是一篇从真实工程出发的训练教材。目标不是背公式，而是能拿着一条 JSONL 样本，完整说清它如何变成张量、经过网络、获得 reward、形成 loss 并更新参数。

<!-- more -->

"""


def wrap_display_math(markdown: str) -> str:
    pattern = re.compile(r"(^\$\$\n.*?^\$\$$)", re.MULTILINE | re.DOTALL)
    return pattern.sub(lambda match: "{% raw %}\n" + match.group(1) + "\n{% endraw %}", markdown)


def main():
    body = SOURCE.read_text(encoding="utf-8")
    body = re.sub(r"^# .+?\n", "", body, count=1)
    body = body.replace("assets/", "/images/lifeos-agent-training/")
    body = wrap_display_math(body)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(FRONT_MATTER + body, encoding="utf-8")
    print(f"wrote {OUTPUT} ({len((FRONT_MATTER + body).splitlines())} lines)")


if __name__ == "__main__":
    main()
