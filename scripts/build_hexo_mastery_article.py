"""Build the Hexo-safe publication edition of the LifeOS mastery textbook."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


FRONT_MATTER = """---
title: LifeOS-Agent 从零到独立训练：Transformer、SFT、DPO、PPO、GRPO 与 Agent RL 完整推导
date: 2026-07-23 20:30:00
updated: 2026-07-23 20:30:00
mathjax: true
description: "从张量、概率、梯度和 Transformer 开始，以 17.66 涨停价工具调用为贯穿样例，逐步推导 SFT、DPO、PPO、GRPO、CISPO 与 Agent RL，并落到 MiniMind/LifeOS-Agent 的真实数据、张量维度、训练代码、3090 Ti 调参与发布验收。"
categories:
  - AI与大模型
  - 模型训练
tags:
  - LifeOS-Agent
  - MiniMind
  - Transformer
  - SFT
  - DPO
  - PPO
  - GRPO
  - Agent-RL
  - Tool-Calling
type: handbook
difficulty: progressive
review_status: published
---

> 这不是一篇只列名词的综述，而是一份可以从第一页连续学习到独立训练的主教材。我们从“标量、向量、概率和梯度是什么”开始，最终把一条多轮工具轨迹算到逐 token policy loss，并给出真实 MiniMind/LifeOS-Agent 代码位置、3090 Ti 训练步骤和面试验收标准。
>
> 全文只有一个主问题：**一条用户消息如何经过 tokenizer、Transformer、工具环境和训练目标，最终改变模型参数？**

<!-- more -->

![LifeOS-Agent 从数据到训练与评测的完整流水线](/images/lifeos-agent-zero-to-mastery/lifeos_training_pipeline.svg)

"""


IMAGE_INSERTIONS = {
    "## 3. Transformer 数据流与精确维度": (
        "![MiniMind Transformer 的核心张量维度]"
        "(/images/lifeos-agent-zero-to-mastery/network_tensor_dimensions.svg)\n\n"
    ),
    "## 10. Tool Calling 外部循环完整数据流": (
        "![LifeOS-Agent 多轮 Tool Calling 数据流]"
        "(/images/lifeos-agent-zero-to-mastery/agent_rl_multiturn_dataflow.svg)\n\n"
    ),
    "## 11. Agent RL：把工具环境放进在线 Rollout": (
        "![Agent RL 从 Reward 到逐 Token Loss]"
        "(/images/lifeos-agent-zero-to-mastery/agent_rl_reward_to_loss.svg)\n\n"
    ),
    "## 12. 五种训练方法统一对照": (
        "![SFT、DPO、PPO、GRPO 与 Agent RL 的目标对照]"
        "(/images/lifeos-agent-zero-to-mastery/five_stage_training_overview.svg)\n\n"
        "![五种训练方法的 Loss 计算差异]"
        "(/images/lifeos-agent-zero-to-mastery/loss_computation_comparison.svg)\n\n"
    ),
    "## 19. 一条 Agent RL 轨迹从文本到 Loss 的总算例": (
        "![一条 Agent RL 工具轨迹的端到端 Trace]"
        "(/images/lifeos-agent-zero-to-mastery/agent_rl_end_to_end_trace.svg)\n\n"
    ),
}


def remove_title(source: str) -> str:
    lines = source.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).rstrip() + "\n"


def replace_mermaid_map(source: str) -> str:
    pattern = re.compile(
        r"## 全书路线图\n\n```mermaid\n.*?\n```\n",
        flags=re.DOTALL,
    )
    replacement = (
        "## 全书路线图\n\n"
        "![从数学基础到 Agent RL 部署的学习地图]"
        "(/images/lifeos-agent-zero-to-mastery/agent_rl_learning_map.svg)\n"
    )
    return pattern.sub(replacement, source, count=1)


def insert_images(source: str) -> str:
    for heading, image in IMAGE_INSERTIONS.items():
        marker = f"{heading}\n\n"
        if marker not in source:
            raise ValueError(f"missing heading for image insertion: {heading}")
        source = source.replace(marker, f"{marker}{image}", 1)
    return source


def sanitize_math_angles(source: str) -> str:
    """Replace browser-unsafe angle brackets only inside TeX expressions."""
    lines = source.splitlines()
    output: list[str] = []
    in_display_math = False
    in_code = False
    inline_math = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

    def safe_tex(match: re.Match[str]) -> str:
        expression = match.group(1).replace("<", r"\lt ").replace(">", r"\gt ")
        return f"${expression}$"

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            output.append(line)
            continue
        if not in_code and stripped == "$$":
            in_display_math = not in_display_math
            output.append(line)
            continue
        if not in_code and in_display_math:
            line = line.replace("<", r"\lt ").replace(">", r"\gt ")
        elif not in_code:
            line = inline_math.sub(safe_tex, line)
        output.append(line)

    if in_display_math:
        raise ValueError("unbalanced display math during sanitization")
    if in_code:
        raise ValueError("unbalanced code fence during sanitization")
    return "\n".join(output).rstrip() + "\n"


def wrap_display_math(source: str) -> str:
    lines = source.splitlines()
    output: list[str] = []
    in_math = False
    in_code = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            output.append(line)
            continue
        if not in_code and stripped == "$$":
            if not in_math:
                output.extend(["{% raw %}", "$$"])
                in_math = True
            else:
                output.extend(["$$", "{% endraw %}"])
                in_math = False
            continue
        output.append(line)

    if in_math:
        raise ValueError("unbalanced display math")
    if in_code:
        raise ValueError("unbalanced code fence")
    return "\n".join(output).rstrip() + "\n"


def validate_source(source: str) -> None:
    if source.count("{% raw %}") != source.count("{% endraw %}"):
        raise ValueError("unbalanced Hexo raw tags")
    if source.count("\n$$\n") % 2:
        raise ValueError("unbalanced display math delimiters")
    if source.count("```") % 2:
        raise ValueError("unbalanced code fences")
    if "```mermaid" in source:
        raise ValueError("Mermaid block should be replaced with a static SVG")


def build(source_path: Path, output_path: Path) -> None:
    source = source_path.read_text(encoding="utf-8")
    article = remove_title(source)
    article = replace_mermaid_map(article)
    article = insert_images(article)
    article = sanitize_math_angles(article)
    article = wrap_display_math(article)
    article = FRONT_MATTER + article
    validate_source(article)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(article, encoding="utf-8")
    print(f"WROTE {output_path} ({len(article.splitlines())} lines)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("docs/LIFEOS_AGENT_ZERO_TO_MASTERY.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/LifeOS-Agent-Zero-to-Mastery-Blog.md"),
    )
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
