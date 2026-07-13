"""Reproduce the token-level measurements used by the Agent RL guide.

This script needs only the MiniMind tokenizer, not model weights or a GPU.
It intentionally uses fixed messages and outputs so documentation numbers are
deterministic and can be checked after tokenizer/template changes.
"""

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_MINIMIND_REPO = os.environ.get(
    "MINIMIND_REPO", "/Users/caius/Documents/alma/github/minimind-master"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "计算数学表达式的结果，支持加减乘除、幂运算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如123+456、2**10",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

BASE_MESSAGES = [
    {"role": "system", "content": "你是一个会正确调用工具的助手。"},
    {"role": "user", "content": "Compute 2045*6994 for me"},
]

CORRECT_CALL = (
    '<tool_call>{"name":"calculate_math","arguments":'
    '{"expression":"2045*6994"}}</tool_call>'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-path",
        default=str(Path(DEFAULT_MINIMIND_REPO) / "model"),
        help="MiniMind tokenizer directory",
    )
    return parser.parse_args()


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def measure_trajectory(tokenizer, first_output: str, results: list[str], final: str) -> dict:
    prompt_text = tokenizer.apply_chat_template(
        BASE_MESSAGES,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    prompt_ids = encode(tokenizer, prompt_text)
    first_ids = encode(tokenizer, first_output)
    messages = BASE_MESSAGES + [{"role": "assistant", "content": first_output}]
    messages.extend({"role": "tool", "content": result} for result in results)
    observed_text = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    observed_ids = encode(tokenizer, observed_text)
    observation_ids = observed_ids[len(prompt_ids) + len(first_ids) :]
    final_ids = encode(tokenizer, final)
    return {
        "P": len(prompt_ids),
        "R1": len(first_ids),
        "O": len(observation_ids),
        "R2": len(final_ids),
        "L": len(prompt_ids) + len(first_ids) + len(observation_ids) + len(final_ids),
        "C": len(first_ids) + len(final_ids),
        "prompt_chars": len(prompt_text),
        "prompt_first_20_ids": prompt_ids[:20],
        "first_output_ids": first_ids,
        "observation_delta": tokenizer.decode(observation_ids, skip_special_tokens=False),
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    scenarios = {
        "tau1_correct": (CORRECT_CALL, ['{"result": "14302730"}'], "2045 × 6994 = 14302730。"),
        "tau2_missing_gt": (CORRECT_CALL, ['{"result": "14302730"}'], "计算完成。"),
        "tau3_unknown_tool": (
            '<tool_call>{"name":"unknown_tool","arguments":{"expression":"2045*6994"}}</tool_call>',
            ['{"error": "tool not found"}'],
            "无法完成。",
        ),
        "tau4_duplicate_call": (
            CORRECT_CALL + "\n" + CORRECT_CALL,
            ['{"result": "14302730"}', '{"result": "14302730"}'],
            "结果是 14302730。",
        ),
    }
    report = {
        "tokenizer_path": args.tokenizer_path,
        "vocab_size": len(tokenizer),
        "trajectories": {
            name: measure_trajectory(tokenizer, *scenario)
            for name, scenario in scenarios.items()
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
