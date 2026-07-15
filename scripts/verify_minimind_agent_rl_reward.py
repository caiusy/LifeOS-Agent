"""Smoke-test the patched MiniMind Agent RL reward guards on CPU."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimind_repo", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.minimind_repo.resolve()))

    from trainer import train_agent

    search_tool = [{
        "type": "function",
        "function": {
            "name": "search_fake_obsidian",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }]

    def reward(completion, gt, tools, turns=None):
        value = train_agent.calculate_rewards(
            ["prompt"],
            [completion],
            [gt],
            [tools],
            1,
            reward_model=None,
            device="cpu",
            turn_outputs_batch=[turns or [completion]],
            unfinished_batch=[False],
        )
        return value.item()

    malformed = reward("<tool_call>not json", ["SFTDataset"], search_tool)
    missing = reward("我直接猜答案", ["SFTDataset"], search_tool)
    invalid = reward(
        '<tool_call>{"name":"unknown","arguments":{}}</tool_call>',
        ["SFTDataset"],
        search_tool,
    )
    valid = reward(
        "SFTDataset",
        ["SFTDataset"],
        search_tool,
        [
            '<tool_call>{"name":"search_fake_obsidian","arguments":{"query":"SFTDataset"}}</tool_call>',
            "SFTDataset",
        ],
    )
    no_tool = reward("你好，我可以帮助你。", [], None)
    observed = {
        "malformed_required_call": malformed,
        "missing_required_call": missing,
        "invalid_call": invalid,
        "valid_call": valid,
        "valid_no_tool_answer": no_tool,
    }
    expected = {
        "malformed_required_call": -3.0,
        "missing_required_call": -3.0,
        "invalid_call": -3.0,
        "valid_call": 3.0,
        "valid_no_tool_answer": 0.5,
    }
    print(observed)
    if observed != expected:
        raise SystemExit(f"reward guard verification failed; expected {expected}")
    print("PASS: malformed/missing/invalid calls are rejected; valid tool and no-tool paths remain trainable")


if __name__ == "__main__":
    main()
