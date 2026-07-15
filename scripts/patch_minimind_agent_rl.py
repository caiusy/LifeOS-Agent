"""Apply LifeOS reward guards and mock tools to MiniMind train_agent.py."""

import argparse
from pathlib import Path


PATCH_MARKER = "LifeOS hard guard: malformed or missing required tool calls"
ROLLOUT_MARKER = "LifeOS rollout guard: invalid calls cannot open extra turns"


OLD_REWARD_HEADER = """        valid_names = {t['function']['name'] for t in tools} if tools else set()
        tool_calls = []
        for turn_answer in turn_answers: tool_calls.extend(parse_tool_calls(turn_answer))  # 解析tool调用
        reward -= 0.5 * sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)  # 标签扣分
        # -------- 无工具调用：格式+reward奖励 --------
        if not tool_calls:
"""

NEW_REWARD_HEADER = """        valid_names = {t['function']['name'] for t in tools} if tools else set()
        gt = gt_batch[sample_idx]
        tool_calls = []
        for turn_answer in turn_answers: tool_calls.extend(parse_tool_calls(turn_answer))  # 解析tool调用

        # LifeOS hard guard: malformed or missing required tool calls must never
        # receive conversational RM/length rewards, otherwise invalid XML can win.
        tool_tag_errors = sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)
        think_tag_errors = sum(abs(turn.count('<think>') - turn.count('</think>')) for turn in turn_outputs)
        expects_tool_call = bool(valid_names and gt)
        if tool_tag_errors or think_tag_errors or (expects_tool_call and not tool_calls):
            rewards[idx] = -3.0
            continue

        # -------- 无工具调用：格式+reward奖励 --------
        if not tool_calls:
"""

OLD_TOOL_BRANCH = """        else:
            gt = gt_batch[sample_idx]
            valid_call_count = 0
"""

NEW_TOOL_BRANCH = """        else:
            valid_call_count = 0
"""

OLD_AFTER_VALIDATION = """                check = CHECK_ARGS.get(name)
                valid_call_count += int(bool(name in valid_names and check and check(raw)))
            tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)  # tool数差值
"""

NEW_AFTER_VALIDATION = """                check = CHECK_ARGS.get(name)
                valid_call_count += int(bool(name in valid_names and check and check(raw)))
            if valid_call_count != len(tool_calls):
                rewards[idx] = -3.0
                continue
            tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)  # tool数差值
"""

MOCK_INSERT = """
# LifeOS deployment tools used by the dedicated Agent RL corpus.
MOCK_RESULTS.update({
    "list_today_tasks": lambda args: {"tasks": ["整理 Tool Calling 笔记", "复习 SFTDataset", "跑通 LifeOS-Agent v0.1"]},
    "search_fake_obsidian": lambda args: {"results": [{"title": str(args.get("query", "笔记")), "content": "LifeOS fake note result"}]},
})

"""

CHECK_INSERT = """
CHECK_ARGS.update({
    "list_today_tasks": lambda a: isinstance(a, dict),
    "search_fake_obsidian": lambda a: bool(a.get("query")),
})

"""

OLD_ROLLOUT_CALL_CHECK = """        calls = parse_tool_calls(new_text)
        if not calls:
            break
        unfinished = turn == max_turns - 1
"""

NEW_ROLLOUT_CALL_CHECK = """        calls = parse_tool_calls(new_text)
        # LifeOS rollout guard: invalid calls cannot open extra turns. Reward
        # still rejects them, but generation should not waste two more rounds.
        valid_names = {t['function']['name'] for t in tools} if tools else set()
        if not calls or not valid_names or any(call.get("name", "") not in valid_names for call in calls):
            break
        unfinished = turn == max_turns - 1
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one {label} block, found {text.count(old)}")
    return text.replace(old, new, 1)


def patch_text(text: str) -> tuple[str, bool]:
    changed = False
    if PATCH_MARKER not in text:
        text = replace_once(text, OLD_REWARD_HEADER, NEW_REWARD_HEADER, "reward header")
        text = replace_once(text, OLD_TOOL_BRANCH, NEW_TOOL_BRANCH, "tool branch")
        text = replace_once(text, OLD_AFTER_VALIDATION, NEW_AFTER_VALIDATION, "tool validation")
        text = replace_once(text, "# ======== 参数校验 ========\n", MOCK_INSERT + "# ======== 参数校验 ========\n", "mock insertion")
        text = replace_once(text, "# ======== 工具调用解析与执行 ========\n", CHECK_INSERT + "# ======== 工具调用解析与执行 ========\n", "argument check insertion")
        changed = True
    if ROLLOUT_MARKER not in text:
        text = replace_once(text, OLD_ROLLOUT_CALL_CHECK, NEW_ROLLOUT_CALL_CHECK, "rollout call guard")
        changed = True
    return text, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = args.path.read_text(encoding="utf-8")
    patched, changed = patch_text(source)
    if args.check:
        print("PATCHED" if not changed else "NEEDS_PATCH")
        raise SystemExit(0 if not changed else 1)
    if changed:
        args.path.write_text(patched, encoding="utf-8")
        print(f"PATCHED: {args.path}")
    else:
        print(f"ALREADY_PATCHED: {args.path}")


if __name__ == "__main__":
    main()
