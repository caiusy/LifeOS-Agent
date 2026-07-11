"""Show the deterministic part of LifeOS-Agent without loading a model.

This is a learning aid: it exposes routing, selected schemas, XML/JSON parsing,
tool execution, and role=tool message backfill. Model generation is the only
step represented by a fixed sample <tool_call>, so the script runs on any CPU.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lifeos_agent.main import build_messages, parse_tool_calls
from lifeos_agent.router import select_tool_names
from lifeos_agent.tools import execute_tool, get_tools_by_names


CASES = [
    (
        "我之前学 SFTDataset 学到哪了？",
        '<tool_call>{"name":"search_fake_obsidian","arguments":{"query":"SFTDataset"}}</tool_call>',
    ),
    (
        "我今天应该做什么？",
        '<tool_call>{"name":"list_today_tasks","arguments":{}}</tool_call>',
    ),
    (
        "17.66 涨停价是多少？",
        '<tool_call>{"name":"calculate_math","arguments":{"expression":"round(17.66 * 1.1, 2)"}}</tool_call>',
    ),
    ("你好，简单介绍一下你自己", ""),
]


def run_case(user_input: str, sampled_model_output: str) -> None:
    messages = build_messages(user_input)
    tool_names = select_tool_names(user_input)
    tools = get_tools_by_names(tool_names)

    print("\n" + "=" * 80)
    print(f"User input: {user_input}")
    print(f"1. Router output: {tool_names}")
    print(f"2. Injected schemas: {len(tools)}")
    if tools:
        print(json.dumps(tools, ensure_ascii=False, indent=2))
    else:
        print("   tools=None: normal chat does not receive a tool schema")

    print(f"3. Initial messages: count={len(messages)}, roles={[m['role'] for m in messages]}")
    if not sampled_model_output:
        print("4. Model path: direct answer; no tool call and no role=tool backfill")
        return

    print(f"4. Sample model output ({len(sampled_model_output)} chars):")
    print(sampled_model_output)
    calls = parse_tool_calls(sampled_model_output)
    print(f"5. Parsed calls: count={len(calls)}")

    messages.append({"role": "assistant", "content": sampled_model_output})
    for call in calls:
        result = execute_tool(call.get("name", ""), call.get("arguments", {}))
        result_text = json.dumps(result, ensure_ascii=False)
        messages.append({"role": "tool", "content": result_text})
        print(f"6. Execute {call['name']}: {result_text}")

    print(f"7. Backfilled messages: count={len(messages)}, roles={[m['role'] for m in messages]}")
    print("8. Next model turn consumes all four messages and writes the grounded final answer.")


def main() -> None:
    for case in CASES:
        run_case(*case)


if __name__ == "__main__":
    main()
