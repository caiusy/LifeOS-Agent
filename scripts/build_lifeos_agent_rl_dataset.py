"""Build a small LifeOS-specific Agent RL corpus from reviewed SFT seeds."""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_lifeos_sft_mix import extract_tool_names
from lifeos_agent.tools import get_tools_by_names


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def expected_gt(conversations: list[dict], tool_name: str) -> list[str]:
    tool_message = next((m for m in conversations if m.get("role") == "tool"), None)
    result = parse_json(tool_message.get("content")) if tool_message else None
    if not isinstance(result, dict):
        return []
    if tool_name == "calculate_math" and "result" in result:
        return [str(result["result"])]
    if tool_name == "list_today_tasks" and result.get("tasks"):
        return [str(result["tasks"][-1])]
    if tool_name == "search_fake_obsidian" and result.get("results"):
        return [str(result["results"][0]["title"])]
    return []


def convert_row(row: dict) -> tuple[dict, bool]:
    conversations = row.get("conversations", [])
    system = next((dict(m) for m in conversations if m.get("role") == "system"), None)
    user = next((dict(m) for m in conversations if m.get("role") == "user"), None)
    if system is None or user is None:
        raise ValueError("each seed row needs system and user messages")

    names = extract_tool_names(row)
    tools = get_tools_by_names(names)
    system["content"] = system.get("content", "")
    if tools:
        system["tools"] = json.dumps(tools, ensure_ascii=False)
    else:
        system.pop("tools", None)

    gt = expected_gt(conversations, names[0]) if names else []
    if names and not gt:
        raise ValueError(f"cannot derive one GT value for tool sample: {names}")
    prompt_messages = [system, {"role": "user", "content": user.get("content", "")}]
    prompt_messages.append({"role": "assistant", "content": ""})
    return {"conversations": prompt_messages, "gt": gt}, bool(names)


def build_rows(seed_rows: list[dict], tool_repeat: int, no_tool_repeat: int, seed: int) -> list[dict]:
    rows = []
    for row in seed_rows:
        converted, uses_tool = convert_row(row)
        rows.extend([converted] * (tool_repeat if uses_tool else no_tool_repeat))
    random.Random(seed).shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("dataset/lifeos_sft_seed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("dataset/lifeos_agent_rl.jsonl"))
    parser.add_argument("--tool_repeat", type=int, default=10)
    parser.add_argument("--no_tool_repeat", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_rows(load_jsonl(args.input), args.tool_repeat, args.no_tool_repeat, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    tool_rows = sum(bool(row["gt"]) for row in rows)
    print(json.dumps({"output": str(args.output), "rows": len(rows), "tool_rows": tool_rows,
                      "no_tool_rows": len(rows) - tool_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
