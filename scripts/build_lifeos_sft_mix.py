import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lifeos_agent.tools import get_tools_by_names


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_conversation_message(message: dict) -> dict:
    normalized = dict(message)
    normalized.setdefault("content", "")
    normalized.setdefault("reasoning_content", "")
    normalized.setdefault("tools", "")
    normalized.setdefault("tool_calls", "")

    for key in ("tools", "tool_calls"):
        value = normalized.get(key, "")
        if isinstance(value, (list, dict)):
            normalized[key] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            normalized[key] = ""

    if normalized.get("reasoning_content") is None:
        normalized["reasoning_content"] = ""
    if normalized.get("content") is None:
        normalized["content"] = ""
    return normalized


def extract_tool_names(row: dict) -> list[str]:
    names = []
    for message in row.get("conversations", []):
        tool_calls = message.get("tool_calls", "")
        if not tool_calls:
            continue
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except Exception:
                continue
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        if isinstance(tool_calls, list):
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else None
                if name and name not in names:
                    names.append(name)
    return names


def normalize_row(row: dict) -> dict:
    normalized = dict(row)
    normalized["conversations"] = [
        normalize_conversation_message(message)
        for message in row.get("conversations", [])
    ]
    tool_names = extract_tool_names(normalized)
    if tool_names:
        tools = get_tools_by_names(tool_names)
        if tools:
            for message in normalized["conversations"]:
                if message.get("role") == "system":
                    if not message.get("tools"):
                        message["tools"] = json.dumps(tools, ensure_ascii=False)
                    break
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Build a mixed SFT dataset for LifeOS-Agent")
    parser.add_argument(
        "--official",
        type=Path,
        default=Path("dataset/minimind_dataset/sft_t2t_mini.jsonl"),
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("dataset/lifeos_sft_seed.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/lifeos_sft_mixed.jsonl"),
    )
    parser.add_argument(
        "--seed_repeat",
        type=int,
        default=200,
        help="Repeat seed rows to increase their influence in early experiments",
    )
    parser.add_argument(
        "--official_limit",
        type=int,
        default=20000,
        help="Use only the first N official rows for a fast first experiment; 0 means all rows",
    )
    parser.add_argument("--shuffle_seed", type=int, default=42)
    args = parser.parse_args()

    official_rows = [normalize_row(row) for row in load_jsonl(args.official)]
    seed_rows = [normalize_row(row) for row in load_jsonl(args.seed)]

    if args.official_limit > 0:
        official_rows = official_rows[: args.official_limit]

    mixed_rows = official_rows + seed_rows * args.seed_repeat
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(mixed_rows)

    dump_jsonl(args.output, mixed_rows)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "official_rows": len(official_rows),
                "seed_rows": len(seed_rows),
                "seed_repeat": args.seed_repeat,
                "mixed_rows": len(mixed_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
