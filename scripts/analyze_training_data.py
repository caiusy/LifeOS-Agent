"""Stream MiniMind datasets and report shape/loss-relevant statistics.

The scanner never loads a whole JSONL file into memory. It validates every
line while retaining only a bounded reservoir sample for tokenization and
distribution analysis.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer


DATASETS = {
    "pretrain": "dataset/minimind_dataset/pretrain_t2t_mini.jsonl",
    "sft": "dataset/minimind_dataset/sft_t2t_mini.jsonl",
    "dpo": "dataset/minimind_dataset/dpo.jsonl",
    "rlaif": "dataset/minimind_dataset/rlaif.jsonl",
    "agent_rl": "dataset/minimind_dataset/agent_rl.jsonl",
    "lifeos_seed": "dataset/lifeos_sft_seed.jsonl",
}

MAX_LENGTHS = {
    "pretrain": 512,
    "sft": 768,
    "dpo": 768,
    "rlaif": 768,
    "agent_rl": 768,
    "lifeos_seed": 768,
}


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def describe(values: list[int]) -> dict:
    return {
        "min": min(values, default=0),
        "mean": round(mean(values), 2) if values else 0,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0),
    }


def message_text(message: dict) -> str:
    fields = [message.get("content", ""), message.get("reasoning_content", "")]
    for key in ("tools", "tool_calls"):
        value = message.get(key, "")
        if value:
            fields.append(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
    return "\n".join(str(value) for value in fields if value)


def row_texts(name: str, row: dict) -> list[str]:
    if name == "pretrain":
        return [str(row.get("text", ""))]
    if name == "dpo":
        return [
            "\n".join(message_text(m) for m in row.get(branch, []) if isinstance(m, dict))
            for branch in ("chosen", "rejected")
        ]
    return [
        "\n".join(message_text(m) for m in row.get("conversations", []) if isinstance(m, dict))
    ]


def required_fields(name: str) -> tuple[str, ...]:
    if name == "pretrain":
        return ("text",)
    if name == "dpo":
        return ("chosen", "rejected")
    return ("conversations",)


def scan_file(name: str, path: Path, sample_size: int, seed: int) -> tuple[dict, list[dict]]:
    rng = random.Random(seed)
    reservoir: list[dict] = []
    rows = malformed = missing = empty_text = 0
    role_counts: Counter[str] = Counter()
    tool_rows = gt_rows = 0
    sample_hashes: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            rows += 1
            if any(field not in row or row[field] in (None, "", []) for field in required_fields(name)):
                missing += 1

            texts = row_texts(name, row)
            if not any(text.strip() for text in texts):
                empty_text += 1

            messages = row.get("conversations", [])
            if name == "dpo":
                messages = row.get("chosen", []) + row.get("rejected", [])
            for message in messages:
                if isinstance(message, dict):
                    role_counts[str(message.get("role", "missing"))] += 1
            if any(isinstance(m, dict) and (m.get("tools") or m.get("tool_calls") or m.get("role") == "tool") for m in messages):
                tool_rows += 1
            if row.get("gt"):
                gt_rows += 1

            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                replacement = rng.randrange(rows)
                if replacement < sample_size:
                    reservoir[replacement] = row

    for row in reservoir:
        digest = hashlib.sha1(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        sample_hashes[digest] += 1

    metadata = {
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "rows": rows,
        "malformed_json": malformed,
        "missing_required": missing,
        "empty_text": empty_text,
        "sample_rows": len(reservoir),
        "sample_duplicate_rows": sum(count - 1 for count in sample_hashes.values()),
        "tool_rows": tool_rows,
        "tool_row_rate": round(tool_rows / rows, 6) if rows else 0,
        "gt_rows": gt_rows,
        "gt_row_rate": round(gt_rows / rows, 6) if rows else 0,
        "role_counts": dict(role_counts),
    }
    return metadata, reservoir


def calculate_lengths(name: str, metadata: dict, rows: list[dict], tokenizer) -> None:
    texts = [text for row in rows for text in row_texts(name, row)]
    char_lengths = [len(text) for text in texts]
    token_lengths = [len(tokenizer(text, add_special_tokens=False).input_ids) for text in texts]
    max_length = MAX_LENGTHS[name]
    metadata["sample_sequences"] = len(texts)
    metadata["char_length"] = describe(char_lengths)
    metadata["token_length"] = describe(token_lengths)
    metadata["configured_max_length"] = max_length
    metadata["estimated_truncation_rate"] = round(
        sum(length > max_length for length in token_lengths) / len(token_lengths), 6
    ) if token_lengths else 0


def write_svg(results: dict, output: Path) -> None:
    names = list(results)
    width, height = 1180, 560
    chart_left, chart_top, chart_height = 110, 120, 330
    max_value = max(result["token_length"]["p95"] for result in results.values()) or 1
    bar_width = 105
    gap = 65
    colors = ["#c76d3b", "#d99b45", "#3f7f6b", "#5b6f92", "#8b6f9c", "#40798c"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="24" fill="#f6f1e8"/>',
        '<text x="60" y="58" font-family="Georgia,serif" font-size="30" font-weight="700" fill="#17332b">Sample token-length distribution</text>',
        '<text x="60" y="88" font-family="Arial,sans-serif" font-size="15" fill="#586b64">Bar = P95 content tokens; line = configured training maximum. DPO contains chosen and rejected sequences.</text>',
        f'<line x1="{chart_left}" y1="{chart_top + chart_height}" x2="{width-50}" y2="{chart_top + chart_height}" stroke="#879890"/>',
    ]
    for index, name in enumerate(names):
        x = chart_left + index * (bar_width + gap)
        value = results[name]["token_length"]["p95"]
        bar_height = max(2, value / max_value * chart_height)
        y = chart_top + chart_height - bar_height
        parts.extend([
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="10" fill="{colors[index % len(colors)]}"/>',
            f'<text x="{x + bar_width/2}" y="{y - 10:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" font-weight="700" fill="#253f36">{value}</text>',
            f'<text x="{x + bar_width/2}" y="{chart_top + chart_height + 30}" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#253f36">{name}</text>',
            f'<text x="{x + bar_width/2}" y="{chart_top + chart_height + 53}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#687871">max={results[name]["configured_max_length"]}</text>',
        ])
    parts.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream and analyze MiniMind training datasets")
    parser.add_argument("--tokenizer", type=Path, default=Path("models/minimind-3"))
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("analysis/training_data_analysis.json"))
    parser.add_argument("--svg", type=Path, default=Path("docs/assets/training_data_token_lengths.svg"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    results = {}
    for name, relative_path in DATASETS.items():
        path = Path(relative_path)
        if not path.exists():
            continue
        metadata, sample = scan_file(name, path, args.sample_size, args.seed)
        calculate_lengths(name, metadata, sample, tokenizer)
        results[name] = metadata
        print(f"{name}: rows={metadata['rows']}, token_p95={metadata['token_length']['p95']}, truncation={metadata['estimated_truncation_rate']:.2%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_svg(results, args.svg)


if __name__ == "__main__":
    main()
