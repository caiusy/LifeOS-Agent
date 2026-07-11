import argparse
import json
from pathlib import Path


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                json.loads(line)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Inspect local LifeOS / MiniMind datasets")
    parser.add_argument("--dataset_root", type=Path, default=Path("dataset"))
    args = parser.parse_args()

    official_root = args.dataset_root / "minimind_dataset"
    targets = [
        official_root / "sft_t2t_mini.jsonl",
        official_root / "pretrain_t2t_mini.jsonl",
        official_root / "dpo.jsonl",
        official_root / "agent_rl.jsonl",
        args.dataset_root / "lifeos_sft_seed.jsonl",
    ]

    print("Dataset inspection")
    for path in targets:
        exists = path.exists()
        size_mb = round(path.stat().st_size / 1024 / 1024, 2) if exists else 0
        records = count_jsonl(path) if exists and path.suffix == ".jsonl" else 0
        print(
            json.dumps(
                {
                    "path": str(path),
                    "exists": exists,
                    "size_mb": size_mb,
                    "records": records,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
