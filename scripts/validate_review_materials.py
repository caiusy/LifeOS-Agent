"""Validate that review-only conversations cannot be mistaken for training data."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REVIEW_ROOT = ROOT / "review_materials"
TRAINING_CODE = [ROOT / "scripts", ROOT / "lifeos_agent"]


def main() -> None:
    errors = []
    records = 0
    for path in sorted(REVIEW_ROOT.rglob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            records += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
                continue
            if item.get("use_for_training") is not False:
                errors.append(f"{path}:{line_number}: use_for_training must be false")
            if item.get("status") != "review_only":
                errors.append(f"{path}:{line_number}: status must be review_only")

    for root in TRAINING_CODE:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".sh"} or path.name == Path(__file__).name:
                continue
            if "review_materials" in path.read_text(encoding="utf-8", errors="replace"):
                errors.append(f"training/runtime code references review_materials: {path}")

    if not records:
        errors.append("no review JSONL records found")
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        raise SystemExit(1)
    print(f"PASS: {records} review-only records; no training/runtime references")


if __name__ == "__main__":
    main()
