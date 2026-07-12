"""Validate the evaluation corpus and run deterministic non-model baselines."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lifeos_agent.main import parse_tool_calls
from lifeos_agent.router import select_tool_names
from lifeos_agent.tools import TOOL_MAP, execute_tool


def load_cases(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def expected_tools(item):
    value = item["expected_tool"]
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def validate(cases):
    errors = []
    ids = [item.get("id") for item in cases]
    if len(cases) != 100:
        errors.append(f"expected 100 cases, got {len(cases)}")
    if len(set(ids)) != len(ids):
        errors.append("case ids are not unique")

    router_pass = parser_pass = executor_pass = 0
    router_total = parser_total = executor_total = 0
    for item in cases:
        case_id = item.get("id", "unknown")
        for name in expected_tools(item):
            if name not in TOOL_MAP:
                errors.append(f"{case_id}: unknown expected tool {name}")

        if item["category"] not in {"parser_robustness", "tool_error"}:
            router_total += 1
            actual = select_tool_names(item["user_input"])
            if actual == expected_tools(item):
                router_pass += 1
            else:
                errors.append(f"{case_id}: router expected {expected_tools(item)}, got {actual}")

        if item["category"] == "parser_robustness":
            parser_total += 1
            actual_count = len(parse_tool_calls(item["model_output_fixture"]))
            if actual_count == item["expected_parsed_call_count"]:
                parser_pass += 1
            else:
                errors.append(f"{case_id}: parser expected {item['expected_parsed_call_count']}, got {actual_count}")

        if item["category"] in {"math", "today_tasks", "tool_error"}:
            executor_total += 1
            result = execute_tool(item["expected_tool"], item["expected_arguments"])
            expected = item["expected_tool_result"]
            if "error_contains" in expected:
                passed = expected["error_contains"] in result.get("error", "")
            else:
                passed = result == expected
            if passed:
                executor_pass += 1
            else:
                errors.append(f"{case_id}: executor expected {expected}, got {result}")

    return errors, {
        "cases": len(cases),
        "categories": dict(Counter(item["category"] for item in cases)),
        "router": f"{router_pass}/{router_total}",
        "parser": f"{parser_pass}/{parser_total}",
        "executor": f"{executor_pass}/{executor_total}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval" / "lifeos_eval.jsonl")
    args = parser.parse_args()
    errors, summary = validate(load_cases(args.dataset))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        print("\nValidation errors:")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    print("\nPASS: dataset and deterministic baselines are valid")


if __name__ == "__main__":
    main()
