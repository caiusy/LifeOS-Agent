"""Parse MiniMind DPO/PPO logs and render dependency-free SVG curves."""

import argparse
import json
import re
from pathlib import Path
from statistics import mean


DPO_PATTERN = re.compile(r"\((\d+)/(\d+)\), loss: ([\d.]+), dpo_loss: ([\d.]+).+learning_rate: ([\d.]+)")
PPO_PATTERN = re.compile(
    r"\((\d+)/(\d+)\), Reward: ([-\d.]+), KL_ref: ([-\d.]+), Approx KL: ([-\d.]+), "
    r"ClipFrac: ([-\d.]+), Critic Loss: ([-\d.]+), Avg Response Len: ([-\d.]+)"
)


def parse_dpo(path: Path) -> list[dict]:
    points = []
    for match in DPO_PATTERN.finditer(path.read_text(encoding="utf-8", errors="replace")):
        step, total, loss, dpo_loss, lr = match.groups()
        points.append({"step": int(step), "total": int(total), "loss": float(loss), "dpo_loss": float(dpo_loss), "lr": float(lr)})
    return points


def parse_ppo(path: Path) -> list[dict]:
    points = []
    for match in PPO_PATTERN.finditer(path.read_text(encoding="utf-8", errors="replace")):
        step, total, reward, kl_ref, approx_kl, clipfrac, critic_loss, avg_len = match.groups()
        points.append({
            "step": int(step), "total": int(total), "reward": float(reward),
            "kl_ref": float(kl_ref), "approx_kl": float(approx_kl),
            "clipfrac": float(clipfrac), "critic_loss": float(critic_loss),
            "avg_response_len": float(avg_len),
        })
    return points


def summarize(points: list[dict], fields: tuple[str, ...]) -> dict:
    if not points:
        return {"points": 0}
    result = {"points": len(points), "last_step": points[-1]["step"], "total_steps": points[-1]["total"]}
    for field in fields:
        values = [point[field] for point in points]
        tail = values[-min(100, len(values)):]
        result[field] = {
            "first": values[0], "last": values[-1],
            "mean": round(mean(values), 6), "last_100_mean": round(mean(tail), 6),
            "min": min(values), "max": max(values),
        }
    return result


def polyline(points: list[dict], field: str, x: int, y: int, width: int, height: int) -> str:
    values = [point[field] for point in points]
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    span = high - low or 1
    coords = []
    for index, value in enumerate(values):
        px = x + index / (len(values) - 1) * width
        py = y + height - (value - low) / span * height
        coords.append(f"{px:.1f},{py:.1f}")
    return " ".join(coords)


def write_svg(dpo: list[dict], ppo: list[dict], output: Path) -> None:
    width, height = 1280, 700
    panels = [
        ("DPO loss", dpo, "dpo_loss", 60, 130, "#c76d3b"),
        ("PPO reward (running snapshot)", ppo, "reward", 650, 130, "#3f7f6b"),
        ("PPO critic loss", ppo, "critic_loss", 60, 420, "#5b6f92"),
        ("PPO KL to reference", ppo, "kl_ref", 650, 420, "#8b6f9c"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="24" fill="#f6f1e8"/>',
        '<text x="60" y="58" font-family="Georgia,serif" font-size="30" font-weight="700" fill="#17332b">LifeOS-Agent training metrics</text>',
        '<text x="60" y="88" font-family="Arial,sans-serif" font-size="15" fill="#586b64">DPO is complete; PPO panels are a live snapshot and are not convergence claims.</text>',
    ]
    for title, points, field, x, y, color in panels:
        panel_width, panel_height = 540, 190
        parts.extend([
            f'<rect x="{x}" y="{y}" width="{panel_width}" height="{panel_height}" rx="16" fill="#fff" fill-opacity=".72" stroke="#a7b6af"/>',
            f'<text x="{x+20}" y="{y+32}" font-family="Arial,sans-serif" font-size="18" font-weight="700" fill="#253f36">{title}</text>',
            f'<polyline points="{polyline(points, field, x+20, y+50, panel_width-40, panel_height-70)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>',
        ])
        if points:
            parts.append(f'<text x="{x+panel_width-20}" y="{y+32}" text-anchor="end" font-family="Arial,sans-serif" font-size="13" fill="#687871">step {points[-1]["step"]}/{points[-1]["total"]}</text>')
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo", type=Path, default=Path("analysis/logs/lifeos_agent_dpo_v1.log"))
    parser.add_argument("--ppo", type=Path, default=Path("analysis/logs/lifeos_agent_ppo_v1.log"))
    parser.add_argument("--output", type=Path, default=Path("analysis/training_log_analysis.json"))
    parser.add_argument("--svg", type=Path, default=Path("docs/assets/training_metrics.svg"))
    args = parser.parse_args()
    dpo, ppo = parse_dpo(args.dpo), parse_ppo(args.ppo)
    report = {
        "dpo": summarize(dpo, ("loss", "dpo_loss", "lr")),
        "ppo": summarize(ppo, ("reward", "kl_ref", "approx_kl", "clipfrac", "critic_loss", "avg_response_len")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_svg(dpo, ppo, args.svg)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
