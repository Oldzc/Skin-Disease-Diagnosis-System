from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot figures for experiment suite outputs.")
    parser.add_argument("--experiments-dir", type=str, default="artifacts/experiments")
    parser.add_argument("--output-dir", type=str, default="artifacts/experiments")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _has_skipped(rows: list[dict[str, Any]]) -> bool:
    return bool(rows and rows[0].get("skipped_reason"))


def _save_skip_figure(path: Path, title: str, reason: str, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.42, f"Skipped: {reason}", ha="center", va="center", fontsize=12)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_exp2(rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    out = output_dir / "exp2_multimodal_bar.png"
    if not rows:
        _save_skip_figure(out, "Exp2 Multimodal", "No data", dpi)
        return out

    labels = [r.get("setting", "unknown") for r in rows]
    top1 = [_to_float(r.get("top1")) for r in rows]
    top3 = [_to_float(r.get("top3")) for r in rows]
    macro = [_to_float(r.get("macro_f1")) for r in rows]

    x = np.arange(len(labels))
    w = 0.24

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - w, top1, width=w, label="Top-1", color="#1f77b4")
    ax.bar(x, top3, width=w, label="Top-3", color="#2ca02c")
    ax.bar(x + w, macro, width=w, label="Macro-F1", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Exp2: Image Only vs Image+Text")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_exp3(rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    out = output_dir / "exp3_prompt_json_quality.png"
    if not rows:
        _save_skip_figure(out, "Exp3 Prompt/JSON", "No data", dpi)
        return out
    if _has_skipped(rows):
        _save_skip_figure(out, "Exp3 Prompt/JSON", rows[0].get("skipped_reason", "Skipped"), dpi)
        return out

    labels = [r.get("setting", "unknown") for r in rows]
    json_valid = [_to_float(r.get("json_valid_rate")) for r in rows]
    schema_pass = [_to_float(r.get("schema_pass_rate")) for r in rows]
    label_in_set = [_to_float(r.get("label_in_set_rate")) for r in rows]
    top3_complete = [_to_float(r.get("top3_complete_rate")) for r in rows]

    x = np.arange(len(labels))
    w = 0.18

    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.bar(x - 1.5 * w, json_valid, width=w, label="json_valid_rate", color="#1f77b4")
    ax.bar(x - 0.5 * w, schema_pass, width=w, label="schema_pass_rate", color="#2ca02c")
    ax.bar(x + 0.5 * w, label_in_set, width=w, label="label_in_set_rate", color="#9467bd")
    ax.bar(x + 1.5 * w, top3_complete, width=w, label="top3_complete_rate", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Exp3: Prompt & Output Constraint Quality")
    ax.set_ylabel("Rate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", ncol=2)

    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_exp4(rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    out = output_dir / "exp4_robustness_route.png"
    if not rows:
        _save_skip_figure(out, "Exp4 Robustness", "No data", dpi)
        return out
    if _has_skipped(rows):
        _save_skip_figure(out, "Exp4 Robustness", rows[0].get("skipped_reason", "Skipped"), dpi)
        return out

    scenarios = [r.get("scenario", "unknown") for r in rows]
    api = [_to_float(r.get("source_api")) for r in rows]
    hybrid = [_to_float(r.get("source_local_hybrid")) for r in rows]
    mock = [_to_float(r.get("source_local_mock")) for r in rows]
    failed = [_to_float(r.get("source_failed")) for r in rows]
    route_rate = [_to_float(r.get("route_correct_rate")) for r in rows]

    x = np.arange(len(scenarios))

    fig, ax1 = plt.subplots(figsize=(12.5, 5.6))
    ax1.bar(x, api, label="API", color="#1f77b4")
    ax1.bar(x, hybrid, bottom=api, label="local_hybrid", color="#2ca02c")
    ax1.bar(x, mock, bottom=np.array(api) + np.array(hybrid), label="local_mock", color="#ff7f0e")
    ax1.bar(x, failed, bottom=np.array(api) + np.array(hybrid) + np.array(mock), label="failed", color="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=10, ha="right")
    ax1.set_ylabel("Sample Count")
    ax1.set_title("Exp4: Robustness and Degradation Route")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, route_rate, color="black", marker="o", linewidth=2.0, label="route_correct_rate")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Route Correct Rate")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_exp5(rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    out = output_dir / "exp5_ablation_impact.png"
    if not rows:
        _save_skip_figure(out, "Exp5 Ablation", "No data", dpi)
        return out
    if _has_skipped(rows):
        _save_skip_figure(out, "Exp5 Ablation", rows[0].get("skipped_reason", "Skipped"), dpi)
        return out

    items = [r.get("ablation_item", "unknown") for r in rows]
    top1 = [_to_float(r.get("top1")) for r in rows]
    macro = [_to_float(r.get("macro_f1")) for r in rows]
    succ = [_to_float(r.get("success_rate")) for r in rows]
    fail = [_to_float(r.get("failure_rate")) for r in rows]

    x = np.arange(len(items))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))

    axes[0].bar(x - w / 2, top1, width=w, label="Top-1", color="#1f77b4")
    axes[0].bar(x + w / 2, macro, width=w, label="Macro-F1", color="#ff7f0e")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(items, rotation=20, ha="right")
    axes[0].set_title("Performance Impact")
    axes[0].set_ylabel("Score")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, succ, marker="o", linewidth=2, label="success_rate", color="#2ca02c")
    axes[1].plot(x, fail, marker="o", linewidth=2, label="failure_rate", color="#d62728")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(items, rotation=20, ha="right")
    axes[1].set_title("Availability Impact")
    axes[1].set_ylabel("Rate")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()

    try:
        import matplotlib  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("matplotlib is required. Run: pip install matplotlib") from exc

    exp_dir = Path(args.experiments_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp2_rows = _load_csv(exp_dir / "exp2_multimodal.csv")
    exp3_rows = _load_csv(exp_dir / "exp3_prompt_json.csv")
    exp4_rows = _load_csv(exp_dir / "exp4_robustness.csv")
    exp5_rows = _load_csv(exp_dir / "exp5_ablation.csv")

    p1 = plot_exp2(exp2_rows, out_dir, args.dpi)
    p2 = plot_exp3(exp3_rows, out_dir, args.dpi)
    p3 = plot_exp4(exp4_rows, out_dir, args.dpi)
    p4 = plot_exp5(exp5_rows, out_dir, args.dpi)

    print("Saved figures:")
    print(f"- {p1.resolve()}")
    print(f"- {p2.resolve()}")
    print(f"- {p3.resolve()}")
    print(f"- {p4.resolve()}")


if __name__ == "__main__":
    main()
