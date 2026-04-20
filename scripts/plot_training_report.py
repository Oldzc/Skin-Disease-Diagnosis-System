from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training/evaluation figures from artifacts.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--output-dir", type=str, default="artifacts/figures")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_curves(metrics: dict, output_dir: Path, dpi: int) -> Path | None:
    import matplotlib.pyplot as plt

    history = metrics.get("history", [])
    if not history:
        return None

    epochs = [int(h.get("epoch", i + 1)) for i, h in enumerate(history)]
    train_loss = [float(h.get("train_loss", 0.0)) for h in history]
    train_acc = [float(h.get("train_acc", 0.0)) for h in history]
    test_top1 = [float(h.get("test_top1", 0.0)) for h in history]
    test_top3 = [float(h.get("test_top3", 0.0)) for h in history]
    test_macro_f1 = [float(h.get("test_macro_f1", 0.0)) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    axes[0].plot(epochs, train_loss, marker="o", linewidth=2, color="#1f77b4")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_acc, marker="o", linewidth=2, label="train_acc", color="#2ca02c")
    axes[1].plot(epochs, test_top1, marker="o", linewidth=2, label="test_top1", color="#ff7f0e")
    axes[1].plot(epochs, test_top3, marker="o", linewidth=2, label="test_top3", color="#9467bd")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    best_epoch = int(metrics.get("best_epoch", epochs[int(np.argmax(test_macro_f1))]))
    best_f1 = float(metrics.get("best_macro_f1", max(test_macro_f1)))
    axes[2].plot(epochs, test_macro_f1, marker="o", linewidth=2, color="#d62728")
    axes[2].axvline(best_epoch, linestyle="--", color="gray", alpha=0.8)
    axes[2].scatter([best_epoch], [best_f1], color="red", zorder=3)
    axes[2].set_title("Macro-F1 Curve")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(alpha=0.3)

    fig.suptitle("Training Performance Overview", fontsize=14)
    plt.tight_layout()
    out = output_dir / "training_curves.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_method_comparison(eval_report: dict, output_dir: Path, dpi: int) -> Path | None:
    import matplotlib.pyplot as plt

    rows = eval_report.get("results", [])
    if not rows:
        return None

    methods = [r.get("method", "unknown") for r in rows]
    top1_vals = [float(r.get("top1", 0.0)) for r in rows]
    top3_vals = [float(r.get("top3", 0.0)) for r in rows]
    macro_vals = [float(r.get("macro_f1", 0.0)) for r in rows]

    x = np.arange(len(methods))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - width, top1_vals, width, label="Top-1", color="#1f77b4")
    ax.bar(x, top3_vals, width, label="Top-3", color="#2ca02c")
    ax.bar(x + width, macro_vals, width, label="Macro-F1", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Method Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for xi, v in zip(x - width, top1_vals):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x, top3_vals):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + width, macro_vals):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = output_dir / "method_comparison.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_per_class_recall(metrics: dict, output_dir: Path, dpi: int) -> Path | None:
    import matplotlib.pyplot as plt

    per_class_recall = metrics.get("per_class_recall", {})
    if not per_class_recall:
        per_class_recall = metrics.get("final_eval", {}).get("per_class_recall", {})
    if not per_class_recall:
        return None

    sorted_items = sorted(per_class_recall.items(), key=lambda x: float(x[1]), reverse=True)
    labels = [k for k, _ in sorted_items]
    values = [float(v) for _, v in sorted_items]

    fig_h = max(6, len(labels) * 0.32)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#17becf")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_title("Per-Class Recall (sorted)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "per_class_recall.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()

    try:
        import matplotlib  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("matplotlib is required. Run: pip install matplotlib") from exc

    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load_json(artifacts_dir / "metrics.json")

    created: list[Path] = []
    p = plot_training_curves(metrics, output_dir, dpi=args.dpi)
    if p:
        created.append(p)

    p = plot_per_class_recall(metrics, output_dir, dpi=args.dpi)
    if p:
        created.append(p)

    eval_path = artifacts_dir / "local_eval_report.json"
    if eval_path.exists():
        eval_report = _load_json(eval_path)
        p = plot_method_comparison(eval_report, output_dir, dpi=args.dpi)
        if p:
            created.append(p)
    else:
        print(f"Skip method comparison: {eval_path} not found")

    if not created:
        print("No figures created. Check metrics/history content.")
        return

    print("Created figures:")
    for fp in created:
        print(f"- {fp.resolve()}")


if __name__ == "__main__":
    main()
