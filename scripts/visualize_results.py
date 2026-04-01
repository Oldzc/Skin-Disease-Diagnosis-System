"""
可视化脚本：从 artifacts/metrics.json 生成训练曲线 + 混淆矩阵热力图。

用法：
    python scripts/visualize_results.py
    python scripts/visualize_results.py --artifacts-dir artifacts --output-dir artifacts/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize training curves and confusion matrix")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--output-dir", type=str, default="artifacts/figures")
    return parser.parse_args()


def plot_training_curves(history: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    test_top1 = [h["test_top1"] for h in history]
    test_top3 = [h["test_top3"] for h in history]
    test_f1 = [h["test_macro_f1"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training History", fontsize=14)

    # Loss
    axes[0].plot(epochs, train_loss, "b-o", label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_acc, "b-o", label="Train Acc")
    axes[1].plot(epochs, test_top1, "r-o", label="Test Top-1")
    axes[1].plot(epochs, test_top3, "g-o", label="Test Top-3")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1
    axes[2].plot(epochs, test_f1, "m-o", label="Test Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].set_title("Macro F1 Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_confusion_matrix(
    conf: list[list[int]],
    labels: list[str],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(conf, dtype=float)
    # Normalize by row (true label)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Confusion Matrix (Row-Normalized)", fontsize=13)

    # Annotate cells with raw counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = int(cm[i, j])
            if count > 0:
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(count), ha="center", va="center",
                        fontsize=6, color=color)

    plt.tight_layout()
    out_path = output_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_per_class_metrics(
    conf: list[list[int]],
    labels: list[str],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(conf, dtype=float)
    eps = 1e-9
    precisions, recalls, f1s = [], [], []
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f = 2 * p * r / (p + r + eps)
        precisions.append(round(float(p), 4))
        recalls.append(round(float(r), 4))
        f1s.append(round(float(f), 4))

    x = np.arange(len(labels))
    width = 0.28

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x - width, precisions, width, label="Precision", color="steelblue", alpha=0.8)
    ax.bar(x, recalls, width, label="Recall", color="darkorange", alpha=0.8)
    ax.bar(x + width, f1s, width, label="F1", color="green", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out_path = output_dir / "per_class_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_class_distribution(train_counts: dict[str, int], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = list(train_counts.keys())
    counts = list(train_counts.values())
    colors = ["#d62728" if c < 300 else "#1f77b4" for c in counts]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(labels, counts, color=colors, alpha=0.85)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sample Count")
    ax.set_title("Training Set Class Distribution  (red = < 300 samples)")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out_path = output_dir / "class_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = artifacts_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    label_map_path = artifacts_dir / "label_map.json"
    with label_map_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    if isinstance(labels, dict):
        labels = [labels[str(i)] for i in range(len(labels))]

    print(f"Arch: {metrics.get('arch')}  Epochs: {metrics.get('epochs')}  "
          f"Pretrained: {metrics.get('pretrained')}")
    final = metrics.get("final_eval", {})
    print(f"Final  Top-1: {final.get('top1_acc')}  "
          f"Top-3: {final.get('top3_acc')}  "
          f"Macro-F1: {final.get('macro_f1')}")

    history = metrics.get("history", [])
    if history:
        plot_training_curves(history, output_dir)

    conf = final.get("confusion_matrix")
    if conf:
        plot_confusion_matrix(conf, labels, output_dir)
        plot_per_class_metrics(conf, labels, output_dir)

    train_counts = metrics.get("train_counts", {})
    if train_counts:
        plot_class_distribution(train_counts, output_dir)

    print(f"\nAll figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
