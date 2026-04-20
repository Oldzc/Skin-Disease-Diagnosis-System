from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PREFERRED_ORDER = ["mobilenet_v3_small", "resnet18", "efficientnet_b0"]
METRICS = [
    ("train_loss", "Training Loss over Epochs", "Loss"),
    ("train_acc", "Training Accuracy over Epochs", "Accuracy"),
    ("test_top1", "Validation Top-1 Accuracy over Epochs", "Top-1 Accuracy"),
    ("test_macro_f1", "Validation Macro-F1 over Epochs", "Macro-F1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-epoch training curves for all compared models.")
    parser.add_argument("--multi-artifacts-dir", type=str, default="artifacts/multi_model_compare")
    parser.add_argument("--output-dir", type=str, default="artifacts/figures")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _order_key(arch: str) -> tuple[int, str]:
    if arch in PREFERRED_ORDER:
        return (PREFERRED_ORDER.index(arch), arch)
    return (len(PREFERRED_ORDER), arch)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_histories(root: Path) -> dict[str, list[dict[str, Any]]]:
    histories: dict[str, list[dict[str, Any]]] = {}
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        data = _load_json(metrics_path)
        arch = str(data.get("arch") or model_dir.name)
        history = data.get("history", [])
        if not isinstance(history, list) or not history:
            continue
        history = sorted(history, key=lambda r: int(r.get("epoch", 0)))
        histories[arch] = history
    return dict(sorted(histories.items(), key=lambda kv: _order_key(kv[0])))


def plot_combined(histories: dict[str, list[dict[str, Any]]], out_dir: Path, dpi: int) -> Path:
    out = out_dir / "multi_model_training_curves.png"
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    axes = axes.flatten()

    for ax, (metric_key, metric_name, y_label) in zip(axes, METRICS):
        has_line = False
        for arch, hist in histories.items():
            epochs = [int(x.get("epoch", 0)) for x in hist]
            values = [x.get(metric_key) for x in hist]
            if not any(v is not None for v in values):
                continue
            vals = [float(v) if v is not None else np.nan for v in values]
            ax.plot(epochs, vals, marker="o", linewidth=1.8, markersize=3.8, label=arch)
            has_line = True

        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        if "loss" in metric_key:
            ax.set_ylabel(y_label)
        else:
            ax.set_ylabel(y_label)
            ax.set_ylim(0.0, 1.02)
        if has_line:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("Training Process Curves of Three Backbones", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_single_metric(
    histories: dict[str, list[dict[str, Any]]],
    metric_key: str,
    metric_name: str,
    y_label: str,
    out_dir: Path,
    dpi: int,
) -> Path:
    out = out_dir / f"multi_model_{metric_key}_curve.png"
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    for arch, hist in histories.items():
        epochs = [int(x.get("epoch", 0)) for x in hist]
        values = [x.get(metric_key) for x in hist]
        if not any(v is not None for v in values):
            continue
        vals = [float(v) if v is not None else np.nan for v in values]
        ax.plot(epochs, vals, marker="o", linewidth=2.0, markersize=4.2, label=arch)

    ax.set_title(metric_name)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    if "loss" in metric_key:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(y_label)
        ax.set_ylim(0.0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.multi_artifacts_dir)
    if not root.exists():
        raise FileNotFoundError(f"multi artifacts dir not found: {root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = load_histories(root)
    if not histories:
        raise RuntimeError(f"No history found under: {root}")

    outputs: list[Path] = []
    outputs.append(plot_combined(histories, out_dir, args.dpi))
    for key, name, y_label in METRICS:
        outputs.append(plot_single_metric(histories, key, name, y_label, out_dir, args.dpi))

    print("Loaded models:")
    for arch, hist in histories.items():
        print(f"- {arch}: {len(hist)} epochs")

    print("\nSaved figures:")
    for p in outputs:
        print(f"- {p.resolve()}")


if __name__ == "__main__":
    main()
