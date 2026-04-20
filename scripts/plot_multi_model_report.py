from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

PREFERRED_ORDER = ["mobilenet_v3_small", "resnet18", "efficientnet_b0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multi-model comparison figures from artifacts.")
    parser.add_argument("--multi-artifacts-dir", type=str, default="artifacts/multi_model_compare")
    parser.add_argument("--output-dir", type=str, default="artifacts/multi_model_compare/figures")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _order_key(arch: str) -> tuple[int, str]:
    if arch in PREFERRED_ORDER:
        return (PREFERRED_ORDER.index(arch), arch)
    return (len(PREFERRED_ORDER), arch)


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        metrics = _load_json(metrics_path)
        final_eval = metrics.get("final_eval", {})

        arch = str(metrics.get("arch") or model_dir.name)
        row: dict[str, Any] = {
            "arch": arch,
            "top1": float(final_eval.get("top1_acc", 0.0)),
            "top3": float(final_eval.get("top3_acc", 0.0)),
            "macro_f1": float(final_eval.get("macro_f1", 0.0)),
            "model_params": int(metrics.get("model_params", 0)),
            "model_size_mb": float(metrics.get("model_size_mb", 0.0)),
            "inference_ms_per_image": float(metrics.get("inference_ms_per_image", 0.0)),
            "train_seconds": float(metrics.get("train_seconds", 0.0)),
            "best_epoch": int(metrics.get("best_epoch", 0)),
            "best_macro_f1": float(metrics.get("best_macro_f1", 0.0)),
            "metrics_path": str(metrics_path.resolve()),
        }

        eval_report_path = model_dir / "local_eval_report.json"
        if eval_report_path.exists():
            eval_report = _load_json(eval_report_path)
            for item in eval_report.get("results", []):
                method = str(item.get("method", ""))
                if method == "image_text_fusion":
                    row["fusion_top1"] = float(item.get("top1", 0.0))
                    row["fusion_top3"] = float(item.get("top3", 0.0))
                    row["fusion_macro_f1"] = float(item.get("macro_f1", 0.0))
                    break

        rows.append(row)

    rows.sort(key=lambda x: _order_key(str(x["arch"])))
    return rows


def save_summary(rows: list[dict[str, Any]], root: Path) -> tuple[Path, Path]:
    summary_json = root / "compare_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({"results": rows}, f, ensure_ascii=False, indent=2)

    summary_csv = root / "compare_summary.csv"
    fields = [
        "arch",
        "top1",
        "top3",
        "macro_f1",
        "model_params",
        "model_size_mb",
        "inference_ms_per_image",
        "train_seconds",
        "best_epoch",
        "best_macro_f1",
        "fusion_top1",
        "fusion_top3",
        "fusion_macro_f1",
        "metrics_path",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return summary_json, summary_csv


def plot_accuracy(rows: list[dict[str, Any]], out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    arches = [str(r["arch"]) for r in rows]
    top1 = [float(r["top1"]) for r in rows]
    top3 = [float(r["top3"]) for r in rows]
    macro = [float(r["macro_f1"]) for r in rows]

    x = np.arange(len(arches))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(x - width, top1, width, label="Top-1", color="#1f77b4")
    ax.bar(x, top3, width, label="Top-3", color="#2ca02c")
    ax.bar(x + width, macro, width, label="Macro-F1", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(arches)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Accuracy Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for xi, v in zip(x - width, top1):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x, top3):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + width, macro):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = out_dir / "compare_accuracy.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_efficiency(rows: list[dict[str, Any]], out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    arches = [str(r["arch"]) for r in rows]
    sizes = [float(r["model_size_mb"]) for r in rows]
    lats = [float(r["inference_ms_per_image"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    axes[0].bar(arches, sizes, color="#17becf")
    axes[0].set_title("Model Size")
    axes[0].set_ylabel("MB")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(sizes):
        axes[0].text(i, v + max(sizes) * 0.02 if max(sizes) > 0 else 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(arches, lats, color="#9467bd")
    axes[1].set_title("Inference Latency")
    axes[1].set_ylabel("ms / image")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(lats):
        axes[1].text(i, v + max(lats) * 0.02 if max(lats) > 0 else 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = out_dir / "compare_efficiency.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_tradeoff(rows: list[dict[str, Any]], out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    arches = [str(r["arch"]) for r in rows]
    top1 = [float(r["top1"]) for r in rows]
    lats = [float(r["inference_ms_per_image"]) for r in rows]
    sizes = [float(r["model_size_mb"]) for r in rows]

    bubble_scale = 80.0
    areas = [max(s, 0.1) * bubble_scale for s in sizes]

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.scatter(lats, top1, s=areas, alpha=0.65, color="#1f77b4", edgecolors="black")
    for x, y, name in zip(lats, top1, arches):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 6), fontsize=9)

    ax.set_xlabel("Inference Latency (ms / image)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Accuracy-Latency Trade-off (bubble size = model MB)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    out = out_dir / "compare_tradeoff.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()

    try:
        import matplotlib  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("matplotlib is required. Run: pip install matplotlib") from exc

    root = Path(args.multi_artifacts_dir)
    if not root.exists():
        raise FileNotFoundError(f"multi artifacts dir not found: {root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(root)
    if not rows:
        raise RuntimeError(f"No model results found under: {root}")

    summary_json, summary_csv = save_summary(rows, root)
    fig1 = plot_accuracy(rows, out_dir, args.dpi)
    fig2 = plot_efficiency(rows, out_dir, args.dpi)
    fig3 = plot_tradeoff(rows, out_dir, args.dpi)

    print("Loaded models:")
    for r in rows:
        print(
            f"- {r['arch']}: top1={r['top1']:.4f}, macro_f1={r['macro_f1']:.4f}, "
            f"size={r['model_size_mb']:.2f}MB, infer={r['inference_ms_per_image']:.2f}ms"
        )

    print("\nSaved:")
    print(f"- {summary_json.resolve()}")
    print(f"- {summary_csv.resolve()}")
    print(f"- {fig1.resolve()}")
    print(f"- {fig2.resolve()}")
    print(f"- {fig3.resolve()}")


if __name__ == "__main__":
    main()
