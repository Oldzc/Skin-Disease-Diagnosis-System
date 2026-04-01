"""
消融实验脚本：系统性对比不同配置下的推理性能。

实验组：
  1. local_mock          — 纯规则兜底（基线）
  2. image_only          — 仅图像模型（α=1.0, β=0, γ=0）
  3. text_only           — 仅文本规则（α=0, β=1.0, γ=0）
  4. image_text_fusion   — 图像+文本融合（默认权重 α=0.7, β=0.25, γ=0.05）
  5. fusion_equal        — 等权融合（α=0.5, β=0.45, γ=0.05）
  6. fusion_text_heavy   — 文本主导（α=0.3, β=0.65, γ=0.05）

用法：
    python scripts/ablation_study.py
    python scripts/ablation_study.py --dataset-root Dataset/archive/SkinDisease \\
        --artifacts-dir artifacts --max-per-class 40 --output artifacts/ablation_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.mock_engine import load_class_labels, mock_infer, resolve_dataset_root
from core.local_hybrid import (
    confusion_matrix,
    f1_macro,
    local_hybrid_infer,
    synthetic_symptom_for_label,
    topk_hit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation study for local inference methods")
    parser.add_argument("--dataset-root", type=str, default="Dataset/archive/SkinDisease")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--max-per-class", type=int, default=40)
    parser.add_argument("--output", type=str, default="artifacts/ablation_report.json")
    return parser.parse_args()


def collect_samples(dataset_root: Path, max_per_class: int) -> list[tuple[Path, str]]:
    test_root = dataset_root / "test"
    samples: list[tuple[Path, str]] = []
    for label_dir in sorted(test_root.iterdir()):
        if not label_dir.is_dir():
            continue
        files = [
            p for p in sorted(label_dir.rglob("*"))
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        for p in files[:max_per_class]:
            samples.append((p, label_dir.name))
    return samples


def _compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    top3_hits: int,
    labels: list[str],
) -> dict:
    n = max(len(y_true), 1)
    top1 = sum(int(t == p) for t, p in zip(y_true, y_pred)) / n
    top3 = top3_hits / n
    macro = f1_macro(y_true, y_pred, labels)
    cm = confusion_matrix(y_true, y_pred, labels)
    return {
        "num_samples": len(y_true),
        "top1": round(top1, 4),
        "top3": round(top3, 4),
        "macro_f1": round(macro, 4),
        "confusion_matrix": cm,
    }


def run_experiment(
    name: str,
    samples: list[tuple[Path, str]],
    labels: list[str],
    artifacts_dir: str,
    alpha: float = 0.7,
    beta: float = 0.25,
    gamma: float = 0.05,
    mode: str = "hybrid",
    use_mock: bool = False,
) -> dict:
    y_true: list[str] = []
    y_pred: list[str] = []
    top3_hit_count = 0

    for path, true_label in samples:
        image_bytes = path.read_bytes()
        symptom = synthetic_symptom_for_label(true_label)

        if use_mock:
            result = mock_infer(symptom_text=symptom, labels=labels)
            top3 = [{"label": result["primary_diagnosis"], "score": result["confidence"]}]
        else:
            result = local_hybrid_infer(
                image_bytes=image_bytes,
                symptom_text=symptom if mode != "image_only" else "",
                labels=labels,
                artifacts_dir=artifacts_dir,
                mode=mode,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            top3 = result.get("top3_candidates", [])

        y_true.append(true_label)
        y_pred.append(result["primary_diagnosis"])
        top3_hit_count += int(topk_hit(true_label, top3))

    metrics = _compute_metrics(y_true, y_pred, top3_hit_count, labels)
    return {"name": name, "alpha": alpha, "beta": beta, "gamma": gamma, **metrics}


def print_summary(results: list[dict]) -> None:
    header = f"{'Experiment':<25} {'Top-1':>7} {'Top-3':>7} {'Macro-F1':>10} {'Samples':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<25} {r['top1']:>7.4f} {r['top3']:>7.4f} "
            f"{r['macro_f1']:>10.4f} {r['num_samples']:>8}"
        )
    print("=" * len(header))


def plot_ablation(results: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    names = [r["name"] for r in results]
    top1 = [r["top1"] for r in results]
    top3 = [r["top3"] for r in results]
    f1 = [r["macro_f1"] for r in results]

    x = np.arange(len(names))
    width = 0.28

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width, top1, width, label="Top-1 Acc", color="steelblue", alpha=0.85)
    ax.bar(x, top3, width, label="Top-3 Acc", color="darkorange", alpha=0.85)
    ax.bar(x + width, f1, width, label="Macro F1", color="green", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Comparison of Inference Methods")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    for i, (t1, t3, f) in enumerate(zip(top1, top3, f1)):
        ax.text(i - width, t1 + 0.01, f"{t1:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i, t3 + 0.01, f"{t3:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + width, f + 0.01, f"{f:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out_path = output_dir / "ablation_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    labels = load_class_labels(dataset_root)
    samples = collect_samples(Path(dataset_root), max_per_class=args.max_per_class)

    if not samples:
        raise RuntimeError("No test images found.")

    print(f"Dataset: {dataset_root}")
    print(f"Labels: {len(labels)}  Samples: {len(samples)}")
    print(f"Artifacts: {args.artifacts_dir}\n")

    experiments = [
        # (name, alpha, beta, gamma, mode, use_mock)
        ("1_local_mock",        0.0,  0.0,  0.0,  "hybrid",     True),
        ("2_image_only",        1.0,  0.0,  0.0,  "image_only", False),
        ("3_text_only",         0.0,  1.0,  0.0,  "hybrid",     False),
        ("4_fusion_default",    0.7,  0.25, 0.05, "hybrid",     False),
        ("5_fusion_equal",      0.5,  0.45, 0.05, "hybrid",     False),
        ("6_fusion_text_heavy", 0.3,  0.65, 0.05, "hybrid",     False),
    ]

    results = []
    for name, alpha, beta, gamma, mode, use_mock in experiments:
        print(f"Running: {name} ...", end=" ", flush=True)
        try:
            r = run_experiment(
                name=name,
                samples=samples,
                labels=labels,
                artifacts_dir=args.artifacts_dir,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                mode=mode,
                use_mock=use_mock,
            )
            results.append(r)
            print(f"Top-1={r['top1']:.4f}  Top-3={r['top3']:.4f}  F1={r['macro_f1']:.4f}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    print_summary(results)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Strip confusion matrices for cleaner JSON summary
    summary = [{k: v for k, v in r.items() if k != "confusion_matrix"} for r in results]
    report = {
        "dataset_root": str(dataset_root),
        "artifacts_dir": args.artifacts_dir,
        "max_per_class": args.max_per_class,
        "num_labels": len(labels),
        "total_samples": len(samples),
        "results": summary,
    }
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {output.resolve()}")

    plot_ablation(results, output.parent)


if __name__ == "__main__":
    main()
