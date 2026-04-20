from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.local_hybrid import (
    confusion_matrix,
    f1_macro,
    local_hybrid_infer,
    synthetic_symptom_for_label,
    topk_hit,
)
from core.mock_engine import load_class_labels, mock_infer, resolve_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate old_mock vs image_only vs image+text_fusion")
    parser.add_argument("--dataset-root", type=str, default="Dataset/archive/SkinDisease")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/multi_model_compare/efficientnet_b0")
    parser.add_argument("--max-per-class", type=int, default=40)
    parser.add_argument("--symptom-mode", type=str, default="label_hint", choices=["label_hint", "empty"])
    parser.add_argument("--output", type=str, default="artifacts/multi_model_compare/efficientnet_b0/local_eval_report.json")
    return parser.parse_args()


def collect_samples(dataset_root: Path, max_per_class: int) -> list[tuple[Path, str]]:
    test_root = dataset_root / "test"
    samples: list[tuple[Path, str]] = []
    for label_dir in sorted(test_root.iterdir()):
        if not label_dir.is_dir():
            continue
        files = [p for p in sorted(label_dir.rglob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        for p in files[:max_per_class]:
            samples.append((p, label_dir.name))
    return samples


def symptom_text_for(label: str, mode: str) -> str:
    if mode == "empty":
        return ""
    return synthetic_symptom_for_label(label)


def evaluate_method(
    name: str,
    samples: list[tuple[Path, str]],
    labels: list[str],
    artifacts_dir: str,
    symptom_mode: str,
):
    y_true: list[str] = []
    y_pred: list[str] = []
    top3_hit_count = 0

    for path, true_label in samples:
        image_bytes = path.read_bytes()
        symptom = symptom_text_for(true_label, symptom_mode)

        if name == "old_mock":
            result = mock_infer(symptom_text=symptom, labels=labels)
            top3 = [{"label": result["primary_diagnosis"], "score": result["confidence"]}]
        elif name == "image_only":
            result = local_hybrid_infer(
                image_bytes=image_bytes,
                symptom_text="",
                labels=labels,
                artifacts_dir=artifacts_dir,
                mode="image_only",
            )
            top3 = result.get("top3_candidates", [])
        elif name == "image_text_fusion":
            result = local_hybrid_infer(
                image_bytes=image_bytes,
                symptom_text=symptom,
                labels=labels,
                artifacts_dir=artifacts_dir,
                mode="hybrid",
            )
            top3 = result.get("top3_candidates", [])
        else:
            raise ValueError(name)

        y_true.append(true_label)
        y_pred.append(result["primary_diagnosis"])
        top3_hit_count += int(topk_hit(true_label, top3))

    top1 = sum(int(t == p) for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
    top3 = top3_hit_count / max(len(y_true), 1)
    macro = f1_macro(y_true, y_pred, labels)
    cm = confusion_matrix(y_true, y_pred, labels)

    return {
        "method": name,
        "num_samples": len(samples),
        "top1": round(top1, 4),
        "top3": round(top3, 4),
        "macro_f1": round(macro, 4),
        "confusion_matrix": cm,
    }


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    labels = load_class_labels(dataset_root)

    samples = collect_samples(Path(dataset_root), max_per_class=args.max_per_class)
    if not samples:
        raise RuntimeError("No test images found.")

    report = {
        "dataset_root": str(dataset_root),
        "artifacts_dir": args.artifacts_dir,
        "symptom_mode": args.symptom_mode,
        "max_per_class": args.max_per_class,
        "results": [
            evaluate_method("old_mock", samples, labels, args.artifacts_dir, args.symptom_mode),
            evaluate_method("image_only", samples, labels, args.artifacts_dir, args.symptom_mode),
            evaluate_method("image_text_fusion", samples, labels, args.artifacts_dir, args.symptom_mode),
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for row in report["results"]:
        print(row)
    print(f"Saved report to {output.resolve()}")


if __name__ == "__main__":
    main()
