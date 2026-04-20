from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.local_model import DEFAULT_IMAGE_SIZE, DEFAULT_MEAN, DEFAULT_STD, build_model, get_eval_transform, get_train_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local skin disease image classifier")
    parser.add_argument("--dataset-root", type=str, default="Dataset/archive/SkinDisease")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument(
        "--arch",
        type=str,
        default="efficientnet_b0",
        choices=["mobilenet_v3_small", "resnet18", "efficientnet_b0"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--freeze-epochs", type=int, default=5, help="Freeze backbone for N warmup epochs.")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--imbalance-strategy",
        type=str,
        default="class_weight",
        choices=["class_weight", "focal"],
        help="How to mitigate class imbalance.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--use-weighted-sampler",
        action="store_true",
        help="Optional weighted sampler. Keep disabled by default to avoid minority overfitting.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-num-classes", type=int, default=22)
    parser.add_argument("--bench-warmup", type=int, default=10)
    parser.add_argument("--bench-repeats", type=int, default=80)
    return parser.parse_args()


def build_manifest(dataset_root: Path, split: str, class_to_idx: dict[str, int], out_csv: Path) -> int:
    split_dir = dataset_root / split
    rows: list[tuple[str, str, int]] = []
    for label in sorted(class_to_idx):
        label_dir = split_dir / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.rglob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                rows.append((str(path.resolve()), label, class_to_idx[label]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "label_idx"])
        writer.writerows(rows)
    return len(rows)


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(pref: str) -> str:
    import torch

    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_split_file_counts(split_dir: Path) -> dict[str, int]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counts: dict[str, int] = {}
    for label_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        counts[label_dir.name] = sum(1 for p in label_dir.rglob("*") if p.suffix.lower() in exts)
    return counts


def validate_dataset(dataset_root: Path, expected_num_classes: int) -> tuple[dict[str, int], dict[str, int]]:
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected train/test folders under: {dataset_root}")

    train_counts = get_split_file_counts(train_dir)
    test_counts = get_split_file_counts(test_dir)
    train_labels = set(train_counts.keys())
    test_labels = set(test_counts.keys())

    if train_labels != test_labels:
        missing_in_test = sorted(train_labels - test_labels)
        missing_in_train = sorted(test_labels - train_labels)
        raise ValueError(
            f"train/test labels mismatch. missing_in_test={missing_in_test}, missing_in_train={missing_in_train}"
        )

    if expected_num_classes > 0 and len(train_labels) != expected_num_classes:
        raise ValueError(f"Expected {expected_num_classes} classes, got {len(train_labels)}")

    empty_train = [k for k, v in train_counts.items() if v == 0]
    empty_test = [k for k, v in test_counts.items() if v == 0]
    if empty_train or empty_test:
        raise ValueError(f"Found empty class directories. train_empty={empty_train}, test_empty={empty_test}")

    return train_counts, test_counts


def _is_backbone_param(name: str, arch: str) -> bool:
    if arch in {"mobilenet_v3_small", "efficientnet_b0"}:
        return name.startswith("features")
    return not name.startswith("fc")


def set_backbone_trainable(model, arch: str, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if _is_backbone_param(name, arch):
            param.requires_grad = trainable
        else:
            param.requires_grad = True


class FocalLoss:
    def __init__(self, alpha=None, gamma: float = 2.0):
        from torch import nn

        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")

    def __call__(self, logits, targets):
        import torch

        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def model_param_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def estimate_model_size_mb_from_params(model) -> float:
    # Float32 parameters => 4 bytes/parameter.
    return float(model_param_count(model) * 4 / (1024 * 1024))


def benchmark_inference_ms_per_image(
    model,
    *,
    device: str,
    image_size: int,
    warmup: int = 10,
    repeats: int = 80,
) -> float:
    import time as _time

    import torch

    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    warmup = max(int(warmup), 0)
    repeats = max(int(repeats), 1)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = _time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = _time.perf_counter()

    return float((t1 - t0) * 1000.0 / repeats)


def evaluate(model, loader, device: str, labels: list[str]):
    import torch

    model.eval()
    total = 0
    correct_top1 = 0
    correct_top3 = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            top1 = torch.argmax(probs, dim=1)
            top3 = torch.topk(probs, k=min(3, probs.shape[1]), dim=1).indices

            total += targets.size(0)
            correct_top1 += (top1 == targets).sum().item()
            correct_top3 += (top3 == targets.unsqueeze(1)).any(dim=1).sum().item()

            all_true.extend(targets.cpu().tolist())
            all_pred.extend(top1.cpu().tolist())

    top1_acc = correct_top1 / max(total, 1)
    top3_acc = correct_top3 / max(total, 1)

    conf = [[0 for _ in labels] for _ in labels]
    for t, p in zip(all_true, all_pred):
        conf[t][p] += 1

    eps = 1e-9
    f1s: list[float] = []
    per_class_recall: dict[str, float] = {}
    for i, label in enumerate(labels):
        tp = conf[i][i]
        fp = sum(conf[r][i] for r in range(len(labels)) if r != i)
        fn = sum(conf[i][c] for c in range(len(labels)) if c != i)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        per_class_recall[label] = float(recall)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1s.append(f1)

    macro_f1 = sum(f1s) / max(len(f1s), 1)
    return {
        "top1_acc": float(top1_acc),
        "top3_acc": float(top3_acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": conf,
        "per_class_recall": per_class_recall,
    }


def main() -> None:
    args = parse_args()

    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader
    from torch.utils.data import WeightedRandomSampler
    from torchvision.datasets import ImageFolder

    set_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    validate_dataset(dataset_root, expected_num_classes=args.expected_num_classes)
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_set = ImageFolder(str(train_dir), transform=get_train_transform(args.image_size))
    test_set = ImageFolder(str(test_dir), transform=get_eval_transform(args.image_size))

    labels = list(train_set.classes)
    class_to_idx = dict(train_set.class_to_idx)
    num_classes = len(labels)

    train_count = build_manifest(dataset_root, "train", class_to_idx, artifacts_dir / "train_manifest.csv")
    test_count = build_manifest(dataset_root, "test", class_to_idx, artifacts_dir / "test_manifest.csv")

    train_target_counts = Counter(train_set.targets)
    total_samples = sum(train_target_counts.values())
    class_weight_list: list[float] = []
    for idx in range(num_classes):
        count = int(train_target_counts.get(idx, 0))
        if count <= 0:
            raise ValueError(f"Class '{labels[idx]}' has zero training samples.")
        class_weight_list.append(total_samples / (num_classes * count))

    sampler = None
    if args.use_weighted_sampler:
        sample_weights = [class_weight_list[target] for target in train_set.targets]
        sampler = WeightedRandomSampler(sample_weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    device = choose_device(args.device)
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(
        arch=args.arch,
        num_classes=num_classes,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_epochs > 0,
    ).to(device)

    if args.freeze_epochs > 0:
        set_backbone_trainable(model, arch=args.arch, trainable=False)
    else:
        set_backbone_trainable(model, arch=args.arch, trainable=True)

    class_weights_tensor = torch.tensor(class_weight_list, dtype=torch.float32, device=device)
    if args.imbalance_strategy == "class_weight":
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=args.focal_gamma)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    amp_enabled = args.use_amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    history: list[dict[str, float]] = []
    started = time.time()

    best_macro_f1 = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, arch=args.arch, trainable=True)
            print(f"[info] unfroze backbone at epoch={epoch}")

        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * targets.size(0)
            running_correct += (logits.argmax(dim=1) == targets).sum().item()
            running_total += targets.size(0)

        scheduler.step()

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        eval_metrics = evaluate(model, test_loader, device=device, labels=labels)

        epoch_stat = {
            "epoch": epoch,
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
            "train_loss": round(float(train_loss), 4),
            "train_acc": round(float(train_acc), 4),
            "test_top1": round(float(eval_metrics["top1_acc"]), 4),
            "test_top3": round(float(eval_metrics["top3_acc"]), 4),
            "test_macro_f1": round(float(eval_metrics["macro_f1"]), 4),
        }
        history.append(epoch_stat)
        print(epoch_stat)

        current_macro_f1 = eval_metrics["macro_f1"]
        if current_macro_f1 > best_macro_f1 + args.min_delta:
            best_macro_f1 = current_macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop_patience:
                print(
                    f"[early-stop] no macro_f1 improvement for {args.early_stop_patience} epochs. "
                    f"best_epoch={best_epoch}, best_macro_f1={best_macro_f1:.4f}"
                )
                break

    elapsed = round(time.time() - started, 2)
    model.load_state_dict(best_state)

    model_params = model_param_count(model)
    model_size_mb = estimate_model_size_mb_from_params(model)
    inference_ms = benchmark_inference_ms_per_image(
        model,
        device=device,
        image_size=args.image_size,
        warmup=args.bench_warmup,
        repeats=args.bench_repeats,
    )

    train_count_by_label = {labels[i]: int(train_target_counts.get(i, 0)) for i in range(num_classes)}

    ckpt = {
        "arch": args.arch,
        "num_classes": num_classes,
        "state_dict": model.cpu().state_dict(),
        "image_size": args.image_size,
        "mean": DEFAULT_MEAN,
        "std": DEFAULT_STD,
    }
    torch.save(ckpt, artifacts_dir / "local_model.pkl")

    with (artifacts_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    final_eval = evaluate(model, test_loader, device="cpu", labels=labels)
    metrics = {
        "arch": args.arch,
        "epochs_requested": args.epochs,
        "epochs_trained": len(history),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "freeze_epochs": args.freeze_epochs,
        "pretrained": args.pretrained,
        "imbalance_strategy": args.imbalance_strategy,
        "focal_gamma": args.focal_gamma,
        "class_weights": {labels[i]: round(float(class_weight_list[i]), 6) for i in range(num_classes)},
        "use_weighted_sampler": args.use_weighted_sampler,
        "early_stop_patience": args.early_stop_patience,
        "use_amp": bool(amp_enabled),
        "scheduler": "CosineAnnealingLR",
        "best_epoch": best_epoch,
        "best_macro_f1": round(float(best_macro_f1), 4),
        "model_params": model_params,
        "model_size_mb": round(float(model_size_mb), 3),
        "inference_ms_per_image": round(float(inference_ms), 3),
        "benchmark_device": device,
        "bench_warmup": args.bench_warmup,
        "bench_repeats": args.bench_repeats,
        "device_used": device,
        "image_size": args.image_size,
        "mean": DEFAULT_MEAN,
        "std": DEFAULT_STD,
        "train_samples": train_count,
        "test_samples": test_count,
        "train_counts": train_count_by_label,
        "history": history,
        "final_eval": {
            "top1_acc": round(float(final_eval["top1_acc"]), 4),
            "top3_acc": round(float(final_eval["top3_acc"]), 4),
            "macro_f1": round(float(final_eval["macro_f1"]), 4),
            "confusion_matrix": final_eval["confusion_matrix"],
            "per_class_recall": {k: round(float(v), 4) for k, v in final_eval["per_class_recall"].items()},
        },
        "per_class_recall": {k: round(float(v), 4) for k, v in final_eval["per_class_recall"].items()},
        "train_seconds": elapsed,
    }
    with (artifacts_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved artifacts to: {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
