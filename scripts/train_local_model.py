from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from core.local_model import DEFAULT_IMAGE_SIZE, DEFAULT_MEAN, DEFAULT_STD, build_model, get_eval_transform, get_train_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local skin disease image classifier")
    parser.add_argument("--dataset-root", type=str, default="Dataset/archive/SkinDisease")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
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


def choose_device(pref: str) -> str:
    import torch

    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


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
            images = images.to(device)
            targets = targets.to(device)
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
    for i in range(len(labels)):
        tp = conf[i][i]
        fp = sum(conf[r][i] for r in range(len(labels)) if r != i)
        fn = sum(conf[i][c] for c in range(len(labels)) if c != i)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1s.append(f1)

    macro_f1 = sum(f1s) / max(len(f1s), 1)
    return {
        "top1_acc": round(float(top1_acc), 4),
        "top3_acc": round(float(top3_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "confusion_matrix": conf,
    }


def main() -> None:
    args = parse_args()

    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    dataset_root = Path(args.dataset_root)
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected train/test folders under: {dataset_root}")

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_set = ImageFolder(str(train_dir), transform=get_train_transform(args.image_size))
    test_set = ImageFolder(str(test_dir), transform=get_eval_transform(args.image_size))

    labels = list(train_set.classes)
    class_to_idx = dict(train_set.class_to_idx)

    train_count = build_manifest(dataset_root, "train", class_to_idx, artifacts_dir / "train_manifest.csv")
    test_count = build_manifest(dataset_root, "test", class_to_idx, artifacts_dir / "test_manifest.csv")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = choose_device(args.device)
    model = build_model(
        arch=args.arch,
        num_classes=len(labels),
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    history: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            running_correct += (logits.argmax(dim=1) == targets).sum().item()
            running_total += targets.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        eval_metrics = evaluate(model, test_loader, device=device, labels=labels)
        epoch_stat = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "train_acc": round(float(train_acc), 4),
            "test_top1": eval_metrics["top1_acc"],
            "test_top3": eval_metrics["top3_acc"],
            "test_macro_f1": eval_metrics["macro_f1"],
        }
        history.append(epoch_stat)
        print(epoch_stat)

    elapsed = round(time.time() - started, 2)

    train_counts = Counter(train_set.targets)
    train_count_by_label = {labels[i]: int(train_counts.get(i, 0)) for i in range(len(labels))}

    ckpt = {
        "arch": args.arch,
        "num_classes": len(labels),
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
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "freeze_backbone": args.freeze_backbone,
        "pretrained": args.pretrained,
        "device_used": device,
        "image_size": args.image_size,
        "mean": DEFAULT_MEAN,
        "std": DEFAULT_STD,
        "train_samples": train_count,
        "test_samples": test_count,
        "train_counts": train_count_by_label,
        "history": history,
        "final_eval": final_eval,
        "train_seconds": elapsed,
    }
    with (artifacts_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved artifacts to: {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
