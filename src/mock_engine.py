from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

DISCLAIMER = "初步筛查结果，非临床诊断结论"

# Symptom keywords for deterministic local fallback.
KEYWORD_RULES: dict[str, tuple[str, ...]] = {
    "Acne": ("痘", "粉刺", "丘疹", "blackhead", "acne", "pimple"),
    "Actinic_Keratosis": ("日晒", "角化", "粗糙斑", "actinic", "sun damage"),
    "Benign_tumors": ("良性", "肿物", "包块", "nevus", "benign"),
    "Bullous": ("水疱", "大疱", "起泡", "blister", "bullous"),
    "Candidiasis": ("念珠菌", "真菌", "白色分泌物", "candida", "yeast"),
    "DrugEruption": ("用药后", "药疹", "过敏药物", "drug eruption"),
    "Eczema": ("瘙痒", "湿疹", "渗出", "eczema", "itch"),
    "Infestations_Bites": ("虫咬", "叮咬", "螨", "bite", "infestation"),
    "Lichen": ("扁平苔藓", "苔藓样", "紫红", "lichen"),
    "Lupus": ("狼疮", "蝶形红斑", "lupus"),
    "Moles": ("痣", "色素痣", "mole", "nevus"),
    "Psoriasis": ("银屑病", "鳞屑", "斑块", "psoriasis", "scaly"),
    "Rosacea": ("酒渣鼻", "潮红", "毛细血管扩张", "rosacea"),
    "Seborrh_Keratoses": ("脂溢性角化", "老年斑", "seborrheic", "keratosis"),
    "SkinCancer": ("皮肤癌", "恶性", "出血", "ulcer", "cancer"),
    "Sun_Sunlight_Damage": ("晒伤", "日光", "sunburn", "sunlight"),
    "Tinea": ("癣", "体癣", "足癣", "ringworm", "tinea"),
    "Unknown_Normal": ("正常", "无明显异常", "normal"),
    "Vascular_Tumors": ("血管瘤", "血管性肿物", "vascular tumor", "hemangioma"),
    "Vasculitis": ("血管炎", "紫癜", "vasculitis"),
    "Vitiligo": ("白癜风", "色素脱失", "vitiligo", "depigmentation"),
    "Warts": ("疣", "寻常疣", "wart", "verruca"),
}


def resolve_dataset_root(preferred: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if preferred:
        candidates.append(Path(preferred))

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "Dataset" / "archive" / "SkinDisease",
            cwd / "dataset" / "archive" / "SkinDisease",
            cwd / "Dataset",
            cwd / "dataset",
        ]
    )

    for path in candidates:
        if path.exists() and path.is_dir():
            if (path / "train").exists() or (path / "test").exists():
                return path
    raise FileNotFoundError("未找到数据集目录，请确认 Dataset/archive/SkinDisease 存在。")


def load_class_labels(dataset_root: str | Path) -> list[str]:
    root = Path(dataset_root)
    labels: set[str] = set()
    for split in ("train", "test"):
        split_dir = root / split
        if split_dir.exists():
            for folder in split_dir.iterdir():
                if folder.is_dir():
                    labels.add(folder.name)
    if not labels:
        raise ValueError(f"在 {root} 下未找到类别目录。")
    return sorted(labels)


def _match_scores(symptom_text: str, labels: Iterable[str]) -> dict[str, int]:
    text = (symptom_text or "").lower()
    scores: dict[str, int] = {}
    for label in labels:
        keywords = KEYWORD_RULES.get(label, ())
        score = sum(1 for kw in keywords if kw.lower() in text and kw.strip())
        scores[label] = score
    return scores


def _deterministic_pick(seed_text: str, labels: list[str]) -> str:
    digest = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(labels)
    return labels[idx]


def mock_infer(symptom_text: str, labels: list[str]) -> dict:
    if not labels:
        raise ValueError("labels 不能为空。")

    text = (symptom_text or "").strip()
    scores = _match_scores(text, labels)
    max_score = max(scores.values()) if scores else 0

    if max_score > 0:
        top_labels = sorted([k for k, v in scores.items() if v == max_score])
        primary = top_labels[0]
        confidence = min(0.55 + max_score * 0.12, 0.93)
    elif not text and "Unknown_Normal" in labels:
        primary = "Unknown_Normal"
        confidence = 0.42
    else:
        primary = _deterministic_pick(text or "empty_symptom", labels)
        confidence = 0.46

    return {
        "primary_diagnosis": primary,
        "confidence": round(float(confidence), 2),
        "source": "local_mock",
        "mock_result": True,
        "note": DISCLAIMER,
    }
