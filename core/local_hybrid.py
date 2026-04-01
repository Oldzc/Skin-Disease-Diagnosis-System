from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.local_model import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MEAN,
    DEFAULT_STD,
    build_model,
    get_eval_transform,
)
from src.mock_engine import DISCLAIMER

DEFAULT_ARTIFACTS_DIR = "artifacts"
MODEL_FILE = "local_model.pkl"
LABEL_MAP_FILE = "label_map.json"
METRICS_FILE = "metrics.json"

NEGATION_WORDS = ("无", "没有", "未见", "否认", "不", "not", "no", "without", "denies")

DURATION_HINTS = {
    "acute": ("急性", "突发", "短期", "近几天", "sudden", "acute"),
    "chronic": ("慢性", "反复", "长期", "数月", "chronic", "recurrent"),
}

LOCATION_HINTS = {
    "face": ("面部", "脸", "鼻", "额头", "cheek", "face", "nose", "forehead"),
    "trunk": ("躯干", "胸", "背", "腹", "trunk", "chest", "back", "abdomen"),
    "limb": ("四肢", "手", "脚", "腿", "臂", "limb", "hand", "foot", "leg", "arm"),
    "scalp": ("头皮", "scalp"),
}

SEVERITY_HINTS = {
    "mild": ("轻", "轻微", "偶发", "mild", "slight"),
    "moderate": ("中度", "moderate"),
    "severe": ("重", "严重", "剧烈", "明显", "severe", "intense"),
}

ITCH_HINTS = {
    "no_itch": ("无瘙痒", "不痒"),
    "mild_itch": ("轻微瘙痒", "偶尔痒"),
    "moderate_itch": ("中度瘙痒",),
    "severe_itch": ("剧烈瘙痒", "奇痒", "瘙痒剧烈"),
}

PAIN_HINTS = {
    "no_pain": ("无疼痛", "不痛"),
    "mild_pain": ("轻微疼痛", "隐痛"),
    "moderate_pain": ("中度疼痛",),
    "severe_pain": ("剧烈疼痛", "疼痛剧烈", "剧痛"),
}

TRIGGER_HINTS = {
    "sun": ("日晒诱因", "日晒后", "晒后"),
    "drug": ("用药诱因", "用药后", "药物诱因"),
    "bite": ("虫咬诱因", "虫咬后", "叮咬后"),
    "contact": ("接触刺激物", "接触诱因", "接触后"),
}

MORPHOLOGY_HINTS = {
    "erythema": ("红斑",),
    "scale": ("鳞屑", "脱屑"),
    "vesicle": ("水疱", "起泡"),
    "ulcer": ("溃疡",),
    "papule": ("丘疹",),
    "pigment": ("色素改变", "色素沉着", "色素脱失"),
}

RECURRENCE_HINTS = {
    "first": ("首次发作",),
    "recurrent": ("反复发作", "复发"),
}

AGE_HINTS = {
    "child": ("儿童",),
    "youth": ("青年",),
    "middle": ("中年",),
    "elderly": ("老年",),
}

LABEL_RULES: dict[str, dict[str, Any]] = {
    "Acne": {
        "keywords": {"痘": 1.4, "粉刺": 1.4, "丘疹": 1.0, "acne": 1.4, "pimple": 1.2, "blackhead": 1.3},
        "locations": {"face": 0.25, "trunk": 0.1},
        "duration": {"chronic": 0.1},
        "itch": {"mild_itch": 0.05},
        "morphology": {"papule": 0.2},
        "recurrence": {"recurrent": 0.1},
        "age": {"youth": 0.2},
    },
    "Actinic_Keratosis": {
        "keywords": {"日晒": 1.2, "角化": 1.3, "粗糙": 0.7, "actinic": 1.2, "sun damage": 1.2},
        "locations": {"face": 0.2, "limb": 0.15},
        "trigger": {"sun": 0.3},
        "morphology": {"scale": 0.15},
        "age": {"elderly": 0.2},
    },
    "Benign_tumors": {
        "keywords": {"良性": 1.2, "肿物": 1.0, "包块": 0.8, "benign": 1.2, "nevus": 0.9},
        "pain": {"no_pain": 0.05},
    },
    "Bullous": {
        "keywords": {"水疱": 1.5, "大疱": 1.5, "起泡": 1.2, "blister": 1.4, "bullous": 1.4},
        "severity": {"severe": 0.2},
        "pain": {"moderate_pain": 0.1, "severe_pain": 0.15},
        "morphology": {"vesicle": 0.3},
    },
    "Candidiasis": {
        "keywords": {"念珠菌": 1.5, "真菌": 0.8, "candida": 1.5, "yeast": 1.2},
        "locations": {"trunk": 0.1, "limb": 0.1},
        "itch": {"moderate_itch": 0.1, "severe_itch": 0.1},
        "morphology": {"erythema": 0.1},
    },
    "DrugEruption": {
        "keywords": {"药疹": 1.6, "用药后": 1.4, "药物过敏": 1.3, "drug eruption": 1.5},
        "duration": {"acute": 0.15},
        "trigger": {"drug": 0.4},
        "itch": {"moderate_itch": 0.1, "severe_itch": 0.15},
        "morphology": {"erythema": 0.1},
        "recurrence": {"first": 0.05},
    },
    "Eczema": {
        "keywords": {"瘙痒": 1.3, "湿疹": 1.5, "渗出": 1.0, "eczema": 1.5, "itch": 1.2},
        "severity": {"mild": 0.05, "severe": 0.1},
        "duration": {"chronic": 0.1},
        "itch": {"moderate_itch": 0.15, "severe_itch": 0.25},
        "morphology": {"erythema": 0.1, "scale": 0.1},
        "recurrence": {"recurrent": 0.2},
    },
    "Infestations_Bites": {
        "keywords": {"虫咬": 1.5, "叮咬": 1.4, "螨": 1.1, "bite": 1.3, "infestation": 1.2},
        "duration": {"acute": 0.15},
        "trigger": {"bite": 0.4},
        "itch": {"moderate_itch": 0.1, "severe_itch": 0.15},
        "morphology": {"papule": 0.1},
    },
    "Lichen": {
        "keywords": {"扁平苔藓": 1.6, "苔藓样": 1.3, "lichen": 1.5},
        "itch": {"mild_itch": 0.1, "moderate_itch": 0.1},
        "morphology": {"papule": 0.15},
        "recurrence": {"recurrent": 0.1},
    },
    "Lupus": {
        "keywords": {"狼疮": 1.7, "蝶形红斑": 1.4, "lupus": 1.7},
        "locations": {"face": 0.2},
        "trigger": {"sun": 0.2},
        "morphology": {"erythema": 0.2},
        "recurrence": {"recurrent": 0.15},
    },
    "Moles": {
        "keywords": {"痣": 1.4, "色素痣": 1.5, "mole": 1.3, "nevus": 1.3},
        "pain": {"no_pain": 0.05},
        "itch": {"no_itch": 0.05},
        "morphology": {"pigment": 0.2},
    },
    "Psoriasis": {
        "keywords": {"银屑病": 1.6, "鳞屑": 1.2, "斑块": 1.0, "psoriasis": 1.6, "scaly": 1.1},
        "duration": {"chronic": 0.2},
        "itch": {"moderate_itch": 0.1, "severe_itch": 0.15},
        "morphology": {"erythema": 0.1, "scale": 0.25},
        "recurrence": {"recurrent": 0.25},
    },
    "Rosacea": {
        "keywords": {"酒渣": 1.6, "潮红": 1.2, "毛细血管扩张": 1.1, "rosacea": 1.6},
        "locations": {"face": 0.3},
        "trigger": {"sun": 0.1},
        "morphology": {"erythema": 0.2},
        "recurrence": {"recurrent": 0.1},
        "age": {"middle": 0.1},
    },
    "Seborrh_Keratoses": {
        "keywords": {"脂溢性角化": 1.6, "老年斑": 1.3, "seborrheic": 1.4, "keratosis": 1.2},
        "pain": {"no_pain": 0.05},
        "morphology": {"scale": 0.1, "pigment": 0.1},
        "age": {"elderly": 0.3},
    },
    "SkinCancer": {
        "keywords": {"皮肤癌": 1.8, "恶性": 1.4, "溃疡": 1.2, "出血": 1.0, "cancer": 1.7, "ulcer": 1.1},
        "severity": {"severe": 0.2},
        "pain": {"moderate_pain": 0.1, "severe_pain": 0.15},
        "morphology": {"ulcer": 0.3, "pigment": 0.1},
        "age": {"elderly": 0.15},
    },
    "Sun_Sunlight_Damage": {
        "keywords": {"晒伤": 1.5, "日光": 1.2, "sunburn": 1.5, "sunlight": 1.2},
        "duration": {"acute": 0.1},
        "trigger": {"sun": 0.4},
        "pain": {"mild_pain": 0.1, "moderate_pain": 0.15},
        "morphology": {"erythema": 0.2},
    },
    "Tinea": {
        "keywords": {"癣": 1.6, "体癣": 1.5, "足癣": 1.5, "ringworm": 1.4, "tinea": 1.5},
        "locations": {"limb": 0.2},
        "itch": {"moderate_itch": 0.15, "severe_itch": 0.15},
        "morphology": {"erythema": 0.1, "scale": 0.15},
        "recurrence": {"recurrent": 0.1},
    },
    "Unknown_Normal": {
        "keywords": {"正常": 1.4, "无明显异常": 1.6, "normal": 1.4},
        "pain": {"no_pain": 0.1},
        "itch": {"no_itch": 0.1},
    },
    "Vascular_Tumors": {
        "keywords": {"血管瘤": 1.6, "血管性肿物": 1.2, "hemangioma": 1.5, "vascular tumor": 1.4},
        "morphology": {"erythema": 0.1},
        "age": {"child": 0.15},
    },
    "Vasculitis": {
        "keywords": {"血管炎": 1.7, "紫癜": 1.4, "vasculitis": 1.7},
        "severity": {"severe": 0.1},
        "pain": {"moderate_pain": 0.1},
        "morphology": {"erythema": 0.1},
    },
    "Vitiligo": {
        "keywords": {"白癜风": 1.7, "色素脱失": 1.4, "vitiligo": 1.7, "depigmentation": 1.4},
        "pain": {"no_pain": 0.05},
        "itch": {"no_itch": 0.05},
        "morphology": {"pigment": 0.3},
    },
    "Warts": {
        "keywords": {"疣": 1.6, "寻常疣": 1.6, "wart": 1.5, "verruca": 1.4},
        "morphology": {"papule": 0.1},
        "recurrence": {"recurrent": 0.1},
    },
}


@dataclass
class LoadedArtifacts:
    model: Any
    labels: list[str]
    train_counts: dict[str, int]
    image_size: int
    mean: list[float]
    std: list[float]


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exps = np.exp(shifted)
    denom = np.sum(exps)
    if denom <= 0:
        return np.ones_like(scores) / len(scores)
    return exps / denom


def _normalize_probs(arr: np.ndarray) -> np.ndarray:
    total = float(np.sum(arr))
    if total <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / total


def _safe_label_array(labels: list[str], default: float = 1e-6) -> np.ndarray:
    return np.full((len(labels),), float(default), dtype=np.float64)


def get_artifact_paths(artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR) -> dict[str, Path]:
    root = Path(artifacts_dir)
    return {
        "root": root,
        "model": root / MODEL_FILE,
        "label_map": root / LABEL_MAP_FILE,
        "metrics": root / METRICS_FILE,
    }


def local_hybrid_artifacts_available(artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR) -> bool:
    paths = get_artifact_paths(artifacts_dir)
    return paths["model"].exists() and paths["label_map"].exists()


@lru_cache(maxsize=2)
def _load_artifacts_cached(artifacts_root: str) -> LoadedArtifacts:
    import torch

    paths = get_artifact_paths(artifacts_root)
    if not local_hybrid_artifacts_available(artifacts_root):
        raise FileNotFoundError(f"local_hybrid artifacts not found under: {paths['root']}")

    with paths["label_map"].open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    if isinstance(label_map, dict):
        idx_to_label = [label_map[str(i)] for i in range(len(label_map))]
    else:
        idx_to_label = list(label_map)

    train_counts: dict[str, int] = {}
    image_size = DEFAULT_IMAGE_SIZE
    mean = DEFAULT_MEAN
    std = DEFAULT_STD
    if paths["metrics"].exists():
        with paths["metrics"].open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        train_counts = metrics.get("train_counts", {})
        image_size = int(metrics.get("image_size", DEFAULT_IMAGE_SIZE))
        mean = metrics.get("mean", DEFAULT_MEAN)
        std = metrics.get("std", DEFAULT_STD)

    checkpoint = torch.load(paths["model"], map_location="cpu")
    arch = checkpoint.get("arch", "mobilenet_v3_small")
    ckpt_num_classes = int(checkpoint.get("num_classes", len(idx_to_label)))
    model = build_model(
        arch=arch,
        num_classes=ckpt_num_classes,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return LoadedArtifacts(
        model=model,
        labels=idx_to_label,
        train_counts=train_counts,
        image_size=image_size,
        mean=mean,
        std=std,
    )


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in text


def _is_negated(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.lower())
    for neg in NEGATION_WORDS:
        pattern = rf"{re.escape(neg.lower())}.{{0,4}}{escaped}"
        if re.search(pattern, text):
            return True
    return False


def _detect_flags(text: str, hints: dict[str, tuple[str, ...]]) -> set[str]:
    detected: set[str] = set()
    for key, words in hints.items():
        if any(_contains_phrase(text, w) for w in words):
            detected.add(key)
    return detected


def text_probability(
    symptom_text: str,
    labels: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    text = (symptom_text or "").strip().lower()
    if not text:
        probs = np.ones((len(labels),), dtype=np.float64) / max(len(labels), 1)
        return probs, {"matched_signals": [], "negated_signals": []}

    duration_flags = _detect_flags(text, DURATION_HINTS)
    location_flags = _detect_flags(text, LOCATION_HINTS)
    severity_flags = _detect_flags(text, SEVERITY_HINTS)
    itch_flags = _detect_flags(text, ITCH_HINTS)
    pain_flags = _detect_flags(text, PAIN_HINTS)
    trigger_flags = _detect_flags(text, TRIGGER_HINTS)
    morphology_flags = _detect_flags(text, MORPHOLOGY_HINTS)
    recurrence_flags = _detect_flags(text, RECURRENCE_HINTS)
    age_flags = _detect_flags(text, AGE_HINTS)

    scores = _safe_label_array(labels, default=0.02)
    matched_signals: list[str] = []
    negated_signals: list[str] = []

    for idx, label in enumerate(labels):
        profile = LABEL_RULES.get(label, {})

        for phrase, weight in profile.get("keywords", {}).items():
            if _contains_phrase(text, phrase):
                if _is_negated(text, phrase):
                    scores[idx] -= weight * 0.8
                    negated_signals.append(f"{label}: not {phrase}")
                else:
                    scores[idx] += weight
                    matched_signals.append(f"{label}: {phrase}(+{weight})")

        for loc, bonus in profile.get("locations", {}).items():
            if loc in location_flags:
                scores[idx] += bonus
                matched_signals.append(f"{label}: location={loc}(+{bonus})")

        for dtag, bonus in profile.get("duration", {}).items():
            if dtag in duration_flags:
                scores[idx] += bonus
                matched_signals.append(f"{label}: duration={dtag}(+{bonus})")

        for stag, bonus in profile.get("severity", {}).items():
            if stag in severity_flags:
                scores[idx] += bonus
                matched_signals.append(f"{label}: severity={stag}(+{bonus})")

        for tag, bonus in profile.get("itch", {}).items():
            if tag in itch_flags:
                scores[idx] += bonus
                matched_signals.append(f"{label}: itch={tag}(+{bonus})")

        for tag, bonus in profile.get("pain", {}).items():
            if tag in pain_flags:
                scores[idx] += bonus
                matched_signals.append(f"{label}: pain={tag}(+{bonus})")

        for tag, bonus in profile.get("trigger", {}).items():
            if tag in trigger_flags:
                scores[idx] += bonus
                matched_signals.append(f"{label}: trigger={tag}(+{bonus})")

        for tag, bonus in profile.get("morphology", {}).items():
            if tag in morphology_flags:
                scores[idx] += bonus
                matched_signals.append(f"{label}: morphology={tag}(+{bonus})")

        for tag, bonus in profile.get("recurrence", {}).items():
            if tag in recurrence_flags:
                scores[idx] += bonus
                matched_signals.append(f"{label}: recurrence={tag}(+{bonus})")

        for tag, bonus in profile.get("age", {}).items():
            if tag in age_flags:
                scores[idx] += bonus
                matched_signals.append(f"{label}: age={tag}(+{bonus})")

    probs = _softmax(scores)
    trace = {
        "duration_flags": sorted(duration_flags),
        "location_flags": sorted(location_flags),
        "severity_flags": sorted(severity_flags),
        "itch_flags": sorted(itch_flags),
        "pain_flags": sorted(pain_flags),
        "trigger_flags": sorted(trigger_flags),
        "morphology_flags": sorted(morphology_flags),
        "recurrence_flags": sorted(recurrence_flags),
        "age_flags": sorted(age_flags),
        "matched_signals": matched_signals[:20],
        "negated_signals": negated_signals[:8],
    }
    return probs, trace


def image_probability(
    image_bytes: bytes,
    labels: list[str],
    *,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    artifacts = _load_artifacts_cached(str(Path(artifacts_dir).resolve()))
    transform = get_eval_transform(image_size=artifacts.image_size)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        logits = artifacts.model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)

    out = _safe_label_array(labels, default=1e-9)
    model_label_to_prob = {label: float(probs[i]) for i, label in enumerate(artifacts.labels)}
    for i, label in enumerate(labels):
        out[i] = model_label_to_prob.get(label, 1e-9)
    out = _normalize_probs(out)

    prior_counts = artifacts.train_counts or {}
    return out, {
        "image_size": artifacts.image_size,
        "model_labels": artifacts.labels,
        "train_counts_found": bool(prior_counts),
        "train_counts": prior_counts,
    }


def _prior_probability(labels: list[str], train_counts: dict[str, int] | None) -> np.ndarray:
    if not train_counts:
        return np.ones((len(labels),), dtype=np.float64) / max(len(labels), 1)

    arr = _safe_label_array(labels, default=0.0)
    for i, label in enumerate(labels):
        arr[i] = float(train_counts.get(label, 0.0))
    return _normalize_probs(arr)


def _topk(labels: list[str], probs: np.ndarray, k: int = 3) -> list[dict[str, float | str]]:
    k = min(k, len(labels))
    idxs = np.argsort(probs)[::-1][:k]
    return [{"label": labels[i], "score": round(float(probs[i]), 4)} for i in idxs]


def local_hybrid_infer(
    *,
    image_bytes: bytes,
    symptom_text: str,
    labels: list[str],
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    alpha: float = 0.7,
    beta: float = 0.25,
    gamma: float = 0.05,
    mode: str = "hybrid",
) -> dict[str, Any]:
    if not labels:
        raise ValueError("labels must not be empty")

    image_probs, image_trace = image_probability(
        image_bytes=image_bytes,
        labels=labels,
        artifacts_dir=artifacts_dir,
    )

    if mode == "image_only":
        final = image_probs
        text_trace = {"matched_signals": [], "negated_signals": []}
        used_weights = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}
    else:
        text_probs, text_trace = text_probability(symptom_text=symptom_text, labels=labels)
        train_counts = image_trace.get("train_counts", {})
        priors = _prior_probability(labels=labels, train_counts=train_counts)

        if not (symptom_text or "").strip():
            alpha_used, beta_used, gamma_used = 0.9, 0.05, 0.05
        else:
            alpha_used, beta_used, gamma_used = alpha, beta, gamma

        final = alpha_used * image_probs + beta_used * text_probs + gamma_used * priors
        final = _normalize_probs(final)
        used_weights = {
            "alpha": round(alpha_used, 3),
            "beta": round(beta_used, 3),
            "gamma": round(gamma_used, 3),
        }

    top3 = _topk(labels=labels, probs=final, k=3)
    primary = top3[0]["label"]
    confidence = float(top3[0]["score"])

    result = {
        "primary_diagnosis": primary,
        "confidence": round(confidence, 2),
        "source": "local_hybrid",
        "mock_result": True,
        "note": DISCLAIMER,
        "top3_candidates": top3,
        "decision_trace": {
            "mode": mode,
            "weights": used_weights,
            "text": text_trace,
            "image": {
                "image_size": image_trace.get("image_size"),
                "train_counts_found": image_trace.get("train_counts_found", False),
            },
        },
    }
    return result


SYMP_TEMPLATE: dict[str, str] = {
    "Acne": "慢性，面部，丘疹，轻微瘙痒，无疼痛，不明诱因，反复发作，青年。面部反复出现痘痘和粉刺。",
    "Actinic_Keratosis": "慢性，面部，鳞屑，无瘙痒，无疼痛，日晒诱因，首次发作，老年。暴晒后出现粗糙角化斑块。",
    "Benign_tumors": "慢性，躯干，无瘙痒，无疼痛，不明诱因，首次发作。皮肤局部出现边界较清楚的良性样肿物。",
    "Bullous": "急性，四肢，水疱，无瘙痒，中度疼痛，不明诱因，首次发作。局部出现水疱和大疱，部分疼痛。",
    "Candidiasis": "慢性，躯干，红斑，中度瘙痒，无疼痛，不明诱因，反复发作。皮肤潮湿部位发红并伴真菌样瘙痒。",
    "DrugEruption": "急性，躯干，红斑，剧烈瘙痒，无疼痛，用药诱因，首次发作。近期用药后出现全身性红疹和瘙痒。",
    "Eczema": "慢性，四肢，红斑，剧烈瘙痒，无疼痛，不明诱因，反复发作。反复瘙痒伴红斑和脱屑，病程较长。",
    "Infestations_Bites": "急性，四肢，丘疹，剧烈瘙痒，轻微疼痛，虫咬诱因，首次发作。虫咬后局部红肿瘙痒。",
    "Lichen": "慢性，四肢，丘疹，轻微瘙痒，无疼痛，不明诱因，反复发作。出现苔藓样丘疹并有轻度瘙痒。",
    "Lupus": "慢性，面部，红斑，无瘙痒，无疼痛，日晒诱因，反复发作。面部蝶形红斑，日晒后加重。",
    "Moles": "慢性，面部，色素改变，无瘙痒，无疼痛，不明诱因，首次发作。色素痣样皮损，边界相对清楚。",
    "Psoriasis": "慢性，四肢，鳞屑，中度瘙痒，无疼痛，不明诱因，反复发作。红斑基础上有银白色鳞屑，反复发作。",
    "Rosacea": "慢性，面部，红斑，无瘙痒，无疼痛，日晒诱因，反复发作，中年。面部持续潮红伴毛细血管扩张。",
    "Seborrh_Keratoses": "慢性，面部，鳞屑，无瘙痒，无疼痛，不明诱因，首次发作，老年。老年斑样角化性斑块。",
    "SkinCancer": "慢性，面部，溃疡，无瘙痒，中度疼痛，不明诱因，首次发作，老年。皮损增大并有溃疡或出血倾向。",
    "Sun_Sunlight_Damage": "急性，面部，红斑，无瘙痒，中度疼痛，日晒诱因，首次发作。日晒后皮肤发红疼痛并有损伤。",
    "Tinea": "慢性，四肢，红斑，中度瘙痒，无疼痛，不明诱因，反复发作。环形红斑伴鳞屑，考虑癣。",
    "Unknown_Normal": "无瘙痒，无疼痛，不明诱因，首次发作。皮肤外观基本正常，无明显异常。",
    "Vascular_Tumors": "慢性，躯干，红斑，无瘙痒，无疼痛，不明诱因，首次发作，儿童。血管瘤样红色隆起皮损。",
    "Vasculitis": "急性，四肢，红斑，无瘙痒，中度疼痛，不明诱因，首次发作。出现紫癜样皮疹，考虑血管炎。",
    "Vitiligo": "慢性，面部，色素改变，无瘙痒，无疼痛，不明诱因，反复发作。局部色素脱失形成白斑。",
    "Warts": "慢性，四肢，丘疹，无瘙痒，无疼痛，不明诱因，反复发作。皮肤出现疣状赘生物。",
}


def synthetic_symptom_for_label(label: str) -> str:
    return SYMP_TEMPLATE.get(label, "皮肤出现不适皮损，需要初步筛查。")


def f1_macro(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    eps = 1e-9
    f1_scores: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_scores.append(f1)
    return float(sum(f1_scores) / max(len(f1_scores), 1))


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    idx = {label: i for i, label in enumerate(labels)}
    mat = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            mat[idx[t]][idx[p]] += 1
    return mat


def topk_hit(true_label: str, topk: list[dict[str, float | str]]) -> bool:
    return any(item.get("label") == true_label for item in topk)
