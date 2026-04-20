from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import inference as inf
from core.local_hybrid import f1_macro, local_hybrid_infer, synthetic_symptom_for_label, topk_hit
from core.mock_engine import DISCLAIMER, load_class_labels, mock_infer, resolve_dataset_root

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FAILED_LABEL = "__FAILED__"
PROVIDER = inf.PROVIDER_QWEN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment suite for exp2-exp5.")
    parser.add_argument("--dataset-root", type=str, default="Dataset/archive/SkinDisease")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/multi_model_compare/efficientnet_b0")
    parser.add_argument("--output-dir", type=str, default="artifacts/experiments")
    parser.add_argument("--max-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=40)
    parser.add_argument("--api-key", type=str, default=os.getenv("QWEN_API_KEY", ""))
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("QWEN_MODEL", inf.PROVIDER_DEFAULTS[PROVIDER]["model"]),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("QWEN_BASE_URL", inf.PROVIDER_DEFAULTS[PROVIDER]["base_url"]),
    )
    return parser.parse_args()


def _detect_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _collect_samples(dataset_root: Path, max_per_class: int, seed: int) -> list[tuple[Path, str]]:
    rng = random.Random(seed)
    samples: list[tuple[Path, str]] = []
    test_root = dataset_root / "test"
    for class_dir in sorted([p for p in test_root.iterdir() if p.is_dir()]):
        files = [p for p in sorted(class_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
        if not files:
            continue
        if len(files) <= max_per_class:
            picked = files
        else:
            picked = sorted(rng.sample(files, max_per_class))
        for p in picked:
            samples.append((p, class_dir.name))
    return samples


def _round(v: float, nd: int = 4) -> float:
    return round(float(v), nd)


def _result_top3(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not result:
        return []
    top3 = result.get("top3_candidates")
    if isinstance(top3, list) and top3:
        return top3
    primary = result.get("primary_diagnosis")
    confidence = float(result.get("confidence", 0.0))
    if isinstance(primary, str) and primary:
        return [{"label": primary, "score": confidence}]
    return []


def _safe_pred_label(result: dict[str, Any] | None) -> str:
    if not result:
        return FAILED_LABEL
    pred = result.get("primary_diagnosis")
    if isinstance(pred, str) and pred.strip():
        return pred.strip()
    return FAILED_LABEL


def _compute_metrics(
    *,
    y_true: list[str],
    y_pred: list[str],
    top3_list: list[list[dict[str, Any]]],
    labels: list[str],
    latencies_ms: list[float],
    success_count: int,
) -> dict[str, Any]:
    total = len(y_true)
    correct_top1 = sum(int(t == p) for t, p in zip(y_true, y_pred))
    correct_top3 = sum(int(topk_hit(t, tk)) for t, tk in zip(y_true, top3_list))
    top1 = correct_top1 / max(total, 1)
    top3 = correct_top3 / max(total, 1)
    macro = f1_macro(y_true, y_pred, labels)
    success_rate = success_count / max(total, 1)
    failure_rate = 1.0 - success_rate
    avg_latency = sum(latencies_ms) / max(len(latencies_ms), 1)
    return {
        "num_samples": total,
        "top1": _round(top1),
        "top3": _round(top3),
        "macro_f1": _round(macro),
        "success_rate": _round(success_rate),
        "failure_rate": _round(failure_rate),
        "avg_latency_ms": _round(avg_latency, nd=2),
    }


def _schema_flags(result: dict[str, Any] | None, labels: list[str]) -> dict[str, int]:
    if not result:
        return {
            "schema_pass": 0,
            "label_in_set": 0,
            "top3_complete": 0,
        }

    primary = result.get("primary_diagnosis")
    conf = result.get("confidence")
    top3 = result.get("top3_candidates")

    label_in_set = int(isinstance(primary, str) and primary in labels)
    conf_ok = int(isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0)
    top3_ok = 0
    if isinstance(top3, list) and len(top3) == 3:
        seen: set[str] = set()
        ok = True
        for item in top3:
            if not isinstance(item, dict):
                ok = False
                break
            lb = item.get("label")
            sc = item.get("score")
            if not isinstance(lb, str) or lb not in labels or lb in seen:
                ok = False
                break
            if not isinstance(sc, (int, float)) or not (0.0 <= float(sc) <= 1.0):
                ok = False
                break
            seen.add(lb)
        top3_ok = int(ok)

    schema_pass = int(label_in_set and conf_ok and top3_ok)
    return {
        "schema_pass": schema_pass,
        "label_in_set": label_in_set,
        "top3_complete": top3_ok,
    }


def _normalize_key(s: str) -> str:
    return re.sub(r"[\s/_-]+", "", s.lower())


def _map_label_loose(raw: Any, labels: list[str]) -> str | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if text in labels:
        return text
    key = _normalize_key(text)
    for lb in labels:
        if _normalize_key(lb) == key:
            return lb
    return None


def _extract_top_labels_from_text(text: str, labels: list[str]) -> list[str]:
    lowered = text.lower()
    hits: list[tuple[int, str]] = []
    for lb in labels:
        candidates = {
            lb.lower(),
            lb.lower().replace("_", " "),
            lb.lower().replace("_", ""),
        }
        idx_list = [lowered.find(c) for c in candidates if c]
        idx_list = [i for i in idx_list if i >= 0]
        if idx_list:
            hits.append((min(idx_list), lb))
    hits.sort(key=lambda x: x[0])
    ordered = [lb for _, lb in hits]
    dedup: list[str] = []
    for lb in ordered:
        if lb not in dedup:
            dedup.append(lb)
    return dedup


def _extract_first_confidence(text: str) -> float | None:
    for m in re.finditer(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", text):
        try:
            val = float(m.group(0))
        except ValueError:
            continue
        if 0.0 <= val <= 1.0:
            return val
    return None


def _parse_response(
    *,
    text: str,
    labels: list[str],
    strict_json: bool,
    source: str,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    flags = {
        "json_valid": 0,
        "schema_pass": 0,
        "label_in_set": 0,
        "top3_complete": 0,
    }

    parsed_obj: dict[str, Any] | None = None
    try:
        parsed_obj = json.loads(inf._extract_json_text(text))
        flags["json_valid"] = 1
    except Exception:
        parsed_obj = None

    if parsed_obj is not None:
        try:
            normalized = inf._normalize_result(parsed=parsed_obj, labels=labels, source=source)
            schema = _schema_flags(normalized, labels)
            flags.update(schema)
            return normalized, flags
        except Exception:
            if strict_json:
                return None, flags
    elif strict_json:
        return None, flags

    candidate_labels: list[str] = []
    if isinstance(parsed_obj, dict):
        primary = _map_label_loose(parsed_obj.get("primary_diagnosis"), labels)
        if primary:
            candidate_labels.append(primary)

        raw_top3 = parsed_obj.get("top3_candidates")
        if isinstance(raw_top3, list):
            for item in raw_top3:
                if not isinstance(item, dict):
                    continue
                lb = _map_label_loose(item.get("label"), labels)
                if lb:
                    candidate_labels.append(lb)

    candidate_labels.extend(_extract_top_labels_from_text(text, labels))

    dedup: list[str] = []
    for lb in candidate_labels:
        if lb not in dedup:
            dedup.append(lb)

    if not dedup:
        idx = abs(hash(text)) % len(labels)
        dedup = [labels[idx]]

    confidence = _extract_first_confidence(text)
    if confidence is None:
        confidence = 0.5

    top3: list[dict[str, Any]] = []
    for i, lb in enumerate(dedup[:3]):
        if i == 0:
            score = confidence
        else:
            score = max(confidence - 0.15 * i, 0.0)
        top3.append({"label": lb, "score": _round(score)})

    for lb in labels:
        if len(top3) >= 3:
            break
        if lb in [x["label"] for x in top3]:
            continue
        top3.append({"label": lb, "score": 0.0})

    top3 = top3[:3]
    result = {
        "primary_diagnosis": top3[0]["label"],
        "confidence": _round(top3[0]["score"], nd=2),
        "source": source,
        "mock_result": False,
        "note": DISCLAIMER,
        "top3_candidates": top3,
    }
    schema = _schema_flags(result, labels)
    flags.update(schema)
    return result, flags


def _build_prompt(prompt_style: str, symptom_text: str, labels: list[str]) -> str:
    if prompt_style == "normalized":
        return inf._build_prompt(symptom_text=symptom_text, labels=labels)
    label_text = ", ".join(labels)
    return (
        "Please analyze the image and symptom text for a likely skin disease label.\n"
        f"Candidate labels: {label_text}\n"
        f"Symptoms: {symptom_text or 'none'}\n"
        "Return your best judgment."
    )


def _call_api_once(
    *,
    image_bytes: bytes,
    mime_type: str,
    symptom_text: str,
    labels: list[str],
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
    prompt_style: str,
    strict_json: bool,
) -> tuple[dict[str, Any] | None, dict[str, int], str | None]:
    prompt = _build_prompt(prompt_style=prompt_style, symptom_text=symptom_text, labels=labels)
    source = inf._source_name(PROVIDER)
    try:
        text = inf._call_provider(
            provider=PROVIDER,
            api_key=api_key,
            model=model,
            base_url=base_url,
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return None, {"json_valid": 0, "schema_pass": 0, "label_in_set": 0, "top3_complete": 0}, str(exc)

    result, flags = _parse_response(text=text, labels=labels, strict_json=strict_json, source=source)
    if result is None:
        return None, flags, "parse_failed"
    return result, flags, None


def _run_chain_once(
    *,
    image_bytes: bytes,
    mime_type: str,
    symptom_text: str,
    labels: list[str],
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
    prompt_style: str,
    strict_json: bool,
    force_api_failure: bool,
    enable_local_hybrid: bool,
    enable_local_mock: bool,
    artifacts_dir: str | Path,
) -> tuple[dict[str, Any] | None, str | None]:
    if not force_api_failure and api_key:
        result, _, err = _call_api_once(
            image_bytes=image_bytes,
            mime_type=mime_type,
            symptom_text=symptom_text,
            labels=labels,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            prompt_style=prompt_style,
            strict_json=strict_json,
        )
        if result is not None:
            return result, None
        api_error = err or "api_failed"
    else:
        api_error = "api_forced_failed" if force_api_failure else "api_key_missing"

    if enable_local_hybrid:
        try:
            hybrid = local_hybrid_infer(
                image_bytes=image_bytes,
                symptom_text=symptom_text,
                labels=labels,
                artifacts_dir=artifacts_dir,
                mode="hybrid",
            )
            return hybrid, None
        except Exception as exc:  # noqa: BLE001
            api_error = f"{api_error}; local_hybrid_failed={exc}"

    if enable_local_mock:
        try:
            return mock_infer(symptom_text=symptom_text, labels=labels), None
        except Exception as exc:  # noqa: BLE001
            return None, f"{api_error}; local_mock_failed={exc}"

    return None, api_error


def _save_rows(name: str, rows: list[dict[str, Any]], output_dir: Path, metadata: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}.json"
    csv_path = output_dir / f"{name}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": rows}, f, ensure_ascii=False, indent=2)

    keys: list[str] = []
    key_set: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in key_set:
                keys.append(k)
                key_set.add(k)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return json_path, csv_path


def _run_exp2(
    *,
    samples: list[tuple[Path, str]],
    labels: list[str],
    artifacts_dir: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for setting, mode, use_text in [
        ("image_only", "image_only", False),
        ("image_text_fusion", "hybrid", True),
    ]:
        # Warmup once per setting to reduce one-time model load impact on latency.
        if samples:
            warm_path, warm_label = samples[0]
            warm_symptom = synthetic_symptom_for_label(warm_label) if use_text else ""
            warm_image_bytes = warm_path.read_bytes()
            try:
                _ = local_hybrid_infer(
                    image_bytes=warm_image_bytes,
                    symptom_text=warm_symptom,
                    labels=labels,
                    artifacts_dir=artifacts_dir,
                    mode=mode,
                )
            except Exception:
                pass

        y_true: list[str] = []
        y_pred: list[str] = []
        top3_list: list[list[dict[str, Any]]] = []
        latencies_ms: list[float] = []
        success_count = 0

        for path, true_label in samples:
            symptom = synthetic_symptom_for_label(true_label) if use_text else ""
            image_bytes = path.read_bytes()
            t0 = time.perf_counter()
            try:
                result = local_hybrid_infer(
                    image_bytes=image_bytes,
                    symptom_text=symptom,
                    labels=labels,
                    artifacts_dir=artifacts_dir,
                    mode=mode,
                )
                success_count += 1
            except Exception:  # noqa: BLE001
                result = None
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            y_true.append(true_label)
            y_pred.append(_safe_pred_label(result))
            top3_list.append(_result_top3(result))

        row = {"experiment": "exp2_multimodal", "setting": setting}
        row.update(_compute_metrics(y_true=y_true, y_pred=y_pred, top3_list=top3_list, labels=labels, latencies_ms=latencies_ms, success_count=success_count))
        rows.append(row)
    return rows


def _run_exp3(
    *,
    samples: list[tuple[Path, str]],
    labels: list[str],
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not api_key:
        return [
            {
                "experiment": "exp3_prompt_json",
                "setting": "all",
                "skipped_reason": "QWEN_API_KEY missing",
            }
        ]

    for prompt_style, strict_json in [
        ("raw", False),
        ("raw", True),
        ("normalized", False),
        ("normalized", True),
    ]:
        setting = f"{prompt_style}_prompt + {'json_constraint' if strict_json else 'no_json_constraint'}"
        y_true: list[str] = []
        y_pred: list[str] = []
        top3_list: list[list[dict[str, Any]]] = []
        latencies_ms: list[float] = []
        success_count = 0
        json_valid = 0
        schema_pass = 0
        label_in_set = 0
        top3_complete = 0

        for path, true_label in samples:
            symptom = synthetic_symptom_for_label(true_label)
            image_bytes = path.read_bytes()
            t0 = time.perf_counter()
            result, flags, _ = _call_api_once(
                image_bytes=image_bytes,
                mime_type=_detect_mime(path),
                symptom_text=symptom,
                labels=labels,
                api_key=api_key,
                model=model,
                base_url=base_url,
                timeout=timeout,
                prompt_style=prompt_style,
                strict_json=strict_json,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            if result is not None:
                success_count += 1

            json_valid += int(flags.get("json_valid", 0))
            schema_pass += int(flags.get("schema_pass", 0))
            label_in_set += int(flags.get("label_in_set", 0))
            top3_complete += int(flags.get("top3_complete", 0))

            y_true.append(true_label)
            y_pred.append(_safe_pred_label(result))
            top3_list.append(_result_top3(result))

        total = len(samples)
        row = {"experiment": "exp3_prompt_json", "setting": setting}
        row.update(_compute_metrics(y_true=y_true, y_pred=y_pred, top3_list=top3_list, labels=labels, latencies_ms=latencies_ms, success_count=success_count))
        row["json_valid_rate"] = _round(json_valid / max(total, 1))
        row["schema_pass_rate"] = _round(schema_pass / max(total, 1))
        row["label_in_set_rate"] = _round(label_in_set / max(total, 1))
        row["top3_complete_rate"] = _round(top3_complete / max(total, 1))
        rows.append(row)
    return rows


def _route_match(expected: str, actual: str | None) -> bool:
    if not actual:
        return False
    if expected == "api":
        return actual.endswith("_api")
    return actual == expected


def _run_exp4(
    *,
    samples: list[tuple[Path, str]],
    labels: list[str],
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
    artifacts_dir: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not api_key:
        return [
            {
                "experiment": "exp4_robustness",
                "scenario": "all",
                "skipped_reason": "QWEN_API_KEY missing",
            }
        ]

    missing_artifacts = str(Path(artifacts_dir) / "_missing_forced")
    scenarios = [
        {
            "scenario": "A_api_normal",
            "expected_route": "api",
            "force_api_failure": False,
            "artifacts_dir": artifacts_dir,
        },
        {
            "scenario": "B_api_fail_to_local_hybrid",
            "expected_route": "local_hybrid",
            "force_api_failure": True,
            "artifacts_dir": artifacts_dir,
        },
        {
            "scenario": "C_api_and_local_hybrid_fail_to_local_mock",
            "expected_route": "local_mock",
            "force_api_failure": True,
            "artifacts_dir": missing_artifacts,
        },
    ]

    for cfg in scenarios:
        y_true: list[str] = []
        y_pred: list[str] = []
        top3_list: list[list[dict[str, Any]]] = []
        latencies_ms: list[float] = []
        success_count = 0
        route_correct = 0
        src_counter: Counter[str] = Counter()

        for path, true_label in samples:
            symptom = synthetic_symptom_for_label(true_label)
            image_bytes = path.read_bytes()
            t0 = time.perf_counter()
            result, _ = _run_chain_once(
                image_bytes=image_bytes,
                mime_type=_detect_mime(path),
                symptom_text=symptom,
                labels=labels,
                api_key=api_key,
                model=model,
                base_url=base_url,
                timeout=timeout,
                prompt_style="normalized",
                strict_json=True,
                force_api_failure=cfg["force_api_failure"],
                enable_local_hybrid=True,
                enable_local_mock=True,
                artifacts_dir=cfg["artifacts_dir"],
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            source = result.get("source") if isinstance(result, dict) else None
            src_counter[str(source or "failed")] += 1
            route_correct += int(_route_match(cfg["expected_route"], source))
            if result is not None:
                success_count += 1

            y_true.append(true_label)
            y_pred.append(_safe_pred_label(result))
            top3_list.append(_result_top3(result))

        row = {
            "experiment": "exp4_robustness",
            "scenario": cfg["scenario"],
            "expected_route": cfg["expected_route"],
            "route_correct_rate": _round(route_correct / max(len(samples), 1)),
            "source_api": src_counter.get("qwen_vl_api", 0),
            "source_local_hybrid": src_counter.get("local_hybrid", 0),
            "source_local_mock": src_counter.get("local_mock", 0),
            "source_failed": src_counter.get("failed", 0),
        }
        row.update(_compute_metrics(y_true=y_true, y_pred=y_pred, top3_list=top3_list, labels=labels, latencies_ms=latencies_ms, success_count=success_count))
        rows.append(row)
    return rows


def _run_exp5(
    *,
    samples: list[tuple[Path, str]],
    labels: list[str],
    api_key: str,
    model: str,
    base_url: str,
    timeout: int,
    artifacts_dir: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not api_key:
        return [
            {
                "experiment": "exp5_ablation",
                "ablation_item": "all",
                "skipped_reason": "QWEN_API_KEY missing",
            }
        ]

    missing_artifacts = str(Path(artifacts_dir) / "_missing_forced")
    ablations = [
        {
            "ablation_item": "baseline_full",
            "prompt_style": "normalized",
            "strict_json": True,
            "use_text": True,
            "force_api_failure": False,
            "enable_local_hybrid": True,
            "enable_local_mock": True,
            "artifacts_dir": artifacts_dir,
            "degradation_policy": "api->local_hybrid->local_mock",
        },
        {
            "ablation_item": "remove_text_input",
            "prompt_style": "normalized",
            "strict_json": True,
            "use_text": False,
            "force_api_failure": False,
            "enable_local_hybrid": True,
            "enable_local_mock": True,
            "artifacts_dir": artifacts_dir,
            "degradation_policy": "api->local_hybrid->local_mock",
        },
        {
            "ablation_item": "remove_prompt_template",
            "prompt_style": "raw",
            "strict_json": True,
            "use_text": True,
            "force_api_failure": False,
            "enable_local_hybrid": True,
            "enable_local_mock": True,
            "artifacts_dir": artifacts_dir,
            "degradation_policy": "api->local_hybrid->local_mock",
        },
        {
            "ablation_item": "remove_json_constraint",
            "prompt_style": "normalized",
            "strict_json": False,
            "use_text": True,
            "force_api_failure": False,
            "enable_local_hybrid": True,
            "enable_local_mock": True,
            "artifacts_dir": artifacts_dir,
            "degradation_policy": "api->local_hybrid->local_mock",
        },
        {
            "ablation_item": "remove_second_layer_fallback",
            "prompt_style": "normalized",
            "strict_json": True,
            "use_text": True,
            "force_api_failure": True,
            "enable_local_hybrid": False,
            "enable_local_mock": True,
            "artifacts_dir": artifacts_dir,
            "degradation_policy": "api_fail->local_mock",
        },
        {
            "ablation_item": "remove_third_layer_fallback",
            "prompt_style": "normalized",
            "strict_json": True,
            "use_text": True,
            "force_api_failure": True,
            "enable_local_hybrid": True,
            "enable_local_mock": False,
            "artifacts_dir": missing_artifacts,
            "degradation_policy": "api_fail+local_hybrid_fail->failed",
        },
    ]

    for cfg in ablations:
        y_true: list[str] = []
        y_pred: list[str] = []
        top3_list: list[list[dict[str, Any]]] = []
        latencies_ms: list[float] = []
        success_count = 0
        src_counter: Counter[str] = Counter()

        for path, true_label in samples:
            symptom = synthetic_symptom_for_label(true_label) if cfg["use_text"] else ""
            image_bytes = path.read_bytes()
            t0 = time.perf_counter()
            result, _ = _run_chain_once(
                image_bytes=image_bytes,
                mime_type=_detect_mime(path),
                symptom_text=symptom,
                labels=labels,
                api_key=api_key,
                model=model,
                base_url=base_url,
                timeout=timeout,
                prompt_style=cfg["prompt_style"],
                strict_json=cfg["strict_json"],
                force_api_failure=cfg["force_api_failure"],
                enable_local_hybrid=cfg["enable_local_hybrid"],
                enable_local_mock=cfg["enable_local_mock"],
                artifacts_dir=cfg["artifacts_dir"],
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            source = result.get("source") if isinstance(result, dict) else None
            src_counter[str(source or "failed")] += 1

            if result is not None:
                success_count += 1

            y_true.append(true_label)
            y_pred.append(_safe_pred_label(result))
            top3_list.append(_result_top3(result))

        row = {
            "experiment": "exp5_ablation",
            "ablation_item": cfg["ablation_item"],
            "degradation_policy": cfg["degradation_policy"],
            "source_api": src_counter.get("qwen_vl_api", 0),
            "source_local_hybrid": src_counter.get("local_hybrid", 0),
            "source_local_mock": src_counter.get("local_mock", 0),
            "source_failed": src_counter.get("failed", 0),
        }
        row.update(_compute_metrics(y_true=y_true, y_pred=y_pred, top3_list=top3_list, labels=labels, latencies_ms=latencies_ms, success_count=success_count))
        rows.append(row)
    return rows


def _build_summary(
    *,
    exp2_rows: list[dict[str, Any]],
    exp3_rows: list[dict[str, Any]],
    exp4_rows: list[dict[str, Any]],
    exp5_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []

    def pick_identifier(row: dict[str, Any]) -> str:
        for key in ("setting", "scenario", "ablation_item"):
            if key in row:
                return str(row[key])
        return "row"

    for exp_name, rows in [
        ("exp2_multimodal", exp2_rows),
        ("exp3_prompt_json", exp3_rows),
        ("exp4_robustness", exp4_rows),
        ("exp5_ablation", exp5_rows),
    ]:
        for row in rows:
            summary.append(
                {
                    "experiment": exp_name,
                    "item": pick_identifier(row),
                    "top1": row.get("top1"),
                    "top3": row.get("top3"),
                    "macro_f1": row.get("macro_f1"),
                    "success_rate": row.get("success_rate"),
                    "failure_rate": row.get("failure_rate"),
                    "avg_latency_ms": row.get("avg_latency_ms"),
                    "route_correct_rate": row.get("route_correct_rate"),
                    "json_valid_rate": row.get("json_valid_rate"),
                    "schema_pass_rate": row.get("schema_pass_rate"),
                    "label_in_set_rate": row.get("label_in_set_rate"),
                    "top3_complete_rate": row.get("top3_complete_rate"),
                    "skipped_reason": row.get("skipped_reason"),
                }
            )

    compare_csv = ROOT / "artifacts" / "multi_model_compare" / "compare_summary.csv"
    summary.append(
        {
            "experiment": "exp1_model_compare_reference",
            "item": "existing_result",
            "reference_csv": str(compare_csv.resolve()) if compare_csv.exists() else str(compare_csv),
            "skipped_reason": None if compare_csv.exists() else "reference_compare_summary_missing",
        }
    )
    return summary


def main() -> None:
    args = parse_args()

    dataset_root = resolve_dataset_root(args.dataset_root)
    labels = load_class_labels(dataset_root)
    samples = _collect_samples(dataset_root=Path(dataset_root), max_per_class=args.max_per_class, seed=args.seed)
    if not samples:
        raise RuntimeError("No test samples found for experiment suite.")

    output_dir = Path(args.output_dir)
    metadata = {
        "dataset_root": str(dataset_root),
        "artifacts_dir": str(Path(args.artifacts_dir)),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "max_per_class": args.max_per_class,
        "num_samples": len(samples),
        "provider": PROVIDER,
        "api_enabled": bool(args.api_key),
        "model": args.model,
        "base_url": args.base_url,
    }

    exp2_rows = _run_exp2(samples=samples, labels=labels, artifacts_dir=args.artifacts_dir)
    exp3_rows = _run_exp3(
        samples=samples,
        labels=labels,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    exp4_rows = _run_exp4(
        samples=samples,
        labels=labels,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        artifacts_dir=args.artifacts_dir,
    )
    exp5_rows = _run_exp5(
        samples=samples,
        labels=labels,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        artifacts_dir=args.artifacts_dir,
    )

    saved: list[tuple[Path, Path]] = []
    saved.append(_save_rows("exp2_multimodal", exp2_rows, output_dir, metadata))
    saved.append(_save_rows("exp3_prompt_json", exp3_rows, output_dir, metadata))
    saved.append(_save_rows("exp4_robustness", exp4_rows, output_dir, metadata))
    saved.append(_save_rows("exp5_ablation", exp5_rows, output_dir, metadata))

    summary_rows = _build_summary(
        exp2_rows=exp2_rows,
        exp3_rows=exp3_rows,
        exp4_rows=exp4_rows,
        exp5_rows=exp5_rows,
    )
    summary_json, summary_csv = _save_rows("experiment_summary", summary_rows, output_dir, metadata)

    print("Experiment suite completed.")
    print(f"- Samples: {len(samples)} (max_per_class={args.max_per_class}, seed={args.seed})")
    print(f"- API enabled: {bool(args.api_key)}")
    for json_path, csv_path in saved:
        print(f"- {json_path.resolve()}")
        print(f"- {csv_path.resolve()}")
    print(f"- {summary_json.resolve()}")
    print(f"- {summary_csv.resolve()}")

    if not args.api_key:
        print("API-related experiments were skipped because QWEN_API_KEY is missing.")


if __name__ == "__main__":
    main()
