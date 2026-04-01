from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

from src.mock_engine import DISCLAIMER, mock_infer


class InferenceError(RuntimeError):
    pass


class DataInspectionError(InferenceError):
    pass


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S | re.I)
    if fenced:
        return fenced.group(1)

    obj = re.search(r"(\{.*\})", stripped, flags=re.S)
    if obj:
        return obj.group(1)

    raise InferenceError("模型响应中未找到 JSON 对象。")


def _normalize_label(raw_label: str, labels: list[str]) -> str:
    if raw_label in labels:
        return raw_label

    canonical = re.sub(r"[\s/\-]+", "_", str(raw_label).strip()).lower()
    for label in labels:
        if label.lower() == canonical:
            return label

    raise InferenceError(f"诊断标签不在候选集合内: {raw_label}")


def _normalize_top3(
    *,
    parsed: dict[str, Any],
    labels: list[str],
    primary: str,
    confidence: float,
) -> list[dict[str, Any]]:
    raw = parsed.get("top3_candidates")
    normalized: list[dict[str, Any]] = []
    used: set[str] = set()

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue

            raw_label = item.get("label", item.get("primary_diagnosis", ""))
            if not raw_label:
                continue

            try:
                label = _normalize_label(str(raw_label), labels)
            except InferenceError:
                continue

            if label in used:
                continue

            raw_score = item.get("score", item.get("confidence", item.get("probability", 0.0)))
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0
            score = max(0.0, min(1.0, score))

            normalized.append({"label": label, "score": score})
            used.add(label)

    if primary not in used:
        normalized.insert(0, {"label": primary, "score": float(confidence)})
        used.add(primary)

    for label in labels:
        if len(normalized) >= 3:
            break
        if label in used:
            continue
        normalized.append({"label": label, "score": 0.0})
        used.add(label)

    normalized = normalized[:3]
    normalized.sort(key=lambda x: x["score"], reverse=True)

    # Force top-1 to align with primary_diagnosis
    if normalized[0]["label"] != primary:
        normalized = [item for item in normalized if item["label"] != primary]
        normalized.insert(0, {"label": primary, "score": float(confidence)})
        normalized = normalized[:3]

    for item in normalized:
        item["score"] = round(float(item["score"]), 4)

    return normalized


def _normalize_result(parsed: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    if "primary_diagnosis" not in parsed:
        raise InferenceError("模型响应缺少 primary_diagnosis 字段。")

    primary = _normalize_label(str(parsed["primary_diagnosis"]), labels)

    try:
        confidence = float(parsed.get("confidence", 0.6))
    except (TypeError, ValueError) as exc:
        raise InferenceError("confidence 不是有效数值。") from exc
    confidence = max(0.0, min(1.0, confidence))

    result = {
        "primary_diagnosis": primary,
        "confidence": round(confidence, 2),
        "source": "qwen_vl_api",
        "mock_result": False,
        "note": DISCLAIMER,
        "top3_candidates": _normalize_top3(
            parsed=parsed,
            labels=labels,
            primary=primary,
            confidence=confidence,
        ),
    }
    return result


def _build_prompt(symptom_text: str, labels: list[str]) -> str:
    label_text = ", ".join(labels)
    user_text = symptom_text.strip() or "(未提供症状描述)"
    return f"""
你是皮肤疾病筛查助手。请基于输入图像与结构化症状信息给出"初步筛查"结果。
只允许从下列标签中选择：{label_text}

结构化症状信息：{user_text}
(格式：病程，部位，瘙痒程度，疼痛程度，诱因，皮损形态，复发情况，年龄段)

你必须只输出一个 JSON 对象，且必须包含以下字段：
{{
  "primary_diagnosis": "<标签之一>",
  "confidence": <0到1之间的小数>,
  "top3_candidates": [
    {{"label": "<标签之一>", "score": <0到1之间的小数>}},
    {{"label": "<标签之一>", "score": <0到1之间的小数>}},
    {{"label": "<标签之一>", "score": <0到1之间的小数>}}
  ],
  "note": "初步筛查结果，非临床诊断结论"
}}

严格要求：
1) 不要输出 markdown，不要输出解释性文字。
2) top3_candidates 必须恰好3项，label 不可重复。
3) top3_candidates[0].label 必须等于 primary_diagnosis。
4) confidence 和 score 必须在 0.00 到 1.00 区间。
5) 若不确定也必须返回最可能的3个标签。
""".strip()



def _call_qwen_vl(
    api_key: str,
    model: str,
    base_url: str,
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    timeout: int,
) -> str:
    try:
        import requests
    except ImportError as exc:
        raise InferenceError("缺少 requests 依赖，无法调用 Qwen-VL API。") from exc

    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    image_data_uri = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是皮肤疾病初筛助手。只返回合法 JSON，不要返回任何额外文本。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            },
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        body = response.text[:300]
        try:
            err_json = response.json()
        except Exception:
            err_json = {}

        err_code = err_json.get("error", {}).get("code") if isinstance(err_json, dict) else None
        if err_code == "data_inspection_failed":
            raise DataInspectionError("Qwen-VL 内容审核拦截（data_inspection_failed）。")
        raise InferenceError(f"Qwen-VL API 错误: HTTP {response.status_code} - {body}")

    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise InferenceError(f"Qwen-VL 响应结构异常: {json.dumps(data)[:300]}") from exc

    if isinstance(content, list):
        texts = [p.get("text", "") for p in content if isinstance(p, dict)]
        merged = "\n".join(t for t in texts if t).strip()
        if not merged:
            raise InferenceError("Qwen-VL 返回为空。")
        return merged

    if not isinstance(content, str) or not content.strip():
        raise InferenceError("Qwen-VL 返回为空。")
    return content.strip()


def _run_local_fallback(
    *,
    image_bytes: bytes,
    symptom_text: str,
    labels: list[str],
    errors: list[str],
) -> tuple[dict[str, Any], list[str]]:
    artifacts_dir = os.getenv("LOCAL_MODEL_DIR", "artifacts")

    try:
        from core.local_hybrid import local_hybrid_infer

        hybrid = local_hybrid_infer(
            image_bytes=image_bytes,
            symptom_text=symptom_text,
            labels=labels,
            artifacts_dir=artifacts_dir,
        )
        errors.append(f"已使用 local_hybrid（artifacts_dir={artifacts_dir}）。")
        return hybrid, errors
    except Exception as exc:
        errors.append(f"local_hybrid 不可用，回退 local_mock：{exc}")
        return mock_infer(symptom_text=symptom_text, labels=labels), errors


def infer_with_qwen_or_mock(
    *,
    image_bytes: bytes,
    mime_type: str,
    symptom_text: str,
    labels: list[str],
    api_key: str | None,
    model: str = "qwen-vl-max-latest",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout: int = 40,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not api_key:
        errors.append("未检测到 QWEN_API_KEY，自动使用本地智能推理。")
        return _run_local_fallback(
            image_bytes=image_bytes,
            symptom_text=symptom_text,
            labels=labels,
            errors=errors,
        )

    prompt = _build_prompt(symptom_text=symptom_text, labels=labels)

    for attempt in (1, 2):
        try:
            text = _call_qwen_vl(
                api_key=api_key,
                model=model,
                base_url=base_url,
                prompt=prompt,
                image_bytes=image_bytes,
                mime_type=mime_type,
                timeout=timeout,
            )
            parsed = json.loads(_extract_json_text(text))
            return _normalize_result(parsed=parsed, labels=labels), errors
        except DataInspectionError as exc:
            errors.append(f"第{attempt}次 Qwen-VL 调用失败: {exc}")
            break
        except Exception as exc:
            errors.append(f"第{attempt}次 Qwen-VL 调用失败: {exc}")

    errors.append("Qwen-VL 调用未成功，已切换本地智能推理。")
    return _run_local_fallback(
        image_bytes=image_bytes,
        symptom_text=symptom_text,
        labels=labels,
        errors=errors,
    )
