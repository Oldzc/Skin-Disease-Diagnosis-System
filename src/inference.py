from __future__ import annotations

import base64
import json
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


def _normalize_result(parsed: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    if "primary_diagnosis" not in parsed:
        raise InferenceError("模型响应缺少 primary_diagnosis 字段。")

    primary = _normalize_label(str(parsed["primary_diagnosis"]), labels)

    try:
        confidence = float(parsed.get("confidence", 0.6))
    except (TypeError, ValueError) as exc:
        raise InferenceError("confidence 不是有效数值。") from exc
    confidence = max(0.0, min(1.0, confidence))

    return {
        "primary_diagnosis": primary,
        "confidence": round(confidence, 2),
        "source": "qwen_vl_api",
        "mock_result": False,
        "note": DISCLAIMER,
    }


def _build_prompt(symptom_text: str, labels: list[str]) -> str:
    label_text = ", ".join(labels)
    user_text = symptom_text.strip() or "（未提供症状描述）"
    return f"""
你是皮肤疾病筛查助手。请基于输入图像与症状文本给出“初步筛查”结果。
只允许从下列标签中选择一个 primary_diagnosis：{label_text}

症状文本：{user_text}

你必须只输出一个 JSON 对象，且只能包含这四个字段：
{{
  "primary_diagnosis": "<标签之一>",
  "confidence": <0到1之间的小数>,
  "note": "初步筛查结果，非临床诊断结论"
}}

要求：
1) 不要输出 markdown，不要输出解释性文字。
2) confidence 必须在 0.00 到 1.00 区间。
3) 若不确定也必须返回最可能的一个标签。
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
        except Exception:  # noqa: BLE001
            err_json = {}

        err_code = (
            err_json.get("error", {}).get("code")
            if isinstance(err_json, dict)
            else None
        )
        if err_code == "data_inspection_failed":
            raise DataInspectionError(
                "Qwen-VL 内容审核拦截（data_inspection_failed）。"
            )
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
        errors.append("未检测到 QWEN_API_KEY，自动使用本地 mock。")
        return mock_infer(symptom_text=symptom_text, labels=labels), errors

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
            errors.append(
                f"第{attempt}次 Qwen-VL 调用失败: {exc} 已自动切换本地 Mock。"
            )
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"第{attempt}次 Qwen-VL 调用失败: {exc}")

    errors.append("Qwen-VL 调用未成功，已切换本地 mock。")
    return mock_infer(symptom_text=symptom_text, labels=labels), errors
