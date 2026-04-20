from __future__ import annotations

import argparse
import json
import os
import sys
import time
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import inference as inf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick smoke test for Qwen-VL API availability.")
    parser.add_argument("--api-key", type=str, default=os.getenv("QWEN_API_KEY", ""))
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("QWEN_MODEL", inf.PROVIDER_DEFAULTS[inf.PROVIDER_QWEN]["model"]),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("QWEN_BASE_URL", inf.PROVIDER_DEFAULTS[inf.PROVIDER_QWEN]["base_url"]),
    )
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def tiny_png_bytes(size: int = 32) -> bytes:
    # Qwen-VL requires width/height > 10, so use a small 32x32 image.
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Pillow is required for smoke test image generation.") from exc

    img = Image.new("RGB", (size, size), (240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        print("FAIL: QWEN_API_KEY missing (or --api-key not provided).")
        raise SystemExit(2)

    prompt = (
        "你是API连通性测试助手。只返回一个JSON对象，不要额外文本："
        '{"status":"ok","tag":"qwen_api_smoke_test"}'
    )

    image_bytes = tiny_png_bytes()
    t0 = time.perf_counter()
    try:
        raw = inf._call_provider(
            provider=inf.PROVIDER_QWEN,
            api_key=args.api_key,
            model=args.model,
            base_url=args.base_url,
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type="image/png",
            timeout=args.timeout,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: API request error -> {exc}")
        raise SystemExit(1)

    snippet = raw.strip().replace("\n", " ")
    if len(snippet) > 220:
        snippet = snippet[:220] + "..."

    parsed_ok = False
    try:
        obj = json.loads(inf._extract_json_text(raw))
        parsed_ok = isinstance(obj, dict)
    except Exception:
        parsed_ok = False

    print("PASS: Qwen API reachable.")
    print(f"Model: {args.model}")
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"JSON parse: {'ok' if parsed_ok else 'failed'}")
    print(f"Raw snippet: {snippet}")


if __name__ == "__main__":
    main()
