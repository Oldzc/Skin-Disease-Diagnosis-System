from __future__ import annotations

import copy
import io
import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

HISTORY_FILE = Path(__file__).parent / "history.json"

from core.inference import infer_with_qwen_or_mock
from core.local_hybrid import local_hybrid_artifacts_available
from src.mock_engine import load_class_labels, resolve_dataset_root

NORMALIZED_SIZE = 512

DISEASE_LABEL_ZH: dict[str, str] = {
    "Acne": "痤疮",
    "Actinic_Keratosis": "日光性角化病",
    "Benign_tumors": "良性肿瘤",
    "Bullous": "大疱性皮肤病",
    "Candidiasis": "念珠菌病",
    "DrugEruption": "药疹",
    "Eczema": "湿疹",
    "Infestations_Bites": "虫咬/寄生虫相关皮损",
    "Lichen": "苔藓样皮肤病",
    "Lupus": "红斑狼疮相关皮损",
    "Moles": "色素痣",
    "Psoriasis": "银屑病",
    "Rosacea": "玫瑰痤疮",
    "Seborrh_Keratoses": "脂溢性角化病",
    "SkinCancer": "皮肤癌",
    "Sun_Sunlight_Damage": "日晒损伤",
    "Tinea": "癣（皮肤真菌感染）",
    "Unknown_Normal": "正常/未见明显异常",
    "Vascular_Tumors": "血管性肿瘤",
    "Vasculitis": "血管炎",
    "Vitiligo": "白癜风",
    "Warts": "疣",
}

SOURCE_ZH: dict[str, str] = {
    "qwen_vl_api": "大模型API路径（Qwen-VL）",
    "local_hybrid": "本地融合路径（local_hybrid）",
    "local_mock": "本地规则路径（local_mock）",
}

MODE_ZH: dict[str, str] = {
    "hybrid": "图像+文本融合",
    "image_only": "仅图像",
}

DURATION_FLAG_ZH: dict[str, str] = {"acute": "急性", "chronic": "慢性"}
LOCATION_FLAG_ZH: dict[str, str] = {
    "face": "面部",
    "trunk": "躯干",
    "limb": "四肢",
    "scalp": "头皮",
}
SEVERITY_FLAG_ZH: dict[str, str] = {"mild": "轻度", "moderate": "中度", "severe": "重度"}

ITCH_FLAG_ZH: dict[str, str] = {
    "no_itch": "无瘙痒", "mild_itch": "轻微瘙痒",
    "moderate_itch": "中度瘙痒", "severe_itch": "剧烈瘙痒",
}
PAIN_FLAG_ZH: dict[str, str] = {
    "no_pain": "无疼痛", "mild_pain": "轻微疼痛",
    "moderate_pain": "中度疼痛", "severe_pain": "剧烈疼痛",
}
TRIGGER_FLAG_ZH: dict[str, str] = {
    "sun": "日晒", "drug": "用药", "bite": "虫咬", "contact": "接触刺激物",
}
MORPHOLOGY_FLAG_ZH: dict[str, str] = {
    "erythema": "红斑", "scale": "鳞屑", "vesicle": "水疱",
    "ulcer": "溃疡", "papule": "丘疹", "pigment": "色素改变",
}
RECURRENCE_FLAG_ZH: dict[str, str] = {
    "first": "首次发作", "recurrent": "反复发作",
}
AGE_FLAG_ZH: dict[str, str] = {
    "child": "儿童", "youth": "青年", "middle": "中年", "elderly": "老年",
}


HISTORY_EXPIRE_DAYS = 7


def _load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            with HISTORY_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_history(records: list[dict]) -> None:
    with HISTORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _purge_expired_history() -> None:
    records = _load_history()
    cutoff = datetime.now().timestamp() - HISTORY_EXPIRE_DAYS * 86400
    kept = []
    for r in records:
        try:
            ts = datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S").timestamp()
            if ts >= cutoff:
                kept.append(r)
        except (KeyError, ValueError):
            kept.append(r)
    if len(kept) < len(records):
        _save_history(kept)


def _append_history(
    symptom_text: str,
    form_data: dict[str, str],
    result: dict,
    filename: str,
) -> None:
    records = _load_history()
    records.insert(0, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "symptom_text": symptom_text,
        "form_data": form_data,
        "primary_diagnosis": result.get("primary_diagnosis", ""),
        "confidence": result.get("confidence", 0),
        "source": result.get("source", ""),
        "top3_candidates": result.get("top3_candidates", []),
    })
    _save_history(records)


def _render_history_panel_sidebar() -> None:
    st.sidebar.subheader("历史记录")

    records = _load_history()

    query = st.sidebar.text_input("搜索（诊断/症状/文件名）", key="history_search").strip().lower()

    if query:
        filtered = [
            r for r in records
            if query in r.get("primary_diagnosis", "").lower()
            or query in r.get("symptom_text", "").lower()
            or query in r.get("filename", "").lower()
            or query in _label_with_zh(r.get("primary_diagnosis", "")).lower()
        ]
    else:
        filtered = records

    if not filtered:
        st.sidebar.info("暂无历史记录。")
    else:
        st.sidebar.caption(f"共 {len(records)} 条，显示 {len(filtered)} 条")
        for rec in filtered[:50]:
            diag_zh = _label_with_zh(rec.get("primary_diagnosis", "?"))
            conf = rec.get("confidence", 0)
            source_zh = _source_to_zh(rec.get("source", ""))
            time_str = rec.get("time", "")
            fname = rec.get("filename", "")
            with st.sidebar.expander(f"{time_str}  {diag_zh}", expanded=False):
                st.write(f"**文件**: {fname}")
                st.write(f"**诊断**: {diag_zh}  置信度: {conf}")
                st.write(f"**来源**: {source_zh}")
                st.write(f"**症状**: {rec.get('symptom_text', '')}")
                top3 = rec.get("top3_candidates", [])
                if top3:
                    st.write("**Top-3**:")
                    for item in top3:
                        st.write(f"- {_label_with_zh(item.get('label', ''))}: {item.get('score', 0)}")

    if st.sidebar.button("清空全部历史", type="secondary"):
        _save_history([])
        st.rerun()


def _render_settings_page(dataset_root: str, labels: list[str]) -> None:
    st.title("运行配置")

    if st.button("← 返回主页", use_container_width=True):
        st.session_state.show_settings = False
        st.rerun()

    st.markdown("---")

    st.subheader("数据集信息")
    st.write(f"数据集目录：`{dataset_root}`")
    st.write(f"类别数：`{len(labels)}`")
    st.write(f"图像归一化尺寸：`{NORMALIZED_SIZE}x{NORMALIZED_SIZE}`")

    st.markdown("---")

    st.subheader("本地模型配置")
    local_model_dir = st.text_input(
        "本地模型目录",
        value=st.session_state.get("local_model_dir", os.getenv("LOCAL_MODEL_DIR", "artifacts")),
    )
    st.session_state.local_model_dir = local_model_dir

    if local_hybrid_artifacts_available(local_model_dir):
        st.success("✓ 已检测到 local_hybrid 模型工件。")
    else:
        st.warning("⚠ 未检测到 local_hybrid 工件，将自动回退 local_mock。")

    st.markdown("---")

    st.subheader("API 配置")
    api_key = st.text_input(
        "QWEN_API_KEY（可留空触发本地推理）",
        value=st.session_state.get("api_key", os.getenv("QWEN_API_KEY", "")),
        type="password",
    )
    st.session_state.api_key = api_key

    model = st.text_input(
        "Qwen-VL 模型名",
        value=st.session_state.get("model", "qwen-vl-max-latest"),
    )
    st.session_state.model = model

    base_url = st.text_input(
        "API Base URL",
        value=st.session_state.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    st.session_state.base_url = base_url

    timeout = st.slider(
        "API 超时（秒）",
        min_value=10,
        max_value=90,
        value=st.session_state.get("timeout", 40),
        step=5,
    )
    st.session_state.timeout = timeout

    st.markdown("---")
    st.info("配置已自动保存到当前会话。")


def _render_history_panel(col) -> None:
    col.subheader("历史记录")

    records = _load_history()

    query = col.text_input("搜索（诊断/症状/文件名）", key="history_search").strip().lower()

    if query:
        filtered = [
            r for r in records
            if query in r.get("primary_diagnosis", "").lower()
            or query in r.get("symptom_text", "").lower()
            or query in r.get("filename", "").lower()
            or query in _label_with_zh(r.get("primary_diagnosis", "")).lower()
        ]
    else:
        filtered = records

    if not filtered:
        col.info("暂无历史记录。")
    else:
        col.caption(f"共 {len(records)} 条，显示 {len(filtered)} 条")
        for rec in filtered[:50]:
            diag_zh = _label_with_zh(rec.get("primary_diagnosis", "?"))
            conf = rec.get("confidence", 0)
            source_zh = _source_to_zh(rec.get("source", ""))
            time_str = rec.get("time", "")
            fname = rec.get("filename", "")
            with col.expander(f"{time_str}  {diag_zh}", expanded=False):
                st.write(f"**文件**: {fname}")
                st.write(f"**诊断**: {diag_zh}  置信度: {conf}")
                st.write(f"**来源**: {source_zh}")
                st.write(f"**症状**: {rec.get('symptom_text', '')}")
                top3 = rec.get("top3_candidates", [])
                if top3:
                    st.write("**Top-3**:")
                    for item in top3:
                        st.write(f"- {_label_with_zh(item.get('label', ''))}: {item.get('score', 0)}")

    if col.button("清空全部历史", type="secondary"):
        _save_history([])
        st.rerun()


def _build_structured_form() -> dict[str, str]:
    st.subheader("症状信息")
    col1, col2 = st.columns(2)
    with col1:
        duration = st.selectbox("病程", ["不确定", "急性（近几天）", "慢性（数周以上）"])
        location = st.selectbox("部位", ["面部", "躯干", "四肢", "头皮", "其他"])
        itch = st.selectbox("瘙痒程度", ["无", "轻微", "中度", "剧烈"])
        pain = st.selectbox("疼痛程度", ["无", "轻微", "中度", "剧烈"])
    with col2:
        trigger = st.selectbox("诱因", ["不明", "日晒", "用药后", "虫咬", "接触刺激物"])
        morphology = st.selectbox("皮损形态", ["其他", "红斑", "鳞屑", "水疱", "溃疡", "丘疹", "色素改变"])
        recurrence = st.selectbox("是否复发", ["不确定", "首次发作", "反复发作"])
        age = st.selectbox("年龄段", ["青年（14-35）", "儿童（<14）", "中年（36-59）", "老年（≥60）"])
    return {
        "duration": duration,
        "location": location,
        "itch": itch,
        "pain": pain,
        "trigger": trigger,
        "morphology": morphology,
        "recurrence": recurrence,
        "age": age,
    }


def _structured_to_text(form: dict[str, str]) -> str:
    parts: list[str] = []

    dur = form["duration"]
    if dur.startswith("急性"):
        parts.append("急性")
    elif dur.startswith("慢性"):
        parts.append("慢性")

    loc = form["location"]
    if loc != "其他":
        parts.append(loc)

    itch = form["itch"]
    if itch == "无":
        parts.append("无瘙痒")
    elif itch == "轻微":
        parts.append("轻微瘙痒")
    elif itch == "中度":
        parts.append("中度瘙痒")
    elif itch == "剧烈":
        parts.append("剧烈瘙痒")

    pain = form["pain"]
    if pain == "无":
        parts.append("无疼痛")
    elif pain == "轻微":
        parts.append("轻微疼痛")
    elif pain == "中度":
        parts.append("中度疼痛")
    elif pain == "剧烈":
        parts.append("剧烈疼痛")

    trigger = form["trigger"]
    if trigger == "日晒":
        parts.append("日晒诱因")
    elif trigger == "用药后":
        parts.append("用药诱因")
    elif trigger == "虫咬":
        parts.append("虫咬诱因")
    elif trigger == "接触刺激物":
        parts.append("接触刺激物")

    morph = form["morphology"]
    if morph != "其他":
        parts.append(morph)

    rec = form["recurrence"]
    if rec == "首次发作":
        parts.append("首次发作")
    elif rec == "反复发作":
        parts.append("反复发作")

    age = form["age"]
    if age.startswith("儿童"):
        parts.append("儿童")
    elif age.startswith("青年"):
        parts.append("青年")
    elif age.startswith("中年"):
        parts.append("中年")
    elif age.startswith("老年"):
        parts.append("老年")

    return "，".join(parts) + "。" if parts else ""


def _detect_mime_type(filename: str, provided_mime: str | None) -> str:
    if provided_mime:
        return provided_mime
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    return "application/octet-stream"


def _label_to_zh(label: str) -> str:
    return DISEASE_LABEL_ZH.get(label, label)


def _source_to_zh(source: str) -> str:
    return SOURCE_ZH.get(source, source)


def _label_with_zh(label: str) -> str:
    zh = _label_to_zh(label)
    if zh == label:
        return label
    return f"{zh}（{label}）"


def _translate_signal_text(signal: str) -> str:
    if ":" not in signal:
        return signal
    prefix, rest = signal.split(":", 1)
    prefix = prefix.strip()
    rest = rest.strip()
    return f"{_label_with_zh(prefix)}: {rest}"


def _translate_result_for_display(result: dict) -> dict:
    view = copy.deepcopy(result)

    if isinstance(view.get("primary_diagnosis"), str):
        view["primary_diagnosis"] = _label_with_zh(view["primary_diagnosis"])

    if isinstance(view.get("source"), str):
        view["source"] = _source_to_zh(view["source"])

    top3 = view.get("top3_candidates")
    if isinstance(top3, list):
        for item in top3:
            if isinstance(item, dict) and isinstance(item.get("label"), str):
                item["label"] = _label_with_zh(item["label"])

    trace = view.get("decision_trace")
    if isinstance(trace, dict):
        mode = trace.get("mode")
        if isinstance(mode, str):
            trace["mode"] = MODE_ZH.get(mode, mode)

        text_trace = trace.get("text")
        if isinstance(text_trace, dict):
            if isinstance(text_trace.get("duration_flags"), list):
                text_trace["duration_flags"] = [
                    DURATION_FLAG_ZH.get(x, x) for x in text_trace["duration_flags"]
                ]
            if isinstance(text_trace.get("location_flags"), list):
                text_trace["location_flags"] = [
                    LOCATION_FLAG_ZH.get(x, x) for x in text_trace["location_flags"]
                ]
            if isinstance(text_trace.get("severity_flags"), list):
                text_trace["severity_flags"] = [
                    SEVERITY_FLAG_ZH.get(x, x) for x in text_trace["severity_flags"]
                ]
            if isinstance(text_trace.get("itch_flags"), list):
                text_trace["itch_flags"] = [
                    ITCH_FLAG_ZH.get(x, x) for x in text_trace["itch_flags"]
                ]
            if isinstance(text_trace.get("pain_flags"), list):
                text_trace["pain_flags"] = [
                    PAIN_FLAG_ZH.get(x, x) for x in text_trace["pain_flags"]
                ]
            if isinstance(text_trace.get("trigger_flags"), list):
                text_trace["trigger_flags"] = [
                    TRIGGER_FLAG_ZH.get(x, x) for x in text_trace["trigger_flags"]
                ]
            if isinstance(text_trace.get("morphology_flags"), list):
                text_trace["morphology_flags"] = [
                    MORPHOLOGY_FLAG_ZH.get(x, x) for x in text_trace["morphology_flags"]
                ]
            if isinstance(text_trace.get("recurrence_flags"), list):
                text_trace["recurrence_flags"] = [
                    RECURRENCE_FLAG_ZH.get(x, x) for x in text_trace["recurrence_flags"]
                ]
            if isinstance(text_trace.get("age_flags"), list):
                text_trace["age_flags"] = [
                    AGE_FLAG_ZH.get(x, x) for x in text_trace["age_flags"]
                ]
            if isinstance(text_trace.get("matched_signals"), list):
                text_trace["matched_signals"] = [
                    _translate_signal_text(x) for x in text_trace["matched_signals"]
                ]
            if isinstance(text_trace.get("negated_signals"), list):
                text_trace["negated_signals"] = [
                    _translate_signal_text(x) for x in text_trace["negated_signals"]
                ]

    return view


def _preprocess_image(
    image_bytes: bytes,
    target_size: int = NORMALIZED_SIZE,
) -> tuple[bytes, str, tuple[int, int], tuple[int, int]]:
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError as exc:
        raise ValueError("上传文件不是有效图片，请使用 JPG/JPEG/PNG。") from exc

    # Apply EXIF rotation correction before stripping metadata
    image = ImageOps.exif_transpose(image)
    original_size = image.size

    if image.mode != "RGB":
        image = image.convert("RGB")

    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    normalized = ImageOps.fit(
        image,
        (target_size, target_size),
        method=resample,
        centering=(0.5, 0.5),
    )

    # Strip all EXIF/metadata by creating a clean image from raw pixel data
    clean = Image.frombytes("RGB", normalized.size, normalized.tobytes())

    buffer = io.BytesIO()
    clean.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue(), "image/jpeg", original_size, normalized.size


def main() -> None:
    st.set_page_config(page_title="皮肤疾病初筛系统", layout="centered")
    st.title("皮肤疾病初筛演示系统")
    st.caption("输入皮肤图片 + 症状文本，返回结构化初步筛查结果（非临床诊断）。")

    _purge_expired_history()

    try:
        dataset_root = resolve_dataset_root()
        labels = load_class_labels(dataset_root)
    except Exception as exc:  # noqa: BLE001
        st.error(f"数据集加载失败：{exc}")
        st.stop()

    # Session state for page navigation
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False

    # Sidebar: History + Settings button
    with st.sidebar:
        if st.button("运行配置", use_container_width=True):
            st.session_state.show_settings = not st.session_state.show_settings

        st.markdown("---")
        _render_history_panel_sidebar()

    # Settings page (overlay)
    if st.session_state.show_settings:
        _render_settings_page(dataset_root, labels)
        return

    # Get config from session state or env
    local_model_dir = st.session_state.get("local_model_dir", os.getenv("LOCAL_MODEL_DIR", "artifacts"))
    api_key = st.session_state.get("api_key", os.getenv("QWEN_API_KEY", ""))
    model = st.session_state.get("model", "qwen-vl-max-latest")
    base_url = st.session_state.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    timeout = st.session_state.get("timeout", 40)

    # Main content
    uploaded = st.file_uploader("上传皮肤图片", type=["jpg", "jpeg", "png"])
    form_data = _build_structured_form()
    symptom_text = _structured_to_text(form_data)

    if uploaded:
        st.image(uploaded, caption="已上传图片", width="stretch")

    consent = st.checkbox(
        "我已阅读并同意《隐私保护与数据去标识化声明》，授权系统仅出于初步筛查目的处理上述图像。",
        value=False,
    )

    run = st.button("开始分析", type="primary", use_container_width=True, disabled=not consent)
    if not run:
        return

    if not uploaded:
        st.error("请先上传一张图片。")
        st.stop()

    detected_mime = _detect_mime_type(uploaded.name, uploaded.type)
    if detected_mime not in {"image/jpeg", "image/png"}:
        st.error("仅支持 JPG/JPEG/PNG 图片格式。")
        st.stop()

    try:
        image_bytes, mime_type, original_size, normalized_size = _preprocess_image(
            uploaded.getvalue(),
            target_size=NORMALIZED_SIZE,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("模型推理中，请稍候..."):
        result, debug_errors = infer_with_qwen_or_mock(
            image_bytes=image_bytes,
            mime_type=mime_type,
            symptom_text=symptom_text,
            labels=labels,
            api_key=api_key or None,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )

    _append_history(
        symptom_text=symptom_text,
        form_data=form_data,
        result=result,
        filename=uploaded.name,
    )

    display_result = _translate_result_for_display(result)

    st.caption(
        "图像预处理："
        f"{original_size[0]}x{original_size[1]} -> "
        f"{normalized_size[0]}x{normalized_size[1]}（RGB/JPEG）"
    )

    st.subheader("输入摘要")
    st.write(f"`{symptom_text}`")

    st.subheader("结构化结果")
    st.json(display_result)

    st.subheader("结果摘要")
    st.write(
        f"初步诊断：`{display_result['primary_diagnosis']}` | "
        f"置信度：`{display_result['confidence']}` | "
        f"来源：`{display_result['source']}`"
    )
    st.info(display_result["note"])

    if display_result.get("top3_candidates"):
        st.subheader("Top-3 候选")
        st.table(
            [
                {"候选诊断": item.get("label"), "分数": item.get("score")}
                for item in display_result["top3_candidates"]
            ]
        )

    if display_result.get("decision_trace"):
        with st.expander("决策轨迹（decision_trace）", expanded=False):
            st.json(display_result["decision_trace"])

    if result.get("mock_result"):
        st.warning("当前为本地路径结果（非API直接结果）。")
        if any("data_inspection_failed" in err for err in debug_errors):
            st.warning("检测到平台内容审核拦截（data_inspection_failed），已自动切换本地推理。")

    if debug_errors:
        with st.expander("调试信息（调用失败原因）", expanded=False):
            for item in debug_errors:
                st.write(f"- {item}")


if __name__ == "__main__":
    main()
