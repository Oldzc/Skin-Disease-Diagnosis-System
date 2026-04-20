from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

APP_DIR = Path(__file__).parent
USERS_FILE = APP_DIR / "users.json"
USER_HISTORY_DIR = APP_DIR / "user_histories"
LEGACY_HISTORY_FILE = APP_DIR / "history.json"

from core.inference import (
    PROVIDER_ANTHROPIC,
    PROVIDER_DEFAULTS,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    PROVIDER_QWEN,
    infer_with_provider,
)
from core.local_hybrid import local_hybrid_artifacts_available
from core.mock_engine import load_class_labels, resolve_dataset_root

PROVIDER_LABELS: dict[str, str] = {
    PROVIDER_QWEN: "Qwen-VL（阿里云通义千问）",
    PROVIDER_OPENAI: "OpenAI（GPT-4o）",
    PROVIDER_ANTHROPIC: "Anthropic（Claude）",
    PROVIDER_GEMINI: "Google Gemini",
}

PROVIDER_ENV_KEY: dict[str, str] = {
    PROVIDER_QWEN: "QWEN_API_KEY",
    PROVIDER_OPENAI: "OPENAI_API_KEY",
    PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
    PROVIDER_GEMINI: "GOOGLE_API_KEY",
}

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
    "local_hybrid": "本地推理",
    "local_mock": "本地规则路径（local_mock）",
}

LOCAL_ARCH_OPTIONS = ["auto", "mobilenet_v3_small", "resnet18", "efficientnet_b0"]
LOCAL_ARCH_LABELS = {
    "auto": "自动（优先当前目录）",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "resnet18": "ResNet18",
    "efficientnet_b0": "EfficientNet-B0",
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


def _normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def _validate_username(username: str) -> tuple[bool, str]:
    name = _normalize_username(username)
    if not re.fullmatch(r"[a-z0-9_-]{3,32}", name):
        return False, "用户名需为 3-32 位，仅支持小写字母、数字、下划线、短横线。"
    return True, ""


def _load_users() -> dict[str, dict]:
    if USERS_FILE.exists():
        try:
            with USERS_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_users(users: dict[str, dict]) -> None:
    with USERS_FILE.open("w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _hash_password(password: str, salt: str) -> str:
    payload = f"{salt}:{password}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _register_user(username: str, password: str) -> tuple[bool, str]:
    ok, msg = _validate_username(username)
    if not ok:
        return False, msg
    if len(password) < 6:
        return False, "密码长度至少 6 位。"

    uname = _normalize_username(username)
    users = _load_users()
    if uname in users:
        return False, "用户名已存在。"

    salt = secrets.token_hex(16)
    users[uname] = {
        "salt": salt,
        "password_hash": _hash_password(password, salt),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_users(users)
    return True, "注册成功。"


def _verify_user(username: str, password: str) -> bool:
    uname = _normalize_username(username)
    users = _load_users()
    record = users.get(uname)
    if not isinstance(record, dict):
        return False
    salt = str(record.get("salt", ""))
    expected = str(record.get("password_hash", ""))
    if not salt or not expected:
        return False
    return _hash_password(password, salt) == expected


def _get_current_user() -> str:
    current = st.session_state.get("current_user", "guest")
    name = _normalize_username(str(current))
    return name if name else "guest"


def _history_file_for_user(username: str) -> Path:
    USER_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _normalize_username(username) or "guest"
    safe_name = re.sub(r"[^a-z0-9_-]", "_", safe_name)
    return USER_HISTORY_DIR / f"{safe_name}.json"


def _ensure_storage_layout() -> None:
    USER_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    if LEGACY_HISTORY_FILE.exists():
        guest_file = _history_file_for_user("guest")
        if not guest_file.exists():
            try:
                with LEGACY_HISTORY_FILE.open("r", encoding="utf-8") as f:
                    legacy = json.load(f)
                if isinstance(legacy, list):
                    with guest_file.open("w", encoding="utf-8") as f:
                        json.dump(legacy, f, ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, OSError):
                pass
        try:
            LEGACY_HISTORY_FILE.unlink()
        except OSError:
            pass


def _find_arch_specific_dir(base_dir: str | Path, arch: str) -> Path | None:
    root = Path(base_dir)
    candidates = _build_arch_dir_candidates(root, arch)
    for p in candidates:
        if local_hybrid_artifacts_available(p):
            return p
    return None


def _build_arch_dir_candidates(root: Path, arch: str) -> list[Path]:
    candidates: list[Path] = []
    roots_to_try: list[Path] = [root]

    # If user sets base_dir to a specific arch folder, allow switching to sibling arch folders.
    if root.name in LOCAL_ARCH_OPTIONS and root.name != "auto":
        roots_to_try.insert(0, root.parent)
    else:
        roots_to_try.extend([root.parent])

    for base in roots_to_try:
        if not str(base):
            continue
        candidates.append(base / arch)
        candidates.append(base / "multi_model_compare" / arch)

    # De-duplicate while preserving order.
    ordered: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(p)
    return ordered


def _resolve_effective_local_model_dir(base_dir: str | Path, arch: str) -> str:
    root = Path(base_dir)
    if arch and arch != "auto":
        picked = _find_arch_specific_dir(root, arch)
        if picked is not None:
            return str(picked)
        # Keep showing the target arch path even when artifacts are missing,
        # so switching arch visibly changes and users know where files are expected.
        preferred = _build_arch_dir_candidates(root, arch)[0]
        return str(preferred)

    if local_hybrid_artifacts_available(root):
        return str(root)

    for cand in LOCAL_ARCH_OPTIONS:
        if cand == "auto":
            continue
        picked = _find_arch_specific_dir(root, cand)
        if picked is not None:
            return str(picked)

    return str(root)


def _load_history(username: str | None = None) -> list[dict]:
    user = _normalize_username(username or _get_current_user()) or "guest"
    history_file = _history_file_for_user(user)
    if history_file.exists():
        try:
            with history_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_history(records: list[dict], username: str | None = None) -> None:
    user = _normalize_username(username or _get_current_user()) or "guest"
    history_file = _history_file_for_user(user)
    with history_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _purge_expired_history() -> None:
    cutoff = datetime.now().timestamp() - HISTORY_EXPIRE_DAYS * 86400
    for history_file in USER_HISTORY_DIR.glob("*.json"):
        try:
            with history_file.open("r", encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list):
                continue
        except (json.JSONDecodeError, OSError):
            continue

        kept = []
        for r in records:
            try:
                ts = datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S").timestamp()
                if ts >= cutoff:
                    kept.append(r)
            except (KeyError, ValueError):
                kept.append(r)
        if len(kept) < len(records):
            with history_file.open("w", encoding="utf-8") as f:
                json.dump(kept, f, ensure_ascii=False, indent=2)


def _append_history(
    symptom_text: str,
    form_data: dict[str, str],
    result: dict,
    filename: str,
) -> None:
    user = _get_current_user()
    records = _load_history(user)
    records.insert(0, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user,
        "filename": filename,
        "symptom_text": symptom_text,
        "form_data": form_data,
        "primary_diagnosis": result.get("primary_diagnosis", ""),
        "confidence": result.get("confidence", 0),
        "source": result.get("source", ""),
        "top3_candidates": result.get("top3_candidates", []),
    })
    _save_history(records, user)


def _render_auth_page() -> None:
    current_user = _get_current_user()
    st.title("用户登录")
    st.caption("本地账号系统：登录后将按用户隔离保存历史记录。")

    if st.button("← 返回主页", use_container_width=True):
        st.session_state.show_auth_page = False
        st.rerun()

    st.markdown("---")
    st.write(f"当前账号：`{current_user}`")

    if current_user != "guest":
        st.success("当前已登录。")
        if st.button("退出登录", use_container_width=True, type="secondary"):
            st.session_state.current_user = "guest"
            st.session_state.show_auth_page = False
            st.rerun()
        return

    mode = st.radio(
        "账号操作",
        ["登录", "注册"],
        key="auth_mode",
        horizontal=True,
    )

    if mode == "登录":
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("用户名", key="login_username")
            password = st.text_input("密码", type="password", key="login_password")
            submit = st.form_submit_button("登录", use_container_width=True)
        if submit:
            if _verify_user(username, password):
                st.session_state.current_user = _normalize_username(username)
                st.session_state.show_auth_page = False
                st.success("登录成功。")
                st.rerun()
            st.error("用户名或密码错误。")
    else:
        st.caption("用户名仅支持小写字母、数字、下划线、短横线（3-32位）。")
        with st.form("register_form", clear_on_submit=False):
            username = st.text_input("用户名", key="reg_username")
            password = st.text_input("密码", type="password", key="reg_password")
            confirm = st.text_input("确认密码", type="password", key="reg_password_confirm")
            submit = st.form_submit_button("注册并登录", use_container_width=True)
        if submit:
            if password != confirm:
                st.error("两次密码不一致。")
            else:
                ok, msg = _register_user(username, password)
                if ok:
                    st.session_state.current_user = _normalize_username(username)
                    st.session_state.show_auth_page = False
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)


def _render_history_panel_sidebar() -> None:
    st.sidebar.subheader("历史记录")

    current_user = _get_current_user()
    records = _load_history(current_user)
    st.sidebar.caption(f"账号：`{current_user}`")

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
            rec_sig = json.dumps(rec, ensure_ascii=False, sort_keys=True)
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
                if st.button("删除", key=f"delete_{hash(rec_sig)}", type="secondary"):
                    st.session_state[f"confirm_{hash(rec_sig)}"] = True
                if st.session_state.get(f"confirm_{hash(rec_sig)}"):
                    st.warning("确认删除这条记录？")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("确认", key=f"yes_{hash(rec_sig)}", type="primary"):
                            new_records = [
                                r for r in records
                                if json.dumps(r, ensure_ascii=False, sort_keys=True) != rec_sig
                            ]
                            _save_history(new_records, current_user)
                            st.session_state.pop(f"confirm_{hash(rec_sig)}", None)
                            st.rerun()
                    with col_no:
                        if st.button("取消", key=f"no_{hash(rec_sig)}"):
                            st.session_state.pop(f"confirm_{hash(rec_sig)}", None)
                            st.rerun()

    if st.sidebar.button("清空当前账号历史", type="secondary"):
        _save_history([], current_user)
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
        value=st.session_state.get("local_model_dir", os.getenv("LOCAL_MODEL_DIR", "artifacts/multi_model_compare")),
    )
    st.session_state.local_model_dir = local_model_dir

    current_arch = st.session_state.get("local_model_arch", os.getenv("LOCAL_MODEL_ARCH", "efficientnet_b0"))
    if current_arch not in LOCAL_ARCH_OPTIONS:
        current_arch = "auto"

    local_arch = st.selectbox(
        "本地推理网络",
        LOCAL_ARCH_OPTIONS,
        index=LOCAL_ARCH_OPTIONS.index(current_arch),
        format_func=lambda x: LOCAL_ARCH_LABELS.get(x, x),
        help="仅影响本地推理路径（local_hybrid / local_mock），API正常时优先走API。",
    )
    st.session_state.local_model_arch = local_arch

    effective_local_model_dir = _resolve_effective_local_model_dir(local_model_dir, local_arch)
    st.session_state.local_model_effective_dir = effective_local_model_dir
    st.caption(f"当前生效模型目录：`{effective_local_model_dir}`")

    if local_hybrid_artifacts_available(effective_local_model_dir):
        st.success("已检测到 local_hybrid 模型工件（当前网络选择可用）。")
    else:
        st.warning("当前网络选择未检测到 local_hybrid 工件，将自动回退 local_mock。")

    st.markdown("---")

    st.subheader("API 配置")

    provider_options = list(PROVIDER_LABELS.keys())
    current_provider = st.session_state.get(
        "provider", os.getenv("PREFERRED_PROVIDER", PROVIDER_QWEN)
    )
    if current_provider not in provider_options:
        current_provider = PROVIDER_QWEN

    provider = provider_options[
        st.selectbox(
            "模型供应商",
            range(len(provider_options)),
            index=provider_options.index(current_provider),
            format_func=lambda i: PROVIDER_LABELS[provider_options[i]],
        )
    ]
    st.session_state.provider = provider

    env_key_name = PROVIDER_ENV_KEY[provider]
    api_key = st.text_input(
        f"API Key（{env_key_name}）",
        value=st.session_state.get("api_key", os.getenv(env_key_name, "")),
        type="password",
        help=f"留空则自动使用本地推理。对应环境变量：{env_key_name}",
    )
    st.session_state.api_key = api_key

    defaults = PROVIDER_DEFAULTS[provider]
    model = st.text_input(
        "模型名称",
        value=st.session_state.get("model", defaults["model"]),
        help=f"默认：{defaults['model']}",
    )
    st.session_state.model = model

    base_url = st.text_input(
        "API Base URL",
        value=st.session_state.get("base_url", defaults["base_url"]),
        help=f"默认：{defaults['base_url']}",
    )
    st.session_state.base_url = base_url

    timeout = st.slider(
        "API 超时（秒）",
        min_value=10,
        max_value=120,
        value=st.session_state.get("timeout", 40),
        step=5,
    )
    st.session_state.timeout = timeout

    st.markdown("---")

    st.subheader("各供应商说明")
    with st.expander("Qwen-VL", expanded=False):
        st.markdown("""
- 申请地址：https://bailian.console.aliyun.com/
- 环境变量：`QWEN_API_KEY`
- 默认模型：`qwen-vl-max-latest`
        """)
    with st.expander("OpenAI", expanded=False):
        st.markdown("""
- 申请地址：https://platform.openai.com
- 环境变量：`OPENAI_API_KEY`
- 默认模型：`gpt-4o`
        """)
    with st.expander("Anthropic", expanded=False):
        st.markdown("""
- 申请地址：https://console.anthropic.com
- 环境变量：`ANTHROPIC_API_KEY`
- 默认模型：`claude-3-5-sonnet-20241022`
        """)
    with st.expander("Google Gemini", expanded=False):
        st.markdown("""
- 申请地址：https://aistudio.google.com
- 环境变量：`GOOGLE_API_KEY`
- 默认模型：`gemini-1.5-pro-latest`
        """)

    st.info("配置已自动保存到当前会话，刷新页面后重置为环境变量默认值。")


def _render_history_panel(col) -> None:
    col.subheader("历史记录")

    current_user = _get_current_user()
    records = _load_history(current_user)
    col.caption(f"账号：`{current_user}`")

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

    if col.button("清空当前账号历史", type="secondary"):
        _save_history([], current_user)
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

    clean = Image.frombytes("RGB", normalized.size, normalized.tobytes())

    buffer = io.BytesIO()
    clean.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue(), "image/jpeg", original_size, normalized.size


def main() -> None:
    st.set_page_config(page_title="皮肤疾病初筛系统", layout="centered")
    st.title("皮肤疾病初筛演示系统")
    st.caption("输入皮肤图片 + 症状文本，返回初步筛查结果（非临床诊断）。")

    _ensure_storage_layout()
    if "current_user" not in st.session_state:
        st.session_state.current_user = "guest"
    if "show_auth_page" not in st.session_state:
        st.session_state.show_auth_page = False

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
        if st.button("用户登录", use_container_width=True):
            st.session_state.show_auth_page = not st.session_state.show_auth_page
            if st.session_state.show_auth_page:
                st.session_state.show_settings = False

        st.markdown("---")

        if st.button("运行配置", use_container_width=True):
            st.session_state.show_settings = not st.session_state.show_settings
            if st.session_state.show_settings:
                st.session_state.show_auth_page = False

        st.markdown("---")
        _render_history_panel_sidebar()

    # Auth page (overlay)
    if st.session_state.show_auth_page:
        _render_auth_page()
        return

    # Settings page (overlay)
    if st.session_state.show_settings:
        _render_settings_page(dataset_root, labels)
        return

    # Get config from session state or env
    provider = st.session_state.get("provider", os.getenv("PREFERRED_PROVIDER", PROVIDER_QWEN))
    if provider not in PROVIDER_LABELS:
        provider = PROVIDER_QWEN
    env_key = PROVIDER_ENV_KEY[provider]
    local_model_dir = st.session_state.get("local_model_dir", os.getenv("LOCAL_MODEL_DIR", "artifacts/multi_model_compare"))
    local_model_arch = st.session_state.get("local_model_arch", os.getenv("LOCAL_MODEL_ARCH", "efficientnet_b0"))
    effective_local_model_dir = _resolve_effective_local_model_dir(local_model_dir, local_model_arch)
    st.session_state.local_model_effective_dir = effective_local_model_dir
    os.environ["LOCAL_MODEL_DIR"] = effective_local_model_dir
    api_key = st.session_state.get("api_key", os.getenv(env_key, ""))
    model = st.session_state.get("model", PROVIDER_DEFAULTS[provider]["model"])
    base_url = st.session_state.get("base_url", PROVIDER_DEFAULTS[provider]["base_url"])
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
        _t0 = __import__("time").time()
        result, debug_errors = infer_with_provider(
            image_bytes=image_bytes,
            mime_type=mime_type,
            symptom_text=symptom_text,
            labels=labels,
            provider=provider,
            api_key=api_key or None,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
        _elapsed = round(__import__("time").time() - _t0, 2)

    _append_history(
        symptom_text=symptom_text,
        form_data=form_data,
        result=result,
        filename=uploaded.name,
    )

    display_result = _translate_result_for_display(result)

    # st.caption(
    #     "图像预处理："
    #     f"{original_size[0]}x{original_size[1]} -> "
    #     f"{normalized_size[0]}x{normalized_size[1]}（RGB/JPEG）"
    # )
  
    if display_result.get("top3_candidates"):
        st.subheader("Top-3 候选")
        st.table(
            [
                {"候选诊断": item.get("label"), "置信度": item.get("score")}
                for item in display_result["top3_candidates"]
            ]
        )

    st.subheader("输入摘要")
    st.write(f"`{symptom_text}`")

    #st.subheader("结构化结果")
    #st.json(display_result)

    st.subheader("结果摘要")
    st.write(
        f"初步诊断：`{display_result['primary_diagnosis']}` | "
        f"置信度：`{display_result['confidence']}` | "
        f"来源：`{display_result['source']}` | "
        f"耗时：`{_elapsed} 秒`"
    )
    st.info(display_result["note"])

    if display_result.get("decision_trace"):
        with st.expander("决策轨迹", expanded=False):
            st.json(display_result["decision_trace"])

    if result.get("mock_result"):
        st.warning("当前为本地推理结果（非API结果）。")
        if any("data_inspection_failed" in err for err in debug_errors):
            st.warning("检测到平台内容审核拦截（data_inspection_failed），已自动切换本地推理。")

    if debug_errors:
        with st.expander("调试信息", expanded=False):
            for item in debug_errors:
                st.write(f"- {item}")


if __name__ == "__main__":
    main()
