from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.4f}", dy: float = 0.01) -> None:
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        y = h + dy if h >= 0 else h - dy * 2.0
        va = "bottom" if h >= 0 else "top"
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            y,
            fmt.format(h),
            ha="center",
            va=va,
            fontsize=8,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot key charts from artifacts/experiment_summary.csv")
    parser.add_argument("--summary-csv", type=str, default="artifacts/experiment_summary.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts/figures")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def group_by_exp(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        out[r.get("experiment", "")].append(r)
    return out


def save_placeholder(path: Path, title: str, reason: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.42, reason, ha="center", va="center", fontsize=11)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_exp1(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> list[Path]:
    out_acc = out_dir / "exp1_model_accuracy.png"
    out_tradeoff = out_dir / "exp1_model_tradeoff.png"
    old_out = out_dir / "exp1_model_compare.png"
    if not rows:
        save_placeholder(out_acc, "Exp1 Model Accuracy", "No data", dpi)
        save_placeholder(out_tradeoff, "Exp1 Model Tradeoff", "No data", dpi)
        if old_out.exists():
            old_out.unlink()
        return [out_acc, out_tradeoff]

    labels = [r.get("item", "unknown") for r in rows]
    top1 = [as_float(r.get("top1")) for r in rows]
    macro = [as_float(r.get("macro_f1")) for r in rows]
    size_mb = [as_float(r.get("model_size_mb")) for r in rows]
    infer_ms = [as_float(r.get("inference_ms_per_image")) for r in rows]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    b1 = ax.bar(x - w / 2, top1, width=w, label="Top-1 Accuracy", color="#1f77b4")
    b2 = ax.bar(x + w / 2, macro, width=w, label="Macro-F1", color="#ff7f0e")
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Exp1: Backbone Comparison (Accuracy)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_acc, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    size_scale = np.array(size_mb, dtype=float)
    size_scale = np.clip(size_scale * 18.0, 80.0, 2200.0)
    scatter = ax.scatter(infer_ms, top1, s=size_scale, c=macro, cmap="viridis", alpha=0.85, edgecolor="k")
    for i, name in enumerate(labels):
        ax.annotate(name, (infer_ms[i], top1[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Inference Time (ms/image)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Exp1: Accuracy-Latency-Size Tradeoff")
    ax.grid(alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Macro-F1")
    fig.tight_layout()
    fig.savefig(out_tradeoff, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if old_out.exists():
        old_out.unlink()
    return [out_acc, out_tradeoff]


def plot_exp2(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> Path:
    out = out_dir / "exp2_multimodal_compare.png"
    if not rows:
        save_placeholder(out, "Exp2 Multimodal", "No data", dpi)
        return out

    labels = [r.get("item", "unknown") for r in rows]
    top1 = [as_float(r.get("top1")) for r in rows]
    top3 = [as_float(r.get("top3")) for r in rows]
    macro = [as_float(r.get("macro_f1")) for r in rows]

    x = np.arange(len(labels))
    w = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    b1 = ax.bar(x - w, top1, width=w, label="Top-1 Accuracy", color="#1f77b4")
    b2 = ax.bar(x, top3, width=w, label="Top-3", color="#2ca02c")
    b3 = ax.bar(x + w, macro, width=w, label="Macro-F1", color="#ff7f0e")
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    annotate_bars(ax, b3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Exp2: Multimodal Input Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_abg(item: str) -> tuple[float, float, float] | None:
    if not item:
        return None
    m = re.search(r"a=([-\d.]+),b=([-\d.]+),g=([-\d.]+)", item.replace(" ", ""))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def plot_exp3(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> list[Path]:
    out = out_dir / "exp3_fusion_top5_plus_spike.png"
    old_out1 = out_dir / "exp3_fusion_top10.png"
    old_out2 = out_dir / "exp3_fusion_latency_tradeoff.png"
    if not rows:
        save_placeholder(out, "Exp3 Fusion Weight Sweep", "No data", dpi)
        if old_out1.exists():
            old_out1.unlink()
        if old_out2.exists():
            old_out2.unlink()
        return [out]

    enriched = []
    for r in rows:
        item = r.get("item", "")
        abg = parse_abg(item)
        enriched.append(
            {
                "item": item,
                "top1": as_float(r.get("top1")),
                "macro": as_float(r.get("macro_f1")),
                "lat": as_float(r.get("avg_latency_ms")),
                "a": abg[0] if abg else np.nan,
                "b": abg[1] if abg else np.nan,
                "g": abg[2] if abg else np.nan,
            }
        )

    selected = sorted(enriched, key=lambda x: (-x["top1"], -x["macro"], x["lat"]))[:5]
    spike_name = "a=0.5,b=0.3,g=0.2"
    spike_row = next((x for x in enriched if x["item"].replace(" ", "") == spike_name), None)
    if spike_row is not None and all(x["item"] != spike_row["item"] for x in selected):
        selected.append(spike_row)

    labels = [x["item"] for x in selected]
    top1 = [x["top1"] for x in selected]
    macro = [x["macro"] for x in selected]
    lat = [x["lat"] for x in selected]

    x = np.arange(len(labels))
    w = 0.26
    fig, ax1 = plt.subplots(figsize=(12.8, 5.8))
    b1 = ax1.bar(x - w / 2, top1, width=w, label="Top-1 Accuracy", color="#1f77b4")
    b2 = ax1.bar(x + w / 2, macro, width=w, label="Macro-F1", color="#ff7f0e")
    annotate_bars(ax1, b1)
    annotate_bars(ax1, b2)
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylabel("Score")
    ax1.set_title("Exp3: Fusion Weight Comparison (Top-5 + Latency Spike)")

    ax2 = ax1.twinx()
    l3 = ax2.plot(x, lat, color="#2ca02c", marker="o", linewidth=1.8, label="Latency (ms)")[0]
    for i, v in enumerate(lat):
        ax2.text(i, v + max(1.0, v * 0.01), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Latency (ms)")

    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1 + [l3], l1 + ["Latency (ms)"], loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if old_out1.exists():
        old_out1.unlink()
    if old_out2.exists():
        old_out2.unlink()
    return [out]


def plot_exp4(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> Path:
    out = out_dir / "exp4_prompt_json_quality.png"
    if not rows:
        save_placeholder(out, "Exp4 Prompt/JSON", "No data", dpi)
        return out

    labels = [r.get("item", "unknown") for r in rows]
    json_valid = [as_float(r.get("json_valid_rate")) for r in rows]
    schema_pass = [as_float(r.get("schema_pass_rate")) for r in rows]
    latency = [as_float(r.get("avg_latency_ms")) for r in rows]

    x = np.arange(len(labels))
    w = 0.28
    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    b1 = ax.bar(x - w / 2, json_valid, width=w, label="JSON Valid Rate", color="#1f77b4")
    b2 = ax.bar(x + w / 2, schema_pass, width=w, label="Schema Pass Rate", color="#2ca02c")
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Exp4: Prompt Constraint and JSON Reliability")
    ax.grid(axis="y", alpha=0.3)

    ax2 = ax.twinx()
    l3 = ax2.plot(x, latency, color="#ff7f0e", marker="o", linewidth=1.8, label="Latency (ms)")[0]
    for i, v in enumerate(latency):
        ax2.text(i, v + max(10.0, v * 0.02), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Latency (ms)")

    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1 + [l3], l1 + ["Latency (ms)"], ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_exp5(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> Path:
    out = out_dir / "exp5_robustness.png"
    if not rows:
        save_placeholder(out, "Exp5 Robustness", "No data", dpi)
        return out

    labels = [r.get("item", "unknown") for r in rows]
    success = [as_float(r.get("success_rate")) for r in rows]
    route = [as_float(r.get("route_correct_rate")) for r in rows]
    latency = [as_float(r.get("avg_latency_ms")) for r in rows]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax1 = plt.subplots(figsize=(11.2, 5.5))
    b1 = ax1.bar(x - w / 2, success, width=w, label="Success Rate", color="#2ca02c")
    b2 = ax1.bar(x + w / 2, route, width=w, label="Route Correct Rate", color="#1f77b4")
    annotate_bars(ax1, b1)
    annotate_bars(ax1, b2)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12, ha="right")
    ax1.set_ylabel("Rate")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("Exp5: Robustness Under Fallback Scenarios")
    ax2 = ax1.twinx()
    l3 = ax2.plot(x, latency, color="black", marker="o", linewidth=1.8, label="Latency (ms)")[0]
    for i, v in enumerate(latency):
        ax2.text(i, v + max(10.0, v * 0.02), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Latency (ms)")
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1 + [l3], l1 + ["Latency (ms)"], loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_exp6(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> Path:
    out = out_dir / "exp6_ablation_delta.png"
    if not rows:
        save_placeholder(out, "Exp6 Ablation", "No data", dpi)
        return out

    baseline = None
    for r in rows:
        if r.get("item", "") == "baseline_full":
            baseline = r
            break
    if baseline is None:
        baseline = rows[0]

    b_top1 = as_float(baseline.get("top1"))
    b_macro = as_float(baseline.get("macro_f1"))

    items = [r.get("item", "unknown") for r in rows if r.get("item", "") != baseline.get("item", "")]
    d_top1 = [as_float(r.get("top1")) - b_top1 for r in rows if r.get("item", "") != baseline.get("item", "")]
    d_macro = [as_float(r.get("macro_f1")) - b_macro for r in rows if r.get("item", "") != baseline.get("item", "")]
    fail = [as_float(r.get("failure_rate")) for r in rows if r.get("item", "") != baseline.get("item", "")]

    x = np.arange(len(items))
    w = 0.26
    fig, ax = plt.subplots(figsize=(13.6, 5.8))
    b1 = ax.bar(x - w, d_top1, width=w, label="Delta Top-1 Accuracy", color="#1f77b4")
    b2 = ax.bar(x, d_macro, width=w, label="Delta Macro-F1", color="#ff7f0e")
    b3 = ax.bar(x + w, fail, width=w, label="Failure Rate", color="#d62728", alpha=0.85)
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    annotate_bars(ax, b3)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(items, rotation=20, ha="right")
    ax.set_ylabel("Delta / Rate")
    ax.set_title("Exp6: Ablation Study (Delta vs Baseline)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_exp7(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> Path:
    out = out_dir / "exp7_external_generalization.png"
    if not rows:
        save_placeholder(out, "Exp7 External Generalization", "No data", dpi)
        return out

    labels = [r.get("item", "unknown") for r in rows]
    top1 = [as_float(r.get("top1")) for r in rows]
    top3 = [as_float(r.get("top3")) for r in rows]
    macro = [as_float(r.get("macro_f1")) for r in rows]
    success = [as_float(r.get("success_rate")) for r in rows]

    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    b1 = ax.bar(x - 1.5 * w, top1, width=w, label="Top-1 Accuracy", color="#1f77b4")
    b2 = ax.bar(x - 0.5 * w, top3, width=w, label="Top-3", color="#2ca02c")
    b3 = ax.bar(x + 0.5 * w, macro, width=w, label="Macro-F1", color="#ff7f0e")
    b4 = ax.bar(x + 1.5 * w, success, width=w, label="Success Rate", color="#9467bd")
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    annotate_bars(ax, b3)
    annotate_bars(ax, b4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score / Rate")
    ax.set_title("Exp7: External Generalization on HAM10000")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(summary_path)
    grouped = group_by_exp(rows)

    outputs: list[Path] = []
    outputs.extend(plot_exp1(grouped.get("exp1_modalcompare", []), out_dir, args.dpi))
    outputs.append(plot_exp2(grouped.get("exp2_multimodal", []), out_dir, args.dpi))
    outputs.extend(plot_exp3(grouped.get("exp3_fusion_weight_sweep", []), out_dir, args.dpi))
    outputs.append(plot_exp4(grouped.get("exp4_prompt_json", []), out_dir, args.dpi))
    outputs.append(plot_exp5(grouped.get("exp5_robustness", []), out_dir, args.dpi))
    outputs.append(plot_exp6(grouped.get("exp6_ablation", []), out_dir, args.dpi))
    outputs.append(plot_exp7(grouped.get("exp7_external_generalization", []), out_dir, args.dpi))

    print("Saved figures:")
    for p in outputs:
        print("-", p.resolve())


if __name__ == "__main__":
    main()
