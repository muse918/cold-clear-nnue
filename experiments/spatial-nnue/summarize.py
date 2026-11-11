#!/usr/bin/env python3
"""Summarize recorded measurements; this script never trains or benchmarks."""
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "results/final"
SEEDS = (918, 919, 920)
NAMES = ("cell", "cnn", "attention")


def read(path):
    return json.loads(path.read_text())


def spread(values):
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values),
            "values": values, "interpretation": "Training-seed range, not a confidence interval."}


def main():
    evaluation = read(FINAL / "evaluation.json")
    result = {"quality_scope": "Hard-drop surrogate teacher and capped solo games; not TETR.IO versus strength.",
              "seeds": list(SEEDS), "heldout_records": evaluation["heldout_records"],
              "models": {}, "decision": {
                  "primary_architecture": "CNN-compiled local LUT with fixed-query attention pooling",
                  "arithmetic": "float32",
                  "ordinary_lock": "sparse dirty-patch and positional-cell accumulator update",
                  "default_event_path": "full refresh on clear or garbage",
                  "optional_event_path": "verified row-source seam reuse; no material total speedup established",
                  "checkpoint_policy": "seed918 is the fixed representative, not chosen from heldout/rollout scores",
                  "status": "research evaluator; full battle state, Rust integration and match validation remain"}}
    metrics = ("rmse", "mean_teacher_regret", "top_action_agreement")
    rollout_metrics = ("mean_pieces", "mean_lines", "capped_survival_rate")
    for name in NAMES:
        entries = [evaluation["models"][f"{name}_seed{seed}"] for seed in SEEDS]
        native = read(FINAL / f"seed918/{name}_native_heldout_benchmark.json")
        search = read(FINAL / f"seed918/{name}_search_timing.json")
        result["models"][name] = {
            "ranking": {m: spread([e["ranking"][m] for e in entries]) for m in metrics},
            "rollout": {m: spread([e["rollout"]["overall"][m] for e in entries]) for m in rollout_metrics},
            "native_model_bytes": native["model_bytes"],
            "active_float_accumulator_bytes": native["accumulator_active_bytes"],
            "allocated_float_accumulator_bytes": native["accumulator_storage_bytes"],
            "cached_parent_timings": native["update_and_head_parent_precomputed"],
            "sibling_timings": native["sibling_pipeline_including_parent_refresh"],
            "one_ply_timings": search["timings"],
            "quantization": {}}
        for bits in (8, 16):
            quant = read(FINAL / f"seed918/{name}_quantized_q{bits}_timing.json")
            result["models"][name]["quantization"][str(bits)] = {
                "memory": quant["memory"], "timings": quant["timings"],
                "float_comparison": quant["float_comparison"]}
    for name in ("dense_cnn", "transformer"):
        result["models"][name] = evaluation["models"][f"{name}_seed918"]
    result["teacher_rollout"] = evaluation["surrogate_teacher_rollout"]
    (FINAL / "summary.json").write_text(json.dumps(result, indent=2))
    plot(result)
    print(json.dumps({name: {"regret": result["models"][name]["ranking"]["mean_teacher_regret"]["mean"],
                              "pieces": result["models"][name]["rollout"]["mean_pieces"]["mean"]}
                      for name in NAMES}, indent=2))


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.dpi": 160})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.9), layout="constrained")
    colors = ["#64748b", "#2563eb", "#0d9488"]
    labels = ["Cell control", "CNN LUT", "Attention LUT"]
    x = np.arange(3)
    ax = axes[0, 0]
    for i, name in enumerate(NAMES):
        data = result["models"][name]["ranking"]["mean_teacher_regret"]
        ax.bar(i, data["mean"], color=colors[i], alpha=.85, width=.55)
        ax.scatter(i + np.array([-.12, 0, .12]), data["values"], c="black", s=22, zorder=3)
        ax.text(i, data["max"] + .15, f'{data["mean"]:.2f}', ha="center")
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 7), ylabel="Mean teacher regret (lower is better)",
           title="A. Held-out action quality: 3 training seeds")
    ax.text(.02, .03, "Dots: seed 918 / 919 / 920; 1,265 parent decisions", transform=ax.transAxes, fontsize=9)

    ax = axes[0, 1]
    for i, name in enumerate(NAMES):
        data = result["models"][name]["rollout"]["mean_pieces"]
        ax.bar(i, data["mean"], color=colors[i], alpha=.85, width=.55)
        ax.scatter(i + np.array([-.12, 0, .12]), data["values"], c="black", s=22, zorder=3)
        ax.text(i, data["max"] + 5, f'{data["mean"]:.1f}', ha="center")
    teacher = result["teacher_rollout"]["overall"]["mean_pieces"]
    ax.axhline(teacher, color="#9a3412", ls="--", lw=1.4)
    ax.text(.02, .95, f"2-ply surrogate teacher: {teacher:.1f}", transform=ax.transAxes,
            color="#9a3412", va="top", fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 180), ylabel="Mean pieces survived (cap: 160)",
           title="B. Matched 24-game surrogate rollouts")
    ax.text(.02, .03, "Same queues / garbage for every model; not versus win rate", transform=ax.transAxes, fontsize=9)

    ax = axes[1, 0]
    methods = ("full_refresh", "incremental_refresh_events", "incremental_row_seams")
    method_labels = ("Full refresh", "Incremental + event refresh", "Incremental + row seams")
    method_colors = ("#94a3b8", "#2563eb", "#0d9488")
    for j, method in enumerate(methods):
        y = [result["models"][name]["one_ply_timings"][method]["median_ns_per_candidate"] / 1000 for name in NAMES]
        ax.bar(x + (j - 1) * .24, y, width=.22, color=method_colors[j], label=method_labels[j])
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 2.05), ylabel="Microseconds / generated candidate",
           title="C. One-ply generation + evaluation + selection")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(.02, .02, "Median of 9 run averages; warm cache, shared CPU", transform=ax.transAxes, fontsize=9)

    ax = axes[1, 1]
    xx = np.arange(2)
    for j, precision in enumerate(("float", "8", "16")):
        y = []
        for name in ("cnn", "attention"):
            data = result["models"][name]
            value = data["cached_parent_timings"]["incremental_row_seams"]["all"]["median_ns"] if precision == "float" else data["quantization"][precision]["timings"]["incremental_row_seams"]["median_ns"]
            y.append(value / 1000)
        ax.bar(xx + (j - 1) * .24, y, width=.22, color=method_colors[j],
               label={"float": "Float32", "8": "q8 weights / int16 accumulator", "16": "q16 weights / int32 accumulator"}[precision])
    ax.set(xticks=xx, xticklabels=labels[1:], ylim=(0, 1.5), ylabel="Microseconds / evaluation",
           title="D. First-layer quantization: no latency gain")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(.02, .02, "Cached parent; row seams; float head in all variants", transform=ax.transAxes, fontsize=9)
    for ax in axes.ravel():
        ax.yaxis.grid(True, color="#e2e8f0", lw=.7)
        ax.set_axisbelow(True)
    fig.suptitle("Spatial NNUE experiment — measured tradeoffs", fontsize=17, weight="bold")
    out = FINAL / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "comparison.png", dpi=180)
    fig.savefig(out / "comparison.svg")
    # Matplotlib's path serializer adds cosmetic trailing spaces to SVG lines.
    svg = out / "comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
