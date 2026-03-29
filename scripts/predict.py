#!/usr/bin/env python3
"""
scripts/make_plots.py
=====================
Generate all publication-quality figures from saved result JSONs.

Figures produced
----------------
  figures/fig1_ood_comparison.png     — Bar chart: all models avg OOD AUROC
  figures/fig2_id_metrics.png         — Scatter: Accuracy vs ECE for all models
  figures/fig3_ood_heatmap.png        — Heatmap: per-dataset AUROC for all models
  figures/fig4_ablation.png           — Ablation bar chart (GOEN variants)
  figures/fig5_seeding.png            — Seeding study with error bars
  figures/fig6_score_hist.png         — Uncertainty score distributions (if available)
  figures/fig7_feature_geometry.png   — Compactness ratio bar chart

Usage::

    python scripts/make_plots.py --results_dir ./results --output_dir ./figures

    # Or pass individual JSONs:
    python scripts/make_plots.py \
        --baselines  results/baselines/all_results.json \
        --goen       results/goen/goen_seed42_results.json \
        --ablation   results/goen/ablation_results.json \
        --seeding    results/goen/seeding_results.json \
        --output_dir figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})

PALETTE = {
    "GOEN":          "#2ecc71",
    "baseline":      "#95a5a6",
    "highlight":     "#e74c3c",
    "secondary":     "#3498db",
}

# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def _load(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def _auto_find(results_dir: Path, filename: str) -> Path | None:
    """Search recursively for a filename inside results_dir."""
    matches = list(results_dir.rglob(filename))
    return matches[0] if matches else None


# ─────────────────────────────────────────────────────────────
# Figure 1: OOD comparison bar chart
# ─────────────────────────────────────────────────────────────

def fig1_ood_comparison(baseline_data: dict, goen_avg: float, out: Path) -> None:
    """Horizontal bar chart of avg OOD AUROC for all methods."""

    # Extract avg AUROC per baseline model
    model_scores: dict[str, float] = {}
    for model_name, groups in baseline_data.items():
        aurocs = [v["AUROC"] for k, v in groups.items() if k.startswith("OOD-")]
        if aurocs:
            model_scores[model_name] = float(np.mean(aurocs))

    model_scores["GOEN (ours)"] = goen_avg

    # Sort ascending
    sorted_items = sorted(model_scores.items(), key=lambda x: x[1])
    names  = [k for k, _ in sorted_items]
    scores = [v for _, v in sorted_items]
    colors = [PALETTE["GOEN"] if "GOEN" in n else PALETTE["baseline"] for n in names]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.55)))
    bars = ax.barh(names, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Average OOD AUROC", fontsize=11)
    ax.set_title("OOD Detection: All Methods vs GOEN", fontsize=13, fontweight="bold")
    ax.set_xlim(max(0, min(scores) - 0.05), 1.02)

    for bar, val in zip(bars, scores):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    # Reference line at KNN (previous best open-source)
    if "KNN" in model_scores:
        ax.axvline(model_scores["KNN"], color=PALETTE["highlight"],
                   linestyle="--", linewidth=1.2, alpha=0.7, label="KNN (prev. best)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 2: Accuracy vs ECE scatter
# ─────────────────────────────────────────────────────────────

def fig2_id_scatter(baseline_data: dict, goen_id: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, groups in baseline_data.items():
        if "ID-CIFAR10" in groups:
            m   = groups["ID-CIFAR10"]
            acc = m.get("Accuracy", None)
            ece = m.get("ECE", None)
            if acc is not None and ece is not None:
                ax.scatter(ece, acc, color=PALETTE["baseline"], s=80, zorder=3)
                ax.annotate(model_name, (ece, acc), textcoords="offset points",
                            xytext=(5, 3), fontsize=7, color="#555")

    # GOEN
    ax.scatter(goen_id["ECE"], goen_id["Accuracy"],
               color=PALETTE["GOEN"], s=160, zorder=5, marker="*",
               label=f"GOEN  (Acc={goen_id['Accuracy']:.3f}, ECE={goen_id['ECE']:.3f})")

    ax.set_xlabel("ECE (↓ better)", fontsize=11)
    ax.set_ylabel("Accuracy (↑ better)", fontsize=11)
    ax.set_title("In-Distribution: Accuracy vs Calibration", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 3: Per-dataset AUROC heatmap
# ─────────────────────────────────────────────────────────────

def fig3_heatmap(baseline_data: dict, goen_ood: dict, out: Path) -> None:
    ood_keys = ["OOD-svhn", "OOD-cifar100", "OOD-synthetic"]
    ood_labels = ["SVHN", "CIFAR-100", "Synthetic"]

    all_models = list(baseline_data.keys()) + ["GOEN (ours)"]
    matrix = []
    for model_name in all_models:
        row = []
        if model_name == "GOEN (ours)":
            src = goen_ood
        else:
            src = baseline_data.get(model_name, {})
        for k in ood_keys:
            # normalise key format
            auroc = src.get(k, src.get(k.replace("OOD-", "OOD-"), {})
                             ).get("AUROC", float("nan")) if isinstance(
                src.get(k, {}), dict) else float("nan")
            row.append(auroc)
        matrix.append(row)

    mat = np.array(matrix)
    fig, ax = plt.subplots(figsize=(7, max(4, len(all_models) * 0.45)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(ood_labels))); ax.set_xticklabels(ood_labels, fontsize=10)
    ax.set_yticks(range(len(all_models))); ax.set_yticklabels(all_models, fontsize=8)
    ax.set_title("OOD AUROC per Dataset", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="AUROC")

    for i in range(len(all_models)):
        for j in range(len(ood_labels)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="black" if 0.6 < v < 0.95 else "white")

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 4: Ablation
# ─────────────────────────────────────────────────────────────

def fig4_ablation(ablation_data: dict, out: Path) -> None:
    names, avgs, accs = [], [], []
    for name, res in ablation_data.items():
        names.append(name)
        avgs.append(res.get("avg_auroc", float("nan")))
        accs.append(res.get("ID", {}).get("Accuracy", float("nan")))

    colors = [PALETTE["GOEN"] if "Default" in n else PALETTE["highlight"] for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("GOEN Ablation Study", fontsize=13, fontweight="bold")

    for ax, vals, ylabel, title in [
        (ax1, avgs, "Avg OOD AUROC", "OOD Detection"),
        (ax2, accs, "ID Accuracy",   "In-Distribution Accuracy"),
    ]:
        bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ylim_lo = max(0, min(v for v in vals if not np.isnan(v)) - 0.05)
        ax.set_ylim(ylim_lo, 1.02)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                        f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 5: Seeding study
# ─────────────────────────────────────────────────────────────

def fig5_seeding(seeding_data: dict, out: Path) -> None:
    agg   = seeding_data.get("aggregated", {})
    seeds = seeding_data.get("seeds", [])

    metric_keys = ["ID_Accuracy", "Avg_AUROC", "OOD_SVHN", "OOD_CIFAR100", "OOD_Synthetic"]
    metric_labels = ["ID Accuracy", "Avg AUROC", "SVHN AUROC", "CIFAR-100 AUROC", "Synthetic AUROC"]

    means = [agg.get(k, {}).get("mean", float("nan")) for k in metric_keys]
    stds  = [agg.get(k, {}).get("std",  float("nan")) for k in metric_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"GOEN Seeding Study  ({len(seeds)} seeds)", fontsize=13, fontweight="bold")

    # Left: summary bar with error bars
    x = np.arange(len(metric_keys))
    ax1.bar(x, means, yerr=stds, capsize=5,
            color=PALETTE["secondary"], edgecolor="white", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(metric_labels, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Score"); ax1.set_title("Mean ± Std across seeds")
    ax1.set_ylim(0, 1.1)
    for i, (m, s) in enumerate(zip(means, stds)):
        if not np.isnan(m):
            ax1.text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=8)

    # Right: per-seed avg AUROC bar
    per_seed = seeding_data.get("per_seed", {})
    seed_labels = [str(s) for s in seeds]
    seed_vals   = [per_seed.get(str(s), {}).get("avg_auroc",
                    per_seed.get(str(s), float("nan"))
                    if isinstance(per_seed.get(str(s)), float) else float("nan"))
                   for s in seeds]

    ax2.bar(seed_labels, seed_vals,
            color=[PALETTE["GOEN"] if i == 0 else PALETTE["secondary"]
                   for i in range(len(seeds))],
            edgecolor="white")
    if not np.isnan(means[1]):
        ax2.axhline(means[1], color="black", linestyle="--", linewidth=1.5,
                    label=f"Mean = {means[1]:.4f}")
    ax2.set_xlabel("Seed"); ax2.set_ylabel("Avg OOD AUROC")
    ax2.set_title("Per-seed Avg OOD AUROC")
    lo = min(v for v in seed_vals if not np.isnan(v)) - 0.02 if seed_vals else 0
    ax2.set_ylim(max(0, lo), 1.02)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate GOEN paper figures")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Root directory to search for result JSONs.")
    parser.add_argument("--baselines",   type=str, default=None)
    parser.add_argument("--goen",        type=str, default=None)
    parser.add_argument("--ablation",    type=str, default=None)
    parser.add_argument("--seeding",     type=str, default=None)
    parser.add_argument("--output_dir",  type=str, default="./figures")
    args = parser.parse_args()

    rdir  = Path(args.results_dir)
    odir  = Path(args.output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    # Auto-discover JSONs if not explicitly provided
    def _resolve(explicit, name):
        return Path(explicit) if explicit else _auto_find(rdir, name)

    bl_path  = _resolve(args.baselines, "all_results.json")
    gn_path  = _resolve(args.goen,      "goen_seed42_results.json")
    abl_path = _resolve(args.ablation,  "ablation_results.json")
    sd_path  = _resolve(args.seeding,   "seeding_results.json")

    baseline_data = _load(bl_path)
    goen_data     = _load(gn_path)
    ablation_data = _load(abl_path)
    seeding_data  = _load(sd_path)

    print(f"[make_plots] Output dir: {odir}")
    print(f"  baselines : {bl_path  or 'NOT FOUND'}")
    print(f"  goen      : {gn_path  or 'NOT FOUND'}")
    print(f"  ablation  : {abl_path or 'NOT FOUND'}")
    print(f"  seeding   : {sd_path  or 'NOT FOUND'}")

    # Extract GOEN numbers
    goen_avg  = float("nan")
    goen_id   = {}
    goen_ood  = {}
    if goen_data:
        inner = goen_data.get("GOEN (ours)", goen_data)
        goen_id  = inner.get("ID", inner.get("ID-CIFAR10", {}))
        goen_avg = inner.get("avg_auroc", float("nan"))
        goen_ood = {k: v for k, v in inner.items() if k.startswith("OOD")}

    generated = []

    if baseline_data:
        if not np.isnan(goen_avg):
            fig1_ood_comparison(baseline_data, goen_avg, odir / "fig1_ood_comparison.png")
            generated.append("fig1_ood_comparison.png")
        if goen_id:
            fig2_id_scatter(baseline_data, goen_id, odir / "fig2_id_scatter.png")
            generated.append("fig2_id_scatter.png")
        if goen_ood:
            fig3_heatmap(baseline_data, goen_ood, odir / "fig3_ood_heatmap.png")
            generated.append("fig3_ood_heatmap.png")

    if ablation_data:
        fig4_ablation(ablation_data, odir / "fig4_ablation.png")
        generated.append("fig4_ablation.png")

    if seeding_data:
        fig5_seeding(seeding_data, odir / "fig5_seeding.png")
        generated.append("fig5_seeding.png")

    if not generated:
        print("\n  [warn] No result JSONs found — no figures generated.")
        print("  Run the training scripts first, then call make_plots.py.")
    else:
        print(f"\n[make_plots] Generated {len(generated)} figure(s) → {odir}/")
        for f in generated:
            print(f"  ✓ {f}")


if __name__ == "__main__":
    main()
