#!/usr/bin/env python3
"""
scripts/seeding.py
==================
Run the 5-seed GOEN stability experiment.

Reports mean ± std across seeds for all key metrics.

Usage::

    python scripts/seeding.py --seeds 42 123 2024 777 314
    python scripts/seeding.py --config configs/goen_default.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goen import Trainer, get_default_config, load_config

DEFAULT_SEEDS = [42, 123, 2024, 777, 314]


def run_seeding(base_cfg: dict, seeds: list[int]) -> dict:
    """Train GOEN with multiple seeds and aggregate results.

    Args:
        base_cfg: Base configuration (seed will be overridden per run).
        seeds:    List of integer seeds.

    Returns:
        Summary dict with per-seed and aggregated results.
    """
    per_seed: dict = {}

    for seed in seeds:
        cfg = {**base_cfg, "seed": seed}
        print(f"\n{'█'*55}")
        print(f"  Seed: {seed}")
        print(f"{'█'*55}")

        trainer = Trainer(cfg)
        model   = trainer.train()
        results = trainer.evaluate(model)
        per_seed[seed] = results

        print(f"  → ID Acc={results['ID']['Accuracy']:.4f}  "
              f"Avg AUROC={results['avg_auroc']:.4f}")

    # Aggregate
    def _agg(fn) -> tuple[float, float]:
        vals = [fn(per_seed[s]) for s in seeds]
        return float(np.mean(vals)), float(np.std(vals))

    metrics = {
        "ID_Accuracy":   _agg(lambda r: r["ID"]["Accuracy"]),
        "ID_ECE":        _agg(lambda r: r["ID"]["ECE"]),
        "ID_NLL":        _agg(lambda r: r["ID"]["NLL"]),
        "ID_Brier":      _agg(lambda r: r["ID"]["Brier"]),
        "OOD_SVHN":      _agg(lambda r: r.get("OOD-svhn", {}).get("AUROC", float("nan"))),
        "OOD_CIFAR100":  _agg(lambda r: r.get("OOD-cifar100", {}).get("AUROC", float("nan"))),
        "OOD_Synthetic": _agg(lambda r: r.get("OOD-synthetic", {}).get("AUROC", float("nan"))),
        "Avg_AUROC":     _agg(lambda r: r["avg_auroc"]),
        "FPR95_SVHN":    _agg(lambda r: r.get("OOD-svhn", {}).get("FPR95", float("nan"))),
        "FPR95_CIFAR100":_agg(lambda r: r.get("OOD-cifar100", {}).get("FPR95", float("nan"))),
    }

    # Print table
    print("\n" + "═" * 55)
    print("  SEEDING RESULTS  (mean ± std)")
    print(f"  {'Metric':<22} {'Mean':>8}  {'±Std':>8}")
    print("  " + "─" * 42)
    for name, (mean, std) in metrics.items():
        arrow = "↑" if any(k in name for k in ["Accuracy", "AUROC", "OOD_"]) else "↓"
        print(f"  {name:<22} {mean:8.4f}  ±{std:.4f}  {arrow}")

    print(f"\n  Per-seed Avg AUROC:")
    for s in seeds:
        print(f"    seed={s:5d}  {per_seed[s]['avg_auroc']:.4f}")

    return {
        "seeds":       seeds,
        "per_seed":    {str(s): per_seed[s] for s in seeds},
        "aggregated":  {k: {"mean": v[0], "std": v[1]}
                        for k, v in metrics.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="GOEN Seeding Study")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seeds",  type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=str, default="results/seeding_results.json")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else get_default_config()

    summary = run_seeding(cfg, args.seeds)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] {out}")

    m, s = summary["aggregated"]["Avg_AUROC"]["mean"], summary["aggregated"]["Avg_AUROC"]["std"]
    print(f"\n  Final: Avg OOD AUROC = {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
