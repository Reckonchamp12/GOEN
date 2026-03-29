#!/usr/bin/env python3
"""
scripts/ablation.py
===================
Run the GOEN ablation study: 4 variants × seed=42.

Variants
--------
  GOEN-Default      All components active.
  NoCenterLoss      center_alpha = 0  (disable CenterLoss in Phase 1).
  SingleScale-L4    single_scale = True  (Layer4 features only).
  NoSVHN-NoiseOnly  svhn_ood = False  (Gaussian noise only in Phase 3).

Usage::

    python scripts/ablation.py --config configs/goen_default.yaml --seed 42
"""

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goen import Trainer, get_default_config, load_config, set_seed
from goen.data import get_cifar10_loaders, get_ood_loaders, get_svhn_ood_loader

ABLATION_GRID = [
    {"name": "GOEN-Default",    "center_alpha": 0.01, "single_scale": False, "svhn_ood": True},
    {"name": "NoCenterLoss",    "center_alpha": 0.00, "single_scale": False, "svhn_ood": True},
    {"name": "SingleScale-L4",  "center_alpha": 0.01, "single_scale": True,  "svhn_ood": True},
    {"name": "NoSVHN-NoiseOnly","center_alpha": 0.01, "single_scale": False, "svhn_ood": False},
]


def run_ablation(base_cfg: dict) -> dict:
    """Train and evaluate all ablation variants.

    The shared CIFAR-10 loaders are built once and reused across all variants
    to ensure a fair comparison.

    Args:
        base_cfg: Base configuration dict.

    Returns:
        ablation_results: Dict mapping variant name → result dict.
    """
    # Build loaders once
    train_l, val_l, feat_l, test_l = get_cifar10_loaders(base_cfg)
    ood_loaders = get_ood_loaders(base_cfg)
    svhn_l      = get_svhn_ood_loader(base_cfg) if base_cfg.get("svhn_ood", True) else None

    ablation_results: dict = {}

    for abl in ABLATION_GRID:
        cfg = {**base_cfg, **{k: v for k, v in abl.items() if k != "name"}}
        tag = abl["name"]

        print(f"\n{'█'*60}")
        print(f"  Variant: {tag}")
        print(f"  center_alpha={cfg['center_alpha']}  "
              f"single_scale={cfg['single_scale']}  "
              f"svhn_ood={cfg['svhn_ood']}")
        print(f"{'█'*60}")

        set_seed(cfg["seed"])
        trainer = Trainer(cfg)
        model   = trainer.train(
            train_l=train_l, val_l=val_l, feat_l=feat_l,
            svhn_l=svhn_l if cfg["svhn_ood"] else None,
        )
        results = trainer.evaluate(model, test_l=test_l, ood_loaders=ood_loaders)
        ablation_results[tag] = results

    # Print comparison table
    print("\n" + "═" * 68)
    print("  ABLATION RESULTS")
    print(f"  {'Model':<22} {'Acc':>6} {'SVHN':>7} {'C100':>7} {'Syn':>7} {'Avg':>8}")
    print("  " + "─" * 58)
    for name, res in ablation_results.items():
        acc  = res["ID"]["Accuracy"]
        sv   = res.get("OOD-svhn", {}).get("AUROC", float("nan"))
        c1   = res.get("OOD-cifar100", {}).get("AUROC", float("nan"))
        sy   = res.get("OOD-synthetic", {}).get("AUROC", float("nan"))
        avg  = res["avg_auroc"]
        mark = " ★" if name == "GOEN-Default" else "  "
        print(f"  {name:<22}{mark} {acc:.4f} {sv:.4f} {c1:.4f} {sy:.4f} {avg:.4f}")

    default_avg = ablation_results["GOEN-Default"]["avg_auroc"]
    print("\n  Contribution (AUROC drop when component removed):")
    for name, res in ablation_results.items():
        if name == "GOEN-Default":
            continue
        drop = default_avg - res["avg_auroc"]
        sign = "+" if drop > 0 else ""
        print(f"    Removing {name:<22} → Δ = {sign}{drop:+.4f}")

    return ablation_results


def main():
    parser = argparse.ArgumentParser(description="GOEN Ablation Study")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/ablation_results.json")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else get_default_config()
    cfg["seed"] = args.seed

    results = run_ablation(cfg)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
