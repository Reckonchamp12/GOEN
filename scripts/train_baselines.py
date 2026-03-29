#!/usr/bin/env python3
"""
scripts/train_goen.py
=====================
Train a single GOEN model.

Usage::

    python scripts/train_goen.py --config configs/goen_default.yaml --seed 42
    python scripts/train_goen.py --seed 42 --data_root ./data --output_dir ./ckpts
"""

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goen import Trainer, get_default_config, load_config, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train GOEN")
    parser.add_argument("--config",     type=str, default=None)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--data_root",  type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--p1_epochs",  type=int, default=None)
    parser.add_argument("--fast",       action="store_true",
                        help="Fast mode: 40 epochs, 10 patience (debug)")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else get_default_config()

    # CLI overrides
    if args.seed       is not None: cfg["seed"]       = args.seed
    if args.data_root  is not None: cfg["data_root"]  = args.data_root
    if args.output_dir is not None: cfg["output_dir"] = args.output_dir
    if args.p1_epochs  is not None: cfg["p1_epochs"]  = args.p1_epochs
    if args.fast:
        cfg["p1_epochs"] = 40
        cfg["p1_patience"] = 10
        cfg["p3_epochs"] = 10
        cfg["p3_patience"] = 6

    print(f"[GOEN] Training with seed={cfg['seed']} on {cfg.get('device','auto')}")
    print(f"[GOEN] data_root={cfg['data_root']}  output_dir={cfg['output_dir']}")

    trainer = Trainer(cfg)
    model   = trainer.train()
    results = trainer.evaluate(model)

    # Save
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"goen_seed{cfg['seed']}.pt"
    save_checkpoint(model, ckpt_path, meta={"config": cfg, "results": results})

    res_path = out_dir / f"goen_seed{cfg['seed']}_results.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "═" * 55)
    print("  GOEN TRAINING COMPLETE")
    print("═" * 55)
    print(f"  ID Accuracy:   {results['ID']['Accuracy']:.4f}")
    print(f"  ID ECE:        {results['ID']['ECE']:.4f}")
    print(f"  Avg OOD AUROC: {results['avg_auroc']:.4f}")
    print(f"  Checkpoint:    {ckpt_path}")
    print(f"  Results:       {res_path}")
    print("═" * 55)


if __name__ == "__main__":
    main()
