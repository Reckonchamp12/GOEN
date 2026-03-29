"""
goen/utils.py
=============
Miscellaneous utilities: reproducibility, logging, checkpoints, config.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

def get_default_config() -> dict:
    """Return the default GOEN configuration dictionary.

    Returns:
        cfg: Dict with all hyperparameters and paths.
    """
    return dict(
        # Reproducibility
        seed          = 42,

        # Data
        data_root     = "./data",
        output_dir    = "./checkpoints",
        batch_size    = 128,
        num_classes   = 10,
        val_size      = 5000,

        # Phase 1: CE + CenterLoss
        p1_epochs     = 80,
        p1_lr         = 0.1,
        p1_momentum   = 0.9,
        p1_wd         = 5e-4,
        center_lr     = 0.5,
        center_alpha  = 0.01,
        p1_patience   = 20,

        # Architecture
        proj_dim      = 512,
        single_scale  = False,

        # Phase 3: Calibration
        p3_epochs     = 20,
        p3_lr         = 1e-3,
        p3_patience   = 10,
        id_target     = 0.05,
        ood_target    = 0.95,
        svhn_ood      = True,
        svhn_n        = 5000,

        # Runtime
        device        = "cuda" if torch.cuda.is_available() else "cpu",
    )


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and merge with defaults.

    Args:
        path: Path to YAML config file.

    Returns:
        cfg: Merged config dict.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load YAML configs: pip install pyyaml") from exc

    cfg = get_default_config()
    with open(path) as f:
        overrides = yaml.safe_load(f) or {}
    cfg.update(overrides)
    return cfg


# ─────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────

def save_checkpoint(model: torch.nn.Module, path: str | Path, meta: dict | None = None) -> None:
    """Save model state dict and optional metadata.

    Args:
        model: Trained model.
        path:  Output path (will create parent dirs).
        meta:  Optional metadata dict (e.g., metrics, config).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if meta is not None:
        payload["meta"] = meta
    torch.save(payload, path)


def load_pretrained(path: str | Path, cfg: dict | None = None) -> "GOEN":  # noqa: F821
    """Load a GOEN model from a checkpoint file.

    Args:
        path: Path to checkpoint (.pt file).
        cfg:  Optional config dict; uses defaults if None.

    Returns:
        Loaded GOEN model in eval mode.
    """
    from .model import GOEN

    cfg      = cfg or get_default_config()
    payload  = torch.load(path, map_location="cpu")
    sd       = payload["state_dict"] if "state_dict" in payload else payload

    model = GOEN(
        num_classes=cfg.get("num_classes", 10),
        proj_dim=cfg.get("proj_dim", 512),
        single_scale=cfg.get("single_scale", False),
    )
    model.load_state_dict(sd)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────

class Logger:
    """Minimal structured logger for training output."""

    def section(self, *lines: str) -> None:
        print(f"\n{'═'*65}")
        for line in lines:
            print(f"  {line}")
        print(f"{'═'*65}")

    def info(self, msg: str) -> None:
        print(msg)

    def step(self, ep: int, total: int, **kwargs) -> None:
        parts = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in kwargs.items())
        print(f"  ep {ep:3d}/{total}  {parts}")


# ─────────────────────────────────────────────────────────────
# Results book-keeping
# ─────────────────────────────────────────────────────────────

class ResultsBook:
    """Accumulate and persist benchmark results as JSON."""

    def __init__(self) -> None:
        self._data: dict = {}

    def record(self, model: str, group: str, metrics: dict) -> None:
        self._data.setdefault(model, {})[group] = {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in metrics.items()
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._data, f, indent=2)
        print(f"\n[Results] Saved → {path}")

    def print_summary(self) -> None:
        print("\n" + "═" * 72)
        print("  BENCHMARK RESULTS SUMMARY")
        print("═" * 72)
        for model_name, groups in self._data.items():
            print(f"\n  ┌─ {model_name}")
            for group, metrics in groups.items():
                print(f"  │  [{group}]")
                for k, v in metrics.items():
                    v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    print(f"  │    {k:<24} {v_str}")
        print("═" * 72)
