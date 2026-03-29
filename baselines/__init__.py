"""
baselines/edl.py
================
Evidential Deep Learning — A5.

Reference: Sensoy et al., "Evidential Deep Learning to Quantify
Classification Uncertainty", NeurIPS 2018.

Models the class probabilities as a Dirichlet distribution.
The network outputs evidence e_c ≥ 0, and alpha_c = e_c + 1.

Uncertainty (vacuity): u(x) = K / S  where S = Σ_c alpha_c.
High vacuity → model has collected little evidence → OOD or ambiguous input.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import EDLNet, edl_loss   # re-export


@torch.no_grad()
def get_alpha(
    model:  EDLNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect Dirichlet alpha parameters over a DataLoader.

    Args:
        model:  Trained EDLNet.
        loader: DataLoader yielding (x, y) batches.
        device: Torch device.

    Returns:
        alpha:  (N, C) Dirichlet parameters.
        labels: (N,) integer labels.
    """
    model.eval()
    all_alpha, all_labels = [], []
    for x, y in loader:
        all_alpha.append(model(x.to(device)).cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_alpha), np.concatenate(all_labels)


def vacuity(alpha: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Compute per-sample vacuity u(x) = K / S.

    Args:
        alpha:       Dirichlet alpha parameters, shape (N, C).
        num_classes: K — number of classes.

    Returns:
        u: (N,) vacuity scores. Higher → more uncertain / more OOD.
    """
    return num_classes / alpha.sum(axis=1)
