"""
baselines/energy.py
===================
Energy Score — B1.

Reference: Liu et al., "Energy-based Out-of-distribution Detection",
NeurIPS 2020.

E(x; T) = −T · log Σ_c exp(f_c(x) / T)

Higher energy → model assigns lower log-partition-function → more OOD.
Unlike MSP, the energy score uses the full logit distribution (not just the max).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import energy_score   # re-export from __init__


@torch.no_grad()
def get_energy_scores(
    model:  torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    T:      float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute energy scores over a DataLoader.

    Args:
        model:  Trained classifier.
        loader: DataLoader yielding (x, y) batches.
        device: Torch device.
        T:      Temperature. Default: 1.0.

    Returns:
        scores: (N,) energy scores. Higher = more OOD.
        labels: (N,) integer labels.
    """
    model.eval()
    all_scores, all_labels = [], []
    for x, y in loader:
        logits = model(x.to(device)).cpu().numpy()
        all_scores.append(energy_score(logits, T))
        all_labels.append(y.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)
