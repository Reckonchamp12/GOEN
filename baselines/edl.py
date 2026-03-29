"""
baselines/mc_dropout.py
=======================
Monte Carlo Dropout — A2.

Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016.

At test time, dropout is kept active and T stochastic forward passes are run.
Epistemic uncertainty is estimated via mutual information between the model
outputs and the model parameters.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import MCDropoutNet


@torch.no_grad()
def get_mc_outputs(
    model:    MCDropoutNet,
    loader:   DataLoader,
    device:   torch.device,
    n_passes: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect T stochastic forward passes for all batches.

    Args:
        model:    MCDropoutNet with dropout active at test time.
        loader:   DataLoader yielding (x, y) batches.
        device:   Torch device.
        n_passes: Number of stochastic forward passes T.

    Returns:
        probs_stack: (T, N, C) stochastic softmax probabilities.
        labels:      (N,) integer labels.
    """
    all_probs, all_labels = [], []
    for x, y in loader:
        p = model.mc_forward(x.to(device), n_passes)   # (T, B, C)
        all_probs.append(p.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_probs, axis=1), np.concatenate(all_labels)
