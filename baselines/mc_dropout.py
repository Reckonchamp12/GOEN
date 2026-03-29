"""
baselines/deep_ensemble.py
==========================
Deep Ensemble — A3.

Reference: Lakshminarayanan et al., "Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles", NeurIPS 2017.

Trains K independent ResNet-18 models from different random seeds.
Predictive uncertainty = total variance of the ensemble's class probabilities.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import ResNet18


def build_ensemble(
    k:          int = 5,
    num_classes: int = 10,
) -> list[ResNet18]:
    """Create K freshly initialised ResNet-18 models.

    Args:
        k:           Number of ensemble members.
        num_classes: Number of output classes.

    Returns:
        List of K ResNet-18 models (untrained).
    """
    return [ResNet18(num_classes=num_classes) for _ in range(k)]


@torch.no_grad()
def ensemble_predict(
    members: list[ResNet18],
    loader:  DataLoader,
    device:  torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference for all ensemble members.

    Args:
        members: List of trained ResNet-18 models.
        loader:  DataLoader yielding (x, y) batches.
        device:  Torch device.

    Returns:
        probs_stack: (K, N, C) per-member softmax probabilities.
        mean_probs:  (N, C)   ensemble mean probabilities.
        labels:      (N,)     ground-truth labels.
    """
    member_probs: list[np.ndarray] = []
    labels_arr: np.ndarray | None  = None

    for mem in members:
        mem.eval()
        ps, ls = [], []
        for x, y in loader:
            ps.append(F.softmax(mem(x.to(device)), dim=-1).cpu().numpy())
            ls.append(y.numpy())
        member_probs.append(np.concatenate(ps))
        if labels_arr is None:
            labels_arr = np.concatenate(ls)

    stack = np.stack(member_probs)   # (K, N, C)
    return stack, stack.mean(axis=0), labels_arr
