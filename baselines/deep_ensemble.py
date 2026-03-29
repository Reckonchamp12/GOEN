"""
baselines/standard_nn.py
========================
Standard Neural Network (Softmax baseline) — A1.

Uncertainty score: 1 − max p(y|x)  (Maximum Softmax Probability, Hendrycks & Gimpel 2017).
Higher score → more uncertain / more OOD.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def msp_score(probs: np.ndarray) -> np.ndarray:
    """Maximum Softmax Probability uncertainty: u(x) = 1 − max_c p(c|x).

    Args:
        probs: Softmax probabilities, shape (N, C).

    Returns:
        scores: (N,) — higher = more OOD.
    """
    return 1.0 - probs.max(axis=1)


@torch.no_grad()
def get_softmax_outputs(
    model:  torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect softmax probabilities and labels over a DataLoader.

    Args:
        model:  Trained classification model.
        loader: DataLoader yielding (x, y) batches.
        device: Torch device.

    Returns:
        probs:  (N, C) softmax probabilities.
        labels: (N,) integer labels.
    """
    model.eval()
    probs_list, labels_list = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs_list.append(F.softmax(logits, dim=-1).cpu().numpy())
        labels_list.append(y.numpy())
    return np.concatenate(probs_list), np.concatenate(labels_list)
