"""
baselines/moe.py
================
Mixture of Experts — A7.

Soft-gated mixture: K expert classification heads on top of a shared
ResNet-18 backbone, with a linear gating network.

Uncertainty: 1 − max gate weight  (low routing confidence → uncertain).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import MoENet   # re-export


@torch.no_grad()
def get_moe_outputs(
    model:  MoENet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect predictions and gate weights from MoENet.

    Args:
        model:  Trained MoENet.
        loader: DataLoader yielding (x, y) batches.
        device: Torch device.

    Returns:
        probs:  (N, C) softmax probabilities from the mixture.
        gates:  (N, K) soft gate weights per expert.
        labels: (N,) integer labels.
    """
    model.eval()
    all_probs, all_gates, all_labels = [], [], []
    for x, y in loader:
        logits, gate = model(x.to(device), return_gate=True)
        all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        all_gates.append(gate.cpu().numpy())
        all_labels.append(y.numpy())
    return (
        np.concatenate(all_probs),
        np.concatenate(all_gates),
        np.concatenate(all_labels),
    )


def gate_entropy_score(gates: np.ndarray) -> np.ndarray:
    """Routing entropy as an alternative uncertainty score.

    H[g] = −Σ_k g_k log g_k. Higher → more uncertain routing.

    Args:
        gates: (N, K) gate weight array.

    Returns:
        entropy: (N,) routing entropy scores.
    """
    eps = 1e-12
    return -(gates * np.log(gates + eps)).sum(axis=1)
