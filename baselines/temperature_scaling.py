"""
baselines/epinet.py
===================
EpiNet — A6.

Reference: Osband et al., "Epistemic Neural Networks", NeurIPS 2023.

Augments a base network with a trainable "epinet" and a fixed random "prior net".
At inference, multiple index samples z ~ N(0,I) are drawn to estimate
epistemic uncertainty via mutual information.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import EpiNet   # re-export


@torch.no_grad()
def get_epinet_outputs(
    model:     EpiNet,
    loader:    DataLoader,
    device:    torch.device,
    n_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect multi-sample predictions from EpiNet.

    Args:
        model:     Trained EpiNet.
        loader:    DataLoader yielding (x, y) batches.
        device:    Torch device.
        n_samples: Number of index samples per input. Default: 20.

    Returns:
        probs_stack: (n_samples, N, C) softmax probabilities.
        labels:      (N,) integer labels.
    """
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        p = model.mc_predict(x.to(device), n_samples)   # (n_samples, B, C)
        all_probs.append(p.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_probs, axis=1), np.concatenate(all_labels)
