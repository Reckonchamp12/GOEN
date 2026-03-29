"""
baselines/odin.py
=================
ODIN: Out-of-DIstribution detector for Neural networks — B2.

Reference: Liang et al., "Enhancing the Reliability of Out-of-distribution
Image Detection in Neural Networks", ICLR 2018.

Two modifications to the standard softmax detector:
  1. Temperature scaling: divide logits by T >> 1 (sharpens distribution).
  2. Input pre-processing: subtract ε · sign(∇_x log max-softmax) from input.
     This nudges the input toward the decision boundary for ID samples,
     increasing the max-softmax gap between ID and OOD.

Uncertainty = 1 − max-ODIN-softmax. Higher → more OOD.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import odin_score   # re-export from __init__


def tune_odin(
    model:        torch.nn.Module,
    val_id:       DataLoader,
    val_ood:      DataLoader,
    device:       torch.device,
    temps:        list[float] = (1.0, 10.0, 100.0, 1000.0),
    epsilons:     list[float] = (0.0, 0.0002, 0.0014, 0.005),
) -> tuple[float, float]:
    """Grid search for best ODIN temperature and epsilon on a validation set.

    Args:
        model:    Trained classifier.
        val_id:   Small ID validation loader (subset).
        val_ood:  Small OOD validation loader (subset).
        device:   Torch device.
        temps:    Temperature values to try.
        epsilons: Epsilon values to try.

    Returns:
        best_T, best_eps: Optimal hyperparameters.
    """
    from sklearn.metrics import roc_auc_score

    best_auroc = 0.0
    best_T     = 1000.0
    best_eps   = 0.0014

    for T in temps:
        for eps in epsilons:
            id_sc  = odin_score(model, val_id,  device, T, eps)
            ood_sc = odin_score(model, val_ood, device, T, eps)
            y      = np.r_[np.zeros(len(id_sc)), np.ones(len(ood_sc))]
            s      = np.r_[id_sc, ood_sc]
            try:
                auroc = float(roc_auc_score(y, s))
            except Exception:
                continue
            if auroc > best_auroc:
                best_auroc = auroc
                best_T     = T
                best_eps   = eps

    print(f"  [ODIN] Best T={best_T}  eps={best_eps}  AUROC={best_auroc:.4f}")
    return best_T, best_eps
