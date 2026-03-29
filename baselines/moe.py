"""
baselines/temperature_scaling.py
=================================
Temperature Scaling — A4.

Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.

Post-hoc calibration method that learns a single scalar temperature T > 0
to divide logits, minimising NLL on the validation set via LBFGS.
Does not change accuracy; only improves calibration (ECE).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from . import TemperatureScaler   # re-export from __init__

__all__ = ["TemperatureScaler", "fit_temperature"]


def fit_temperature(
    model:      nn.Module,
    val_loader: DataLoader,
    device:     torch.device,
    init_temp:  float = 1.5,
    lr:         float = 0.01,
    max_iter:   int   = 200,
) -> TemperatureScaler:
    """Fit temperature scaling on a validation set.

    Args:
        model:      Trained classifier.
        val_loader: Validation DataLoader.
        device:     Torch device.
        init_temp:  Initial temperature value. Default: 1.5.
        lr:         LBFGS learning rate. Default: 0.01.
        max_iter:   LBFGS max iterations. Default: 200.

    Returns:
        Fitted TemperatureScaler wrapping the input model.
    """
    scaler = TemperatureScaler(model)
    scaler.temperature = nn.Parameter(torch.tensor(init_temp))
    return scaler.calibrate(val_loader, device)
