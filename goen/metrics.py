"""
GOEN: Geometry-Optimised Epistemic Network
==========================================

A principled, geometry-aware approach to epistemic uncertainty estimation
and out-of-distribution (OOD) detection.

Typical usage::

    from goen import GOEN, Trainer, get_default_config, load_pretrained

    # Train from scratch
    cfg     = get_default_config()
    trainer = Trainer(cfg)
    model   = trainer.train()
    results = trainer.evaluate(model)

    # Load pretrained
    model = load_pretrained("checkpoints/goen_seed42.pt")
    out   = model.predict(images)   # {"probs", "pred", "uncertainty", "features"}
"""

from .model   import GOEN, ResNet18MS, CenterLoss, CalibHead
from .trainer import Trainer, train_phase1, fit_mahalanobis, train_phase3
from .utils   import (
    get_default_config,
    load_config,
    load_pretrained,
    save_checkpoint,
    set_seed,
    Logger,
    ResultsBook,
)
from .metrics import (
    compute_ece,
    compute_nll,
    compute_brier,
    compute_ood_metrics,
    compute_selective_auc,
    predictive_entropy,
    mutual_information,
    ensemble_variance,
    edl_vacuity,
)

__version__ = "1.0.0"
__all__ = [
    # Model
    "GOEN",
    "ResNet18MS",
    "CenterLoss",
    "CalibHead",
    # Training
    "Trainer",
    "train_phase1",
    "fit_mahalanobis",
    "train_phase3",
    # Utils
    "get_default_config",
    "load_config",
    "load_pretrained",
    "save_checkpoint",
    "set_seed",
    "Logger",
    "ResultsBook",
    # Metrics
    "compute_ece",
    "compute_nll",
    "compute_brier",
    "compute_ood_metrics",
    "compute_selective_auc",
    "predictive_entropy",
    "mutual_information",
    "ensemble_variance",
    "edl_vacuity",
]
