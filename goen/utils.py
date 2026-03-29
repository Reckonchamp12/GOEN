"""
goen/metrics.py
===============
Evaluation metrics for uncertainty quantification and OOD detection.

Functions
---------
compute_ece         Expected Calibration Error (Naeini et al. 2015)
compute_nll         Mean negative log-likelihood
compute_brier       Brier score (multi-class)
compute_ood_metrics AUROC, AUPR, FPR@TPR95, Detection Accuracy
compute_selective_auc  Area under accuracy-coverage (selective prediction) curve
predictive_entropy  H[p(y|x)]
mutual_information  I[y; ω | x] for sample-based methods (MC Dropout, Ensemble)
ensemble_variance   Total predictive variance across ensemble members
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

_EPS = 1e-12


# ─────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────

def compute_ece(
    probs:  np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error.

    Partitions predictions into ``n_bins`` confidence bins and computes
    the weighted average gap between confidence and accuracy.

    Args:
        probs:  Softmax probabilities, shape (N, C).
        labels: Ground-truth labels,   shape (N,).
        n_bins: Number of equal-width bins. Default: 15.

    Returns:
        ECE ∈ [0, 1] — lower is better.
    """
    conf  = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc   = (preds == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum():
            ece += mask.sum() * abs(conf[mask].mean() - acc[mask].mean())
    return float(ece / len(labels))


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood.

    Args:
        probs:  Softmax probabilities, shape (N, C).
        labels: Ground-truth labels,   shape (N,).

    Returns:
        NLL — lower is better.
    """
    return float(-np.log(probs[np.arange(len(labels)), labels] + _EPS).mean())


def compute_brier(
    probs:       np.ndarray,
    labels:      np.ndarray,
    num_classes: int = 10,
) -> float:
    """Multi-class Brier score.

    BS = (1/N) Σ_i Σ_c (p(y=c|x_i) − 1[y_i=c])²

    Args:
        probs:       Softmax probabilities, shape (N, C).
        labels:      Ground-truth labels,   shape (N,).
        num_classes: Number of classes.

    Returns:
        Brier score ∈ [0, 2] — lower is better.
    """
    y_oh = np.eye(num_classes)[labels]
    return float(((probs - y_oh) ** 2).sum(axis=1).mean())


# ─────────────────────────────────────────────────────────────
# OOD Detection
# ─────────────────────────────────────────────────────────────

def compute_ood_metrics(
    id_scores:  np.ndarray,
    ood_scores: np.ndarray,
) -> dict[str, float]:
    """Compute standard OOD detection metrics.

    Convention: higher score → more likely OOD (positive class).

    Metrics
    -------
    AUROC   Area under the ROC curve.
    AUPR    Area under the precision-recall curve (OOD as positive).
    FPR95   False positive rate at 95 % true positive rate.
    DetAcc  Detection accuracy at the Youden-optimal threshold.

    Args:
        id_scores:  Uncertainty scores for in-distribution samples, shape (N_id,).
        ood_scores: Uncertainty scores for OOD samples,             shape (N_ood,).

    Returns:
        dict with keys AUROC, AUPR, FPR95, DetAcc.
    """
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])

    auroc = float(roc_auc_score(y, s))
    aupr  = float(average_precision_score(y, s))

    fpr, tpr, thr = roc_curve(y, s)
    idx95   = np.searchsorted(tpr, 0.95, side="left")
    fpr95   = float(fpr[min(idx95, len(fpr) - 1)])

    best_i  = int(np.argmax(tpr - fpr))
    preds   = (s >= thr[best_i]).astype(int)
    det_acc = float((preds == y).mean())

    return dict(AUROC=auroc, AUPR=aupr, FPR95=fpr95, DetAcc=det_acc)


# ─────────────────────────────────────────────────────────────
# Selective Prediction
# ─────────────────────────────────────────────────────────────

def compute_selective_auc(
    uncertainty: np.ndarray,
    labels:      np.ndarray,
    probs:       np.ndarray,
    n_steps:     int = 100,
) -> float:
    """Area under the accuracy-coverage curve (selective prediction).

    At each coverage level (fraction of data retained), the model abstains
    on the most uncertain samples. AUC summarises the accuracy-coverage
    tradeoff — higher is better.

    Args:
        uncertainty: Per-sample uncertainty scores, shape (N,).
        labels:      Ground-truth labels,            shape (N,).
        probs:       Softmax probabilities,           shape (N, C).
        n_steps:     Number of coverage steps.       Default: 100.

    Returns:
        Selective AUC ∈ [0, 1] — higher is better.
    """
    order   = np.argsort(uncertainty)              # ascending: most certain first
    preds   = probs.argmax(axis=1)
    correct = (preds == labels)[order]
    ns      = np.linspace(1, len(order), n_steps, dtype=int)
    accs    = [correct[:n].mean() for n in ns]
    covs    = ns / len(order)
    return float(np.trapz(accs, covs))


# ─────────────────────────────────────────────────────────────
# Sample-Based Uncertainty Decomposition
# ─────────────────────────────────────────────────────────────

def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """H[p(y|x)] = −Σ_c p_c log p_c.

    Works on (N, C) or (S, N, C) — computed over the last axis.

    Args:
        probs: Softmax probabilities.

    Returns:
        Entropy, same shape minus the last axis.
    """
    return -(probs * np.log(probs + _EPS)).sum(axis=-1)


def mutual_information(probs_samples: np.ndarray) -> np.ndarray:
    """Mutual information I[y; ω | x] = H[E_ω p] − E_ω H[p].

    Epistemic uncertainty estimate for sample-based methods (MC Dropout,
    Deep Ensemble, EpiNet). Higher → more epistemic uncertainty.

    Args:
        probs_samples: (S, N, C) — S stochastic samples.

    Returns:
        mi: (N,) — epistemic uncertainty per sample.
    """
    mean_p  = probs_samples.mean(axis=0)                          # (N, C)
    total_H = predictive_entropy(mean_p)                           # (N,)
    avg_H   = predictive_entropy(probs_samples).mean(axis=0)       # (N,)
    return np.maximum(total_H - avg_H, 0.0)


def ensemble_variance(probs_samples: np.ndarray) -> np.ndarray:
    """Total predictive variance Σ_c Var_ω[p_c(x)].

    Args:
        probs_samples: (S, N, C) — S ensemble member predictions.

    Returns:
        var: (N,) — summed class-wise variance.
    """
    return probs_samples.var(axis=0).sum(axis=-1)


def edl_vacuity(alpha: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """EDL vacuity u(x) = K / S  (Sensoy et al. 2018).

    Args:
        alpha:       Dirichlet parameters, shape (N, C).
        num_classes: Number of classes K. Default: 10.

    Returns:
        u: (N,) — vacuity (epistemic uncertainty). Higher → more OOD.
    """
    S = alpha.sum(axis=1)
    return num_classes / S
