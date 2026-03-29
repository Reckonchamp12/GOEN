"""
goen/detectors.py
=================
Standalone post-hoc OOD detectors that operate on pre-extracted features.
These are used for baseline comparison (B3, B4) but can also be used
independently of the full GOEN pipeline.

Classes
-------
MahalanobisDetector   Class-conditional Mahalanobis distance (Lee et al. 2018)
KNNDetector           k-NN cosine distance (Sun et al. 2022)
"""

from __future__ import annotations

import numpy as np


_EPS = 1e-12


# ─────────────────────────────────────────────────────────────
# Mahalanobis Distance Detector
# ─────────────────────────────────────────────────────────────

class MahalanobisDetector:
    """Class-conditional Mahalanobis distance OOD detector.

    Reference: Lee et al., "A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks", NeurIPS 2018.

    Fits class-conditional Gaussians with a shared (tied) covariance matrix
    on training features, then scores test samples by their minimum
    Mahalanobis distance to any class mean.

    Higher score → more OOD.

    Example::

        det = MahalanobisDetector()
        det.fit(train_features, train_labels)

        id_scores  = det.score(test_features)
        ood_scores = det.score(ood_features)
    """

    def __init__(self, num_classes: int = 10) -> None:
        self.num_classes   = num_classes
        self.class_means:  np.ndarray | None = None
        self.precision:    np.ndarray | None = None

    def fit(
        self,
        features: np.ndarray,
        labels:   np.ndarray,
        reg:      float = 1e-6,
    ) -> "MahalanobisDetector":
        """Fit class means and tied precision matrix.

        Args:
            features: Training features, shape (N, D).
            labels:   Training labels,   shape (N,).
            reg:      Regularisation added to diagonal of covariance. Default: 1e-6.

        Returns:
            self
        """
        N, D = features.shape
        C    = self.num_classes

        self.class_means = np.stack([
            features[labels == c].mean(0) for c in range(C)
        ])                                                # (C, D)

        cov = np.zeros((D, D))
        for c in range(C):
            diff = features[labels == c] - self.class_means[c]
            cov  += diff.T @ diff
        cov          = cov / N + reg * np.eye(D)
        self.precision = np.linalg.inv(cov)               # (D, D)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Compute minimum Mahalanobis distance score for each sample.

        Args:
            features: Test features, shape (N, D).

        Returns:
            scores: (N,) — higher = more likely OOD.
        """
        if self.class_means is None or self.precision is None:
            raise RuntimeError("Call fit() before score().")

        dists = np.stack([
            np.einsum(
                "nd,dd,nd->n",
                features - self.class_means[c],
                self.precision,
                features - self.class_means[c],
            )
            for c in range(self.num_classes)
        ], axis=1)                                        # (N, C)
        return dists.min(axis=1)                          # (N,)


# ─────────────────────────────────────────────────────────────
# KNN Distance Detector
# ─────────────────────────────────────────────────────────────

class KNNDetector:
    """k-nearest-neighbour cosine distance OOD detector.

    Reference: Sun et al., "Out-of-Distribution Detection with Deep
    Nearest Neighbors", ICML 2022.

    Fits on L2-normalised training features, then scores test samples
    by their cosine distance to the k-th nearest training neighbour.

    Higher score → more OOD.

    Example::

        det = KNNDetector(k=5)
        det.fit(train_features)

        id_scores  = det.score(test_features)
        ood_scores = det.score(ood_features)
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._train_norm: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "KNNDetector":
        """L2-normalise and store training features.

        Args:
            features: Training features, shape (N_train, D).

        Returns:
            self
        """
        norms = np.linalg.norm(features, axis=1, keepdims=True) + _EPS
        self._train_norm = features / norms
        return self

    def score(
        self,
        features: np.ndarray,
        chunk:    int = 512,
    ) -> np.ndarray:
        """Compute k-NN cosine distance score for each sample.

        Processes queries in chunks to avoid OOM on large feature sets.

        Args:
            features: Test features, shape (N_test, D).
            chunk:    Number of queries per chunk. Default: 512.

        Returns:
            scores: (N_test,) — cosine distance to k-th nearest neighbour.
                    Higher = more OOD.
        """
        if self._train_norm is None:
            raise RuntimeError("Call fit() before score().")

        norms = np.linalg.norm(features, axis=1, keepdims=True) + _EPS
        q     = features / norms

        out = []
        for i in range(0, len(q), chunk):
            sim  = q[i: i + chunk] @ self._train_norm.T   # (B, N_train)
            dist = 1.0 - sim                               # cosine distance
            kd   = np.sort(dist, axis=1)[:, self.k - 1]   # k-th distance
            out.append(kd)

        return np.concatenate(out)
