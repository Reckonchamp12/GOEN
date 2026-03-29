"""
tests/test_detectors.py
=======================
Unit tests for standalone MahalanobisDetector and KNNDetector.
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goen.detectors import MahalanobisDetector, KNNDetector


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def clustered_data():
    """10 well-separated Gaussian clusters (num_classes=10, D=32)."""
    rng = np.random.RandomState(42)
    C, D, n_per = 10, 32, 100
    centers = rng.randn(C, D) * 5.0          # far-apart centres
    feats   = np.vstack([
        centers[c] + rng.randn(n_per, D) * 0.1
        for c in range(C)
    ])
    labels = np.repeat(np.arange(C), n_per)
    return feats, labels, centers


@pytest.fixture
def ood_data():
    """OOD samples that are far from all cluster centres."""
    rng = np.random.RandomState(0)
    return rng.randn(50, 32) * 20.0   # very large magnitude → far from clusters


# ─────────────────────────────────────────────────────────────
# MahalanobisDetector
# ─────────────────────────────────────────────────────────────

class TestMahalanobisDetector:
    def test_fit_returns_self(self, clustered_data):
        feats, labels, _ = clustered_data
        det = MahalanobisDetector(num_classes=10)
        ret = det.fit(feats, labels)
        assert ret is det

    def test_class_means_shape(self, clustered_data):
        feats, labels, _ = clustered_data
        det = MahalanobisDetector(num_classes=10).fit(feats, labels)
        assert det.class_means.shape == (10, 32)

    def test_precision_shape(self, clustered_data):
        feats, labels, _ = clustered_data
        det = MahalanobisDetector(num_classes=10).fit(feats, labels)
        assert det.precision.shape == (32, 32)

    def test_precision_symmetric(self, clustered_data):
        feats, labels, _ = clustered_data
        det = MahalanobisDetector(num_classes=10).fit(feats, labels)
        assert np.allclose(det.precision, det.precision.T, atol=1e-4)

    def test_score_shape(self, clustered_data):
        feats, labels, _ = clustered_data
        det    = MahalanobisDetector(num_classes=10).fit(feats, labels)
        scores = det.score(feats[:20])
        assert scores.shape == (20,)

    def test_scores_nonneg(self, clustered_data):
        feats, labels, _ = clustered_data
        det = MahalanobisDetector(num_classes=10).fit(feats, labels)
        assert (det.score(feats) >= 0).all()

    def test_id_lower_than_ood(self, clustered_data, ood_data):
        """ID samples (near cluster centres) should score lower than OOD."""
        feats, labels, _ = clustered_data
        det    = MahalanobisDetector(num_classes=10).fit(feats, labels)
        id_sc  = det.score(feats).mean()
        ood_sc = det.score(ood_data).mean()
        assert ood_sc > id_sc, \
            f"OOD mean score {ood_sc:.2f} should exceed ID mean score {id_sc:.2f}"

    def test_score_before_fit_raises(self):
        det = MahalanobisDetector()
        with pytest.raises(RuntimeError):
            det.score(np.random.randn(5, 32))

    def test_exact_centre_score_near_zero(self, clustered_data):
        """A sample exactly at a class centre should have near-zero Maha distance."""
        feats, labels, centers = clustered_data
        det   = MahalanobisDetector(num_classes=10).fit(feats, labels)
        # Use the fitted class means (may differ slightly from true centers)
        score = det.score(det.class_means[:1])
        assert score[0] < 1.0, f"Score at class mean should be near 0, got {score[0]:.4f}"


# ─────────────────────────────────────────────────────────────
# KNNDetector
# ─────────────────────────────────────────────────────────────

class TestKNNDetector:
    def test_fit_returns_self(self, clustered_data):
        feats, _, _ = clustered_data
        det = KNNDetector(k=5)
        ret = det.fit(feats)
        assert ret is det

    def test_score_shape(self, clustered_data):
        feats, _, _ = clustered_data
        det    = KNNDetector(k=5).fit(feats)
        scores = det.score(feats[:30])
        assert scores.shape == (30,)

    def test_scores_in_range(self, clustered_data):
        """Cosine distance is in [0, 2]."""
        feats, _, _ = clustered_data
        det    = KNNDetector(k=5).fit(feats)
        scores = det.score(feats)
        assert (scores >= 0).all()
        assert (scores <= 2.0 + 1e-5).all()

    def test_id_lower_than_ood(self, clustered_data, ood_data):
        feats, _, _ = clustered_data
        det    = KNNDetector(k=5).fit(feats)
        id_sc  = det.score(feats).mean()
        ood_sc = det.score(ood_data).mean()
        assert ood_sc > id_sc

    def test_score_before_fit_raises(self):
        det = KNNDetector(k=3)
        with pytest.raises(RuntimeError):
            det.score(np.random.randn(5, 32))

    def test_different_k(self, clustered_data):
        """Different k values should produce different (or equal) scores."""
        feats, _, _ = clustered_data
        s1 = KNNDetector(k=1).fit(feats).score(feats[:20])
        s5 = KNNDetector(k=5).fit(feats).score(feats[:20])
        assert s1.shape == s5.shape   # shapes must match
        # k=1 score ≤ k=5 score (closer neighbour ≤ 5th neighbour)
        assert (s1 <= s5 + 1e-8).all()

    def test_chunked_equals_full(self, clustered_data):
        """Chunked processing should give identical results to full computation."""
        feats, _, _ = clustered_data
        det   = KNNDetector(k=5).fit(feats)
        full  = det.score(feats[:50], chunk=len(feats))
        chunk = det.score(feats[:50], chunk=10)
        assert np.allclose(full, chunk, atol=1e-6)

    def test_l2_normalisation_in_fit(self, clustered_data):
        """Stored training features should be L2-normalised."""
        feats, _, _ = clustered_data
        det = KNNDetector(k=3).fit(feats)
        norms = np.linalg.norm(det._train_norm, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
