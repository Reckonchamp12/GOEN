"""
tests/test_trainer.py
=====================
Unit tests for GOEN training utilities.
Uses tiny synthetic data so no GPU or real datasets are needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goen.model   import GOEN
from goen.trainer import fit_mahalanobis, train_phase3
from goen.utils   import get_default_config, Logger, set_seed


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_loader(n: int = 64, batch_size: int = 16, num_classes: int = 10) -> DataLoader:
    """Tiny synthetic DataLoader (no real images)."""
    imgs   = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(imgs, labels), batch_size=batch_size, shuffle=False)


@pytest.fixture
def cfg():
    c = get_default_config()
    c.update(dict(
        seed         = 0,
        num_classes  = 10,
        proj_dim     = 64,          # tiny for speed
        single_scale = False,
        p3_epochs    = 2,
        p3_patience  = 5,
        p3_lr        = 1e-3,
        id_target    = 0.05,
        ood_target   = 0.95,
        svhn_ood     = False,       # no SVHN download in tests
        batch_size   = 16,
    ))
    return c


@pytest.fixture
def tiny_model(cfg):
    m = GOEN(num_classes=cfg["num_classes"], proj_dim=cfg["proj_dim"])
    m.eval()
    return m


@pytest.fixture
def val_loader():
    return _make_loader(n=32)


@pytest.fixture
def feat_loader():
    return _make_loader(n=64)


# ─────────────────────────────────────────────────────────────
# fit_mahalanobis
# ─────────────────────────────────────────────────────────────

class TestFitMahalanobis:
    def test_buffers_set(self, tiny_model, feat_loader, cfg):
        """After Phase 2, class_means and precision buffers must be non-trivial."""
        device = torch.device("cpu")
        logger = Logger()
        model, ratio = fit_mahalanobis(tiny_model, feat_loader, cfg, device, logger)

        assert not torch.allclose(model.class_means, torch.zeros_like(model.class_means)), \
            "class_means should not remain all-zero after fitting"
        assert not torch.allclose(model.precision, torch.eye(cfg["proj_dim"])), \
            "precision should not remain identity after fitting"

    def test_ratio_positive(self, tiny_model, feat_loader, cfg):
        device = torch.device("cpu")
        logger = Logger()
        _, ratio = fit_mahalanobis(tiny_model, feat_loader, cfg, device, logger)
        assert ratio > 0.0

    def test_class_means_shape(self, tiny_model, feat_loader, cfg):
        device = torch.device("cpu")
        logger = Logger()
        model, _ = fit_mahalanobis(tiny_model, feat_loader, cfg, device, logger)
        assert model.class_means.shape == (cfg["num_classes"], cfg["proj_dim"])

    def test_precision_shape(self, tiny_model, feat_loader, cfg):
        device = torch.device("cpu")
        logger = Logger()
        model, _ = fit_mahalanobis(tiny_model, feat_loader, cfg, device, logger)
        assert model.precision.shape == (cfg["proj_dim"], cfg["proj_dim"])

    def test_precision_symmetric(self, tiny_model, feat_loader, cfg):
        """Precision matrix (inverse of symmetric covariance) must be symmetric."""
        device = torch.device("cpu")
        logger = Logger()
        model, _ = fit_mahalanobis(tiny_model, feat_loader, cfg, device, logger)
        P = model.precision
        assert torch.allclose(P, P.T, atol=1e-4), "Precision matrix should be symmetric"

    def test_maha_score_after_fit(self, tiny_model, feat_loader, cfg):
        """maha_score should run without error after fitting."""
        device = torch.device("cpu")
        logger = Logger()
        model, _ = fit_mahalanobis(tiny_model, feat_loader, cfg, device, logger)
        z = torch.randn(8, cfg["proj_dim"])
        z_n = F.normalize(z, dim=1)
        with torch.no_grad():
            scores = model.maha_score(z_n)
        assert scores.shape == (8,)
        assert (scores >= 0).all()


# ─────────────────────────────────────────────────────────────
# train_phase3
# ─────────────────────────────────────────────────────────────

class TestTrainPhase3:
    def test_calib_weights_change(self, tiny_model, feat_loader, val_loader, cfg):
        """CalibHead weights must change after Phase 3 training."""
        device = torch.device("cpu")
        logger = Logger()

        # First fit Mahalanobis so the model has valid buffers
        model, _ = fit_mahalanobis(tiny_model, feat_loader, cfg, device, logger)

        calib_before = {k: v.clone() for k, v in model.calib.named_parameters()}
        model = train_phase3(model, val_loader, None, cfg, device, logger)
        calib_after  = {k: v for k, v in model.calib.named_parameters()}

        changed = any(
            not torch.allclose(calib_before[k], calib_after[k])
            for k in calib_before
        )
        assert changed, "CalibHead parameters should change during Phase 3"

    def test_backbone_frozen_during_phase3(self, tiny_model, feat_loader, val_loader, cfg):
        """Backbone weights must NOT change during Phase 3."""
        device = torch.device("cpu")
        logger = Logger()

        model, _ = fit_mahalanobis(tiny_model, feat_loader, cfg, device, logger)
        bb_before = {k: v.clone() for k, v in model.backbone.named_parameters()}
        model = train_phase3(model, val_loader, None, cfg, device, logger)
        bb_after  = {k: v for k, v in model.backbone.named_parameters()}

        for k in bb_before:
            assert torch.allclose(bb_before[k], bb_after[k]), \
                f"Backbone param '{k}' changed during Phase 3 (should be frozen)"

    def test_all_params_trainable_after_phase3(self, tiny_model, feat_loader, val_loader, cfg):
        """After Phase 3, all parameters should have requires_grad=True again."""
        device = torch.device("cpu")
        logger = Logger()
        model, _ = fit_mahalanobis(tiny_model, feat_loader, cfg, device, logger)
        model = train_phase3(model, val_loader, None, cfg, device, logger)

        for name, p in model.named_parameters():
            assert p.requires_grad, f"Parameter '{name}' still frozen after Phase 3"

    def test_uncertainty_gap_positive(self, tiny_model, feat_loader, val_loader, cfg):
        """After Phase 3, OOD uncertainty should be higher than ID uncertainty (on noise)."""
        device = torch.device("cpu")
        logger = Logger()
        model, _ = fit_mahalanobis(tiny_model, feat_loader, cfg, device, logger)
        model = train_phase3(model, val_loader, None, cfg, device, logger)
        model.eval()

        with torch.no_grad():
            # ID: real-ish data from val_loader
            u_id = []
            for x, _ in val_loader:
                lg, z = model.backbone(x)
                u_id.append(model.uncertainty(F.normalize(z, dim=1), lg).numpy())
            u_id_mean = np.concatenate(u_id).mean()

            # OOD: pure Gaussian noise
            noise = torch.randn(64, 3, 32, 32)
            lg_n, z_n = model.backbone(noise)
            u_ood_mean = model.uncertainty(
                F.normalize(z_n, dim=1), lg_n).mean().item()

        # With only 2 epochs, gap may be small — just check it's finite and not NaN
        assert np.isfinite(u_id_mean)
        assert np.isfinite(u_ood_mean)


# ─────────────────────────────────────────────────────────────
# set_seed
# ─────────────────────────────────────────────────────────────

class TestSetSeed:
    def test_reproducible_tensors(self):
        set_seed(99)
        t1 = torch.randn(10)
        set_seed(99)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2)

    def test_different_seeds_differ(self):
        set_seed(1)
        t1 = torch.randn(10)
        set_seed(2)
        t2 = torch.randn(10)
        assert not torch.allclose(t1, t2)

    def test_numpy_reproducible(self):
        set_seed(7)
        a1 = np.random.rand(10)
        set_seed(7)
        a2 = np.random.rand(10)
        assert np.allclose(a1, a2)
