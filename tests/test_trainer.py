"""
tests/test_model.py
===================
Unit tests for GOEN model components.
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goen.model import GOEN, ResNet18MS, CenterLoss, CalibHead


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_batch():
    torch.manual_seed(0)
    return torch.randn(8, 3, 32, 32)

@pytest.fixture
def dummy_labels():
    return torch.randint(0, 10, (8,))

@pytest.fixture
def model():
    m = GOEN(num_classes=10, proj_dim=512)
    m.eval()
    return m


# ─────────────────────────────────────────────────────────────
# ResNet18MS tests
# ─────────────────────────────────────────────────────────────

class TestResNet18MS:
    def test_output_shapes_multiscale(self, dummy_batch):
        net = ResNet18MS(proj_dim=512, single_scale=False)
        logits, z = net(dummy_batch)
        assert logits.shape == (8, 10), f"Expected (8,10), got {logits.shape}"
        assert z.shape     == (8, 512), f"Expected (8,512), got {z.shape}"

    def test_output_shapes_singlescale(self, dummy_batch):
        net = ResNet18MS(proj_dim=512, single_scale=True)
        logits, z = net(dummy_batch)
        assert logits.shape == (8, 10)
        assert z.shape      == (8, 512)

    def test_get_features(self, dummy_batch):
        net = ResNet18MS(proj_dim=256)
        z   = net.get_features(dummy_batch)
        assert z.shape == (8, 256)

    def test_proj_dim_variable(self, dummy_batch):
        for dim in [128, 256, 512]:
            net = ResNet18MS(proj_dim=dim)
            _, z = net(dummy_batch)
            assert z.shape == (8, dim)


# ─────────────────────────────────────────────────────────────
# CenterLoss tests
# ─────────────────────────────────────────────────────────────

class TestCenterLoss:
    def test_forward_scalar(self, dummy_labels):
        clf = CenterLoss(num_classes=10, feat_dim=512)
        z   = torch.randn(8, 512)
        loss = clf(z, dummy_labels)
        assert loss.ndim == 0, "CenterLoss should return a scalar"
        assert loss.item() >= 0.0

    def test_zero_when_features_at_centers(self):
        clf = CenterLoss(num_classes=3, feat_dim=4)
        # Force features to exactly match centers
        labels = torch.tensor([0, 1, 2])
        z      = clf.centers[labels].detach().clone()
        loss   = clf(z, labels)
        assert abs(loss.item()) < 1e-6

    def test_gradient_flows(self, dummy_labels):
        clf  = CenterLoss(num_classes=10, feat_dim=32)
        z    = torch.randn(8, 32, requires_grad=True)
        loss = clf(z, dummy_labels)
        loss.backward()
        assert z.grad is not None


# ─────────────────────────────────────────────────────────────
# CalibHead tests
# ─────────────────────────────────────────────────────────────

class TestCalibHead:
    def test_output_shape(self):
        head = CalibHead()
        x    = torch.randn(16, 3)
        u    = head(x)
        assert u.shape == (16,), f"Expected (16,), got {u.shape}"

    def test_output_range(self):
        head = CalibHead()
        x    = torch.randn(1000, 3) * 10   # large inputs
        u    = head(x)
        assert u.min().item() > 0.0
        assert u.max().item() < 1.0

    def test_gradient_flows(self):
        head = CalibHead()
        x    = torch.randn(8, 3, requires_grad=True)
        u    = head(x)
        u.sum().backward()
        assert x.grad is not None


# ─────────────────────────────────────────────────────────────
# GOEN tests
# ─────────────────────────────────────────────────────────────

class TestGOEN:
    def test_forward_shapes(self, model, dummy_batch):
        with torch.no_grad():
            logits, u, z = model(dummy_batch)
        assert logits.shape == (8, 10)
        assert u.shape      == (8,)
        assert z.shape      == (8, 512)

    def test_uncertainty_range(self, model, dummy_batch):
        with torch.no_grad():
            _, u, _ = model(dummy_batch)
        assert u.min().item() > 0.0
        assert u.max().item() < 1.0

    def test_maha_score(self, model, dummy_batch):
        with torch.no_grad():
            _, z        = model.backbone(dummy_batch)
            z_n         = torch.nn.functional.normalize(z, dim=1)
            maha        = model.maha_score(z_n)
        assert maha.shape  == (8,)
        assert (maha >= 0).all()

    def test_predict_method(self, model, dummy_batch):
        with torch.no_grad():
            out = model.predict(dummy_batch)
        assert set(out.keys()) == {"probs", "pred", "uncertainty", "features"}
        assert out["probs"].shape       == (8, 10)
        assert out["pred"].shape        == (8,)
        assert out["uncertainty"].shape == (8,)
        assert out["features"].shape    == (8, 512)
        # Probabilities sum to 1
        assert torch.allclose(out["probs"].sum(dim=1),
                               torch.ones(8), atol=1e-5)

    def test_single_scale_ablation(self, dummy_batch):
        m = GOEN(num_classes=10, proj_dim=512, single_scale=True)
        m.eval()
        with torch.no_grad():
            logits, u, z = m(dummy_batch)
        assert logits.shape == (8, 10)
        assert u.shape      == (8,)
        assert z.shape      == (8, 512)

    def test_state_dict_roundtrip(self, model, dummy_batch, tmp_path):
        """Verify model can be saved and loaded with identical outputs."""
        ckpt = tmp_path / "test.pt"
        torch.save(model.state_dict(), ckpt)

        model2 = GOEN(num_classes=10, proj_dim=512)
        model2.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model2.eval()

        with torch.no_grad():
            logits1, u1, z1 = model(dummy_batch)
            logits2, u2, z2 = model2(dummy_batch)

        assert torch.allclose(logits1, logits2, atol=1e-6)
        assert torch.allclose(u1,      u2,      atol=1e-6)
        assert torch.allclose(z1,      z2,      atol=1e-6)

    def test_mahalanobis_buffers_registered(self, model):
        """class_means and precision should be registered buffers."""
        buffers = dict(model.named_buffers())
        assert "class_means" in buffers
        assert "precision"   in buffers
        assert buffers["class_means"].shape == (10, 512)
        assert buffers["precision"].shape   == (512, 512)
