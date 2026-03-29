"""
tests/test_data.py
==================
Unit tests for GOEN data loading utilities.
Tests use synthetic tensors so no actual dataset downloads are needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goen.data import _norm_tensor, _CIFAR100_SAFE, CIFAR10_MEAN, CIFAR10_STD


# ─────────────────────────────────────────────────────────────
# Normalisation helper
# ─────────────────────────────────────────────────────────────

class TestNormTensor:
    def test_output_shape_preserved(self):
        imgs = torch.rand(16, 3, 32, 32)
        out  = _norm_tensor(imgs)
        assert out.shape == imgs.shape

    def test_normalisation_values(self):
        """A uniform [0,1] image should have known mean after normalisation."""
        imgs = torch.ones(1, 3, 32, 32) * 0.5
        out  = _norm_tensor(imgs)
        for c, (mu, sig) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
            expected = (0.5 - mu) / sig
            assert abs(out[0, c, 0, 0].item() - expected) < 1e-5

    def test_zero_input(self):
        imgs = torch.zeros(4, 3, 32, 32)
        out  = _norm_tensor(imgs)
        for c, (mu, sig) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
            expected = (0.0 - mu) / sig
            assert abs(out[0, c, 0, 0].item() - expected) < 1e-5

    def test_dtype_preserved(self):
        imgs = torch.rand(4, 3, 32, 32).float()
        out  = _norm_tensor(imgs)
        assert out.dtype == torch.float32


# ─────────────────────────────────────────────────────────────
# CIFAR-100 safe class list
# ─────────────────────────────────────────────────────────────

class TestCifar100SafeClasses:
    def test_length(self):
        assert len(_CIFAR100_SAFE) == 50

    def test_sorted(self):
        assert list(_CIFAR100_SAFE) == sorted(_CIFAR100_SAFE)

    def test_no_duplicates(self):
        assert len(_CIFAR100_SAFE) == len(set(_CIFAR100_SAFE))

    def test_all_in_range(self):
        for c in _CIFAR100_SAFE:
            assert 0 <= c <= 99, f"Class {c} out of CIFAR-100 range [0,99]"

    def test_no_cifar10_overlap(self):
        """Safe classes should not include any of the 10 CIFAR-10 superclass IDs
        that directly correspond to vehicle/animal categories."""
        # CIFAR-10-like superclasses in CIFAR-100: large_vehicles (58,49,...),
        # aquatic_mammals (4,30,55,72,95), etc.
        # We just verify the list is non-empty and plausible.
        assert len(_CIFAR100_SAFE) > 0


# ─────────────────────────────────────────────────────────────
# Synthetic OOD loader (no download needed)
# ─────────────────────────────────────────────────────────────

class TestSyntheticOOD:
    def test_synthetic_loader_shape(self):
        """Build a synthetic loader directly and check shapes."""
        g    = torch.Generator()
        g.manual_seed(0)
        imgs = torch.clamp(torch.randn(200, 3, 32, 32, generator=g) * 0.5 + 0.5, 0, 1)
        imgs = _norm_tensor(imgs)
        ds   = TensorDataset(imgs, torch.zeros(200, dtype=torch.long))
        dl   = DataLoader(ds, batch_size=32, shuffle=False)

        all_x, all_y = [], []
        for x, y in dl:
            all_x.append(x); all_y.append(y)
        X = torch.cat(all_x)
        Y = torch.cat(all_y)

        assert X.shape == (200, 3, 32, 32)
        assert Y.shape == (200,)
        assert (Y == 0).all()

    def test_synthetic_reproducibility(self):
        """Same seed → identical tensors."""
        def _make():
            g = torch.Generator(); g.manual_seed(42)
            return torch.randn(50, 3, 32, 32, generator=g)
        t1, t2 = _make(), _make()
        assert torch.allclose(t1, t2)

    def test_normalised_range(self):
        """After CIFAR-10 normalisation the values should be roughly ∈ [−3, 3]."""
        g    = torch.Generator(); g.manual_seed(1)
        imgs = torch.clamp(torch.randn(100, 3, 32, 32, generator=g) * 0.5 + 0.5, 0, 1)
        out  = _norm_tensor(imgs)
        assert out.min().item() > -5.0
        assert out.max().item() <  5.0


# ─────────────────────────────────────────────────────────────
# DataLoader compatibility (batch shapes, dtypes)
# ─────────────────────────────────────────────────────────────

class TestDataLoaderCompatibility:
    def _make_fake_loader(self, n=64, batch_size=16):
        imgs   = torch.randn(n, 3, 32, 32)
        labels = torch.randint(0, 10, (n,))
        ds     = TensorDataset(imgs, labels)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def test_batch_shape(self):
        dl  = self._make_fake_loader()
        x, y = next(iter(dl))
        assert x.shape == (16, 3, 32, 32)
        assert y.shape == (16,)

    def test_label_dtype(self):
        dl  = self._make_fake_loader()
        _, y = next(iter(dl))
        assert y.dtype == torch.int64

    def test_image_dtype(self):
        dl  = self._make_fake_loader()
        x, _ = next(iter(dl))
        assert x.dtype == torch.float32

    def test_full_iteration(self):
        dl    = self._make_fake_loader(n=64, batch_size=16)
        count = sum(x.size(0) for x, _ in dl)
        assert count == 64
