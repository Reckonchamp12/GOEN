"""
baselines/mahalanobis.py
========================
Mahalanobis Distance OOD Detector — B3.

Reference: Lee et al., "A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks", NeurIPS 2018.

Fits class-conditional Gaussian models with a shared covariance matrix
on training penultimate-layer features. OOD score = minimum Mahalanobis
distance across all class Gaussians.

Note: GOEN's Phase 2 extends this baseline by:
  - Using multi-scale features (Layer2 + Layer4 concat) instead of Layer4 only.
  - Applying L2-normalisation to mitigate feature collapse.
  - Achieving a feature compactness ratio of 7–12× vs 0.43 for raw CE features.
"""

from __future__ import annotations

from goen.detectors import MahalanobisDetector   # re-export canonical implementation

__all__ = ["MahalanobisDetector"]
