"""
goen/model.py
=============
Core GOEN model components:
  - ResNet18MS  : multi-scale ResNet-18 backbone (concat Layer2 + Layer4)
  - CenterLoss  : intra-class compactness loss (Wen et al. 2016)
  - CalibHead   : learned uncertainty calibration head
  - GOEN        : full model combining all components
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# BasicBlock
# ─────────────────────────────────────────────────────────────

class _BasicBlock(nn.Module):
    """Standard pre-activation residual block."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


# ─────────────────────────────────────────────────────────────
# Multi-Scale ResNet-18 Backbone
# ─────────────────────────────────────────────────────────────

class ResNet18MS(nn.Module):
    """ResNet-18 with multi-scale feature extraction for 32×32 inputs.

    Extracts features at two semantic levels:
      - Layer2 output (128-dim): texture and colour — best for domain-shift OOD
      - Layer4 output (512-dim): semantic features  — best for class-level OOD

    Concatenation (640-dim) is projected to ``proj_dim`` via a linear + BN + ReLU.
    This is motivated by the empirical finding that Layer2 AUROC (0.683) > Layer4
    AUROC (0.638) on domain-shifted (SVHN) data despite Layer4 having higher
    semantic discriminability.

    Args:
        proj_dim:     Output feature dimension after projection. Default: 512.
        num_classes:  Number of classification heads. Default: 10.
        single_scale: If True, use only Layer4 (ablation). Default: False.
    """

    L2_DIM: int = 128
    L4_DIM: int = 512

    def __init__(
        self,
        proj_dim:     int  = 512,
        num_classes:  int  = 10,
        single_scale: bool = False,
    ) -> None:
        super().__init__()
        self.in_planes    = 64
        self.single_scale = single_scale
        self.proj_dim     = proj_dim
        in_dim            = self.L4_DIM if single_scale else (self.L2_DIM + self.L4_DIM)

        self.conv1  = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  n=2, stride=1)
        self.layer2 = self._make_layer(128, n=2, stride=2)  # ← tap point 1
        self.layer3 = self._make_layer(256, n=2, stride=2)
        self.layer4 = self._make_layer(512, n=2, stride=2)  # ← tap point 2

        # Multi-scale projection: [128 + 512] → proj_dim
        self.ms_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(proj_dim, num_classes)

    def _make_layer(self, planes: int, n: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (n - 1)
        layers  = []
        for s in strides:
            layers.append(_BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale projected features z ∈ R^{proj_dim}."""
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        f2  = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)   # (B, 128)
        out = self.layer3(out)
        out = self.layer4(out)
        f4  = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)   # (B, 512)
        ms  = f4 if self.single_scale else torch.cat([f2, f4], dim=1)
        return self.ms_proj(ms)                                       # (B, proj_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, features)."""
        z = self.get_features(x)
        return self.fc(z), z


# ─────────────────────────────────────────────────────────────
# Center Loss
# ─────────────────────────────────────────────────────────────

class CenterLoss(nn.Module):
    """Intra-class compactness loss (Wen et al., ECCV 2016).

    L_center = (1/N) Σ_i ||z_i − c_{y_i}||²

    The class centers c are learned jointly with the backbone but updated
    with a separate, typically smaller, learning rate (0.5 by default).

    Args:
        num_classes: Number of class centers C.
        feat_dim:    Feature dimensionality D.
    """

    def __init__(self, num_classes: int, feat_dim: int) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute center loss.

        Args:
            z:      Feature vectors, shape (B, D).
            labels: Class labels,   shape (B,).

        Returns:
            Scalar loss.
        """
        return ((z - self.centers[labels]) ** 2).sum(dim=1).mean()


# ─────────────────────────────────────────────────────────────
# Calibration Head
# ─────────────────────────────────────────────────────────────

class CalibHead(nn.Module):
    """Lightweight MLP that maps three geometry-aware signals to u(x) ∈ (0,1).

    Input features (3-dimensional):
        1. log_maha(x)   — log of minimum Mahalanobis distance to any class mean.
                           High → far from all class manifolds → OOD.
        2. max_cos(x)    — maximum cosine similarity between z and class means
                           (on the unit sphere). Low → far from all classes → OOD.
        3. H[p(y|x)]     — predictive entropy of the softmax distribution.
                           High → uncertain prediction → OOD.

    Architecture:  Linear(3→64) → ReLU → Linear(64→32) → ReLU → Linear(32→1) → Sigmoid

    Trained with binary cross-entropy:
        BCE(u_ID, 0.05) + BCE(u_OOD, 0.95)
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map geometry features to scalar uncertainty.

        Args:
            x: (B, 3) tensor of [log_maha, max_cos, entropy].

        Returns:
            u: (B,) uncertainty scores in (0, 1).
        """
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# GOEN: Full Model
# ─────────────────────────────────────────────────────────────

class GOEN(nn.Module):
    """Geometry-Optimised Epistemic Network.

    Three-phase training:
      Phase 1: Train backbone with CE + CenterLoss.
      Phase 2: Fit class-conditional Gaussians on L2-normalised features.
      Phase 3: Train CalibHead only (backbone frozen) using BCE on ID/OOD pairs.

    At inference, the model returns (logits, u, z) where u ∈ (0,1) is the
    calibrated uncertainty score (higher = more likely OOD / more uncertain).

    Args:
        num_classes:  Number of output classes. Default: 10.
        proj_dim:     Projected feature dimension. Default: 512.
        single_scale: If True, use only Layer4 features (ablation). Default: False.
    """

    def __init__(
        self,
        num_classes:  int  = 10,
        proj_dim:     int  = 512,
        single_scale: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone    = ResNet18MS(proj_dim, num_classes, single_scale)
        self.calib       = CalibHead()

        # Mahalanobis parameters — set in Phase 2, NOT trained by gradient
        self.register_buffer("class_means", torch.zeros(num_classes, proj_dim))
        self.register_buffer("precision",   torch.eye(proj_dim))

    # ── Mahalanobis score ────────────────────────────────────

    def maha_score(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Minimum class-conditional Mahalanobis distance (squared).

        Operates on L2-normalised features to address feature collapse
        (top-1 PC explaining 54.9% of variance in CE-trained features).

        Args:
            z_norm: L2-normalised features, shape (B, D).

        Returns:
            scores: (B,) — higher = farther from all class manifolds = more OOD.
        """
        dists = torch.stack([
            ((z_norm - self.class_means[c]) @ self.precision
             * (z_norm - self.class_means[c])).sum(dim=1)
            for c in range(self.num_classes)
        ], dim=1)                                  # (B, C)
        return dists.min(dim=1).values             # (B,)

    # ── Uncertainty score ────────────────────────────────────

    def uncertainty(
        self,
        z:      torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute calibrated uncertainty score u(x) ∈ (0,1).

        Args:
            z:      Raw (un-normalised) features, shape (B, D).
            logits: Classification logits,        shape (B, C).

        Returns:
            u: (B,) uncertainty scores.
        """
        z_n   = F.normalize(z, dim=1)                                 # (B, D)
        maha  = torch.log(self.maha_score(z_n).clamp(min=1e-4))       # (B,)

        mu_n  = F.normalize(self.class_means, dim=1)                  # (C, D)
        max_c = (z_n @ mu_n.T).max(dim=1).values                      # (B,)

        probs = F.softmax(logits, dim=1)                               # (B, C)
        ent   = -(probs * torch.log(probs + 1e-12)).sum(dim=1)         # (B,)

        feat  = torch.stack([maha, max_c, ent], dim=1)                # (B, 3)
        return self.calib(feat)                                        # (B,)

    # ── Forward ──────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            x: Input images, shape (B, 3, H, W).

        Returns:
            logits: Classification logits, (B, num_classes).
            u:      Uncertainty scores,    (B,)  — 0=certain, 1=uncertain.
            z:      Raw features,          (B, proj_dim).
        """
        logits, z = self.backbone(x)
        u         = self.uncertainty(z, logits)
        return logits, u, z

    # ── Convenience ──────────────────────────────────────────

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convenience wrapper returning a dict with all useful outputs.

        Returns:
            {
              "probs":      (B, C)  softmax class probabilities
              "pred":       (B,)    argmax predicted class
              "uncertainty":(B,)    calibrated uncertainty ∈ (0,1)
              "features":   (B, D)  raw feature vectors
            }
        """
        self.eval()
        logits, u, z = self(x)
        return {
            "probs":       F.softmax(logits, dim=1),
            "pred":        logits.argmax(dim=1),
            "uncertainty": u,
            "features":    z,
        }

    @property
    def feat_dim(self) -> int:
        return self.backbone.proj_dim
