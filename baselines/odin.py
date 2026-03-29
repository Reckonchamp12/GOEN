"""
baselines/__init__.py
=====================
Baseline uncertainty and OOD detection methods for benchmarking against GOEN.

All baselines share the same CIFAR-10 ResNet-18 backbone and training split.

Catalogue
---------
  StandardNN          A1 — Softmax maximum probability (MSP)
  MCDropoutNet        A2 — Monte Carlo Dropout
  DeepEnsemble        A3 — Collection of independently trained ResNet-18s
  TemperatureScaler   A4 — Post-hoc temperature scaling on top of A1
  EDLNet              A5 — Evidential Deep Learning (Dirichlet output)
  EpiNet              A6 — Epinet (base + prior + trainable epinet head)
  MoENet              A7 — Mixture of Experts with soft gating
  energy_score        B1 — Energy-based OOD score
  odin_score          B2 — ODIN (input preprocessing + temperature)
"""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────
# Shared backbone
# ─────────────────────────────────────────────────────────────

class _Block(nn.Module):
    def __init__(self, ip, p, s=1):
        super().__init__()
        self.c1 = nn.Conv2d(ip, p, 3, stride=s, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(p)
        self.c2 = nn.Conv2d(p, p, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(p)
        self.sc = nn.Sequential()
        if s != 1 or ip != p:
            self.sc = nn.Sequential(
                nn.Conv2d(ip, p, 1, stride=s, bias=False), nn.BatchNorm2d(p))
    def forward(self, x):
        return F.relu(self.b2(self.c2(
            F.relu(self.b1(self.c1(x)), True))) + self.sc(x), True)


class ResNet18(nn.Module):
    """Vanilla ResNet-18 for 32×32 inputs."""
    FEAT_DIM = 512

    def __init__(self, num_classes=10, dropout_rate=0.0):
        super().__init__()
        self.ip      = 64
        self.conv1   = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.layer1  = self._m(64,  2, 1)
        self.layer2  = self._m(128, 2, 2)
        self.layer3  = self._m(256, 2, 2)
        self.layer4  = self._m(512, 2, 2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc      = nn.Linear(512, num_classes)

    def _m(self, p, n, s):
        ls = []
        for st in [s] + [1] * (n - 1):
            ls.append(_Block(self.ip, p, st)); self.ip = p
        return nn.Sequential(*ls)

    def forward(self, x, return_features=False):
        o    = F.relu(self.bn1(self.conv1(x)), True)
        o    = self.layer1(o); o = self.layer2(o)
        o    = self.layer3(o); o = self.layer4(o)
        feat = F.adaptive_avg_pool2d(o, 1).view(o.size(0), -1)
        logits = self.fc(self.dropout(feat))
        return (logits, feat) if return_features else logits


# ─────────────────────────────────────────────────────────────
# A1 — Standard Neural Network
# ─────────────────────────────────────────────────────────────

StandardNN = ResNet18   # alias


# ─────────────────────────────────────────────────────────────
# A2 — MC Dropout
# ─────────────────────────────────────────────────────────────

class MCDropoutNet(ResNet18):
    """ResNet-18 with MC Dropout active at test time."""

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__(num_classes=num_classes, dropout_rate=dropout_rate)

    @torch.no_grad()
    def mc_forward(self, x: torch.Tensor, n_passes: int = 20) -> torch.Tensor:
        """Return (n_passes, B, C) stochastic softmax probabilities."""
        self.train()   # keep dropout active
        out = [F.softmax(self(x), dim=-1) for _ in range(n_passes)]
        self.eval()
        return torch.stack(out)


# ─────────────────────────────────────────────────────────────
# A3 — Deep Ensemble
# ─────────────────────────────────────────────────────────────

class DeepEnsemble:
    """Collection of independently trained ResNet-18 models."""

    def __init__(self, members: list[ResNet18]):
        self.members = members

    @torch.no_grad()
    def predict(self, loader: DataLoader, device: torch.device):
        """Return (probs_stack [K,N,C], mean_probs [N,C], labels [N])."""
        all_probs = []
        labels    = None
        for mem in self.members:
            mem.eval()
            ps, ls = [], []
            for x, y in loader:
                ps.append(F.softmax(mem(x.to(device)), -1).cpu().numpy())
                ls.append(y.numpy())
            all_probs.append(np.concatenate(ps))
            if labels is None:
                labels = np.concatenate(ls)
        stack = np.stack(all_probs)   # (K, N, C)
        return stack, stack.mean(0), labels


# ─────────────────────────────────────────────────────────────
# A4 — Temperature Scaling
# ─────────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """Wraps a trained model and applies a learnable scalar temperature."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.tensor(1.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(x)
        return logits / self.temperature.clamp(min=1e-3)

    def calibrate(self, val_loader: DataLoader, device: torch.device) -> "TemperatureScaler":
        """Fit temperature on val set via NLL minimisation (LBFGS)."""
        self.to(device); self.model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                all_logits.append(self.model(x.to(device)))
                all_labels.append(y.to(device))
        logits = torch.cat(all_logits); labels = torch.cat(all_labels)
        opt    = optim.LBFGS([self.temperature], lr=0.01, max_iter=200)
        nll    = nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad()
            loss = nll(logits / self.temperature.clamp(min=1e-3), labels)
            loss.backward(); return loss
        opt.step(closure)
        print(f"  [TS] Optimal temperature: {self.temperature.item():.4f}")
        return self


# ─────────────────────────────────────────────────────────────
# A5 — Evidential Deep Learning
# ─────────────────────────────────────────────────────────────

class EDLNet(nn.Module):
    """Dirichlet-output network (Sensoy et al. 2018)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = ResNet18(num_classes=num_classes)

    def forward(self, x):
        return F.relu(self.backbone(x)) + 1.0   # alpha = evidence + 1


def edl_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    epoch: int,
    total_epochs: int,
    device: torch.device,
) -> torch.Tensor:
    """EDL composite loss with KL annealing (Sensoy et al. 2018, Eq. 5)."""
    y    = F.one_hot(labels, num_classes).float().to(device)
    S    = alpha.sum(dim=1, keepdim=True)
    # Multinomial NLL of Dirichlet-categorical
    nll  = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
    # Annealed KL toward uniform Dir
    lam  = min(1.0, epoch / max(1, total_epochs // 2))
    a_hat= y + (1 - y) * alpha
    S_hat= a_hat.sum(dim=1, keepdim=True)
    kl   = (torch.lgamma(S_hat)
            - torch.lgamma(torch.tensor(float(num_classes), device=device))
            - torch.lgamma(a_hat).sum(dim=1)
            + ((a_hat - 1) * (torch.digamma(a_hat) - torch.digamma(S_hat))).sum(dim=1))
    return (nll + lam * kl).mean()


# ─────────────────────────────────────────────────────────────
# A6 — EpiNet
# ─────────────────────────────────────────────────────────────

class EpiNet(nn.Module):
    """ResNet-18 + trainable epinet + fixed random prior (Osband et al. 2023)."""

    def __init__(self, num_classes=10, hidden=128, z_dim=16):
        super().__init__()
        self.backbone = ResNet18(num_classes=num_classes)
        self.z_dim    = z_dim
        D             = ResNet18.FEAT_DIM

        def _mlp():
            return nn.Sequential(
                nn.Linear(D + z_dim, hidden), nn.ReLU(True),
                nn.Linear(hidden, num_classes))

        self.epinet = _mlp()
        self.prior  = _mlp()
        for p in self.prior.parameters():
            p.requires_grad_(False)

    def _epi(self, feat, z):
        inp = torch.cat([feat, z], dim=1)
        return self.epinet(inp) + self.prior(inp)

    def forward(self, x):
        base, feat = self.backbone(x, return_features=True)
        z = torch.randn(x.size(0), self.z_dim, device=x.device)
        return base + self._epi(feat, z)

    @torch.no_grad()
    def mc_predict(self, x, n_samples=20):
        """Return (n_samples, B, C) softmax probs."""
        self.eval()
        base, feat = self.backbone(x, return_features=True)
        return torch.stack([
            F.softmax(base + self._epi(feat,
                torch.randn(x.size(0), self.z_dim, device=x.device)), -1)
            for _ in range(n_samples)
        ])


# ─────────────────────────────────────────────────────────────
# A7 — Mixture of Experts
# ─────────────────────────────────────────────────────────────

class MoENet(nn.Module):
    """ResNet-18 backbone + K expert heads + soft gating."""

    def __init__(self, num_classes=10, num_experts=5, expert_hidden=64):
        super().__init__()
        self.backbone    = ResNet18(num_classes=num_classes)
        self.num_experts = num_experts
        D = ResNet18.FEAT_DIM
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(D, expert_hidden), nn.ReLU(True),
                          nn.Linear(expert_hidden, num_classes))
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(D, num_experts)

    def forward(self, x, return_gate=False):
        _, feat = self.backbone(x, return_features=True)
        g       = F.softmax(self.gate(feat), dim=-1)          # (B, K)
        el      = torch.stack([e(feat) for e in self.experts], dim=1)  # (B, K, C)
        logits  = (g.unsqueeze(-1) * el).sum(dim=1)           # (B, C)
        return (logits, g) if return_gate else logits


# ─────────────────────────────────────────────────────────────
# B1 — Energy Score
# ─────────────────────────────────────────────────────────────

def energy_score(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """E(x) = −T log Σ_c exp(f_c / T).  Higher → more OOD.

    Reference: Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.
    """
    return -T * np.log(np.exp(logits / T).sum(axis=1))


# ─────────────────────────────────────────────────────────────
# B2 — ODIN
# ─────────────────────────────────────────────────────────────

def odin_score(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    temperature: float = 1000.0,
    epsilon:     float = 0.0014,
) -> np.ndarray:
    """ODIN uncertainty score = 1 − max ODIN-softmax.

    Reference: Liang et al., "Enhancing the Reliability of OOD Image
    Detection in Neural Networks", ICLR 2018.

    Returns higher values for OOD inputs.
    """
    model.eval()
    scores = []
    for x, _ in loader:
        x = x.to(device).requires_grad_(True)
        logits = model(x) / temperature
        (-F.log_softmax(logits, dim=-1).max(dim=1)[0].sum()).backward()
        with torch.no_grad():
            x_p   = (x - epsilon * x.grad.sign()).detach()
            probs = F.softmax(model(x_p) / temperature, dim=-1)
            scores.append((1.0 - probs.max(dim=1)[0]).cpu().numpy())
    return np.concatenate(scores)
