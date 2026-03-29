"""
goen/data.py
============
Data loading utilities for CIFAR-10 (ID) and OOD datasets.

OOD datasets supported:
  - SVHN          (near-OOD — hardest: lives in same feature direction)
  - CIFAR-100     (near-OOD — semantically disjoint subset)
  - Synthetic     (far-OOD  — Gaussian noise in CIFAR-10 normalised space)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# CIFAR-100 classes semantically disjoint from CIFAR-10
_CIFAR100_SAFE = sorted(set([
    54, 62, 70, 82, 92,   # flowers
    0,  51, 57, 61, 72,   # fruit & veg
    9,  10, 16, 28,       # containers
    22, 39, 40, 86, 87,   # household electrical
    5,  20, 25, 84, 94,   # furniture
    6,  7,  14, 18, 24,   # insects
    26, 45, 77, 79, 99,   # invertebrates
    2,  11, 35, 46, 98,   # people
    47, 52, 56, 59, 96,   # trees
    30, 32, 49, 65, 73,   # outdoor scenes
]))[:50]

_train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

_test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def _norm_tensor(imgs: torch.Tensor) -> torch.Tensor:
    """Normalise a (N,3,H,W) float tensor in [0,1] to CIFAR-10 stats."""
    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD,  dtype=torch.float32).view(1, 3, 1, 1)
    return (imgs - mean) / std


def get_cifar10_loaders(
    cfg: dict,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Build CIFAR-10 data loaders.

    Returns:
        train_l: Augmented training loader   (45 k samples).
        val_l:   Non-augmented val loader    (5 k samples).
        feat_l:  Non-augmented training loader for Phase 2 Maha fitting.
        test_l:  Non-augmented test loader   (10 k samples).
    """
    root = cfg["data_root"]
    rng  = np.random.RandomState(cfg["seed"])
    idx  = rng.permutation(50_000)
    vi   = idx[: cfg["val_size"]]
    ti   = idx[cfg["val_size"]:]

    fa  = CIFAR10(root, train=True,  download=True, transform=_train_tf)
    fn  = CIFAR10(root, train=True,  download=True, transform=_test_tf)
    te  = CIFAR10(root, train=False, download=True, transform=_test_tf)

    bs = cfg["batch_size"]
    train_l = DataLoader(Subset(fa, ti), bs, shuffle=True,  num_workers=2, pin_memory=True)
    val_l   = DataLoader(Subset(fn, vi), bs, shuffle=False, num_workers=2, pin_memory=True)
    feat_l  = DataLoader(Subset(fn, ti), 512, shuffle=False, num_workers=2, pin_memory=True)
    test_l  = DataLoader(te,             bs,  shuffle=False, num_workers=2, pin_memory=True)

    return train_l, val_l, feat_l, test_l


def get_ood_loaders(cfg: dict) -> dict[str, DataLoader]:
    """Build all standard OOD evaluation loaders.

    Returns:
        dict mapping dataset name to DataLoader.
        Keys: "svhn", "cifar100", "synthetic".
    """
    loaders: dict[str, DataLoader] = {}
    for name in ("svhn", "cifar100", "synthetic"):
        try:
            loaders[name] = _get_single_ood(name, cfg)
        except Exception as exc:
            print(f"  [data] OOD loader '{name}' skipped: {exc}")
    return loaders


def _get_single_ood(name: str, cfg: dict) -> DataLoader:
    """Build a single OOD DataLoader by name."""
    root = cfg["data_root"]
    bs   = cfg["batch_size"]

    if name == "svhn":
        ds = SVHN(root, split="test", download=True, transform=_test_tf)
        return DataLoader(ds, bs, shuffle=False, num_workers=2, pin_memory=True)

    if name == "cifar100":
        ds   = CIFAR100(root, train=False, download=True, transform=_test_tf)
        mask = np.isin(np.array(ds.targets), _CIFAR100_SAFE)
        sub  = Subset(ds, np.where(mask)[0])
        return DataLoader(sub, bs, shuffle=False, num_workers=2)

    if name == "synthetic":
        g    = torch.Generator()
        g.manual_seed(0)
        imgs = torch.clamp(
            torch.randn(5000, 3, 32, 32, generator=g) * 0.5 + 0.5, 0.0, 1.0,
        )
        imgs = _norm_tensor(imgs)
        ds   = TensorDataset(imgs, torch.zeros(5000, dtype=torch.long))
        return DataLoader(ds, bs, shuffle=False)

    raise ValueError(f"Unknown OOD dataset: '{name}'")


def get_svhn_ood_loader(cfg: dict) -> DataLoader:
    """Load a fixed random subset of SVHN test as hard-OOD for Phase 3.

    Args:
        cfg: Config dict with keys: data_root, batch_size, svhn_n.

    Returns:
        DataLoader over svhn_n SVHN images.
    """
    ds  = SVHN(cfg["data_root"], split="test", download=True, transform=_test_tf)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(ds), min(cfg.get("svhn_n", 5000), len(ds)), replace=False)
    sub = Subset(ds, idx)
    return DataLoader(sub, cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
