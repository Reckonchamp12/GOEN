#!/usr/bin/env python3
"""
scripts/train_baselines.py
==========================
Train and evaluate all 11 baseline models on CIFAR-10.

Baselines
---------
  A1  Standard NN (MSP)
  A1b Temperature Scaling
  A2  MC Dropout
  A3  Deep Ensemble (5 members)
  A5  EDL
  A6  EpiNet
  A7  MoE-K5
  B1  Energy Score      (post-hoc on A1)
  B2  ODIN              (post-hoc on A1)
  B3  Mahalanobis       (post-hoc on A1)
  B4  KNN               (post-hoc on A1)

Usage::

    python scripts/train_baselines.py --data_root ./data --output_dir ./results/baselines
    python scripts/train_baselines.py --method mc_dropout --seed 42
"""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goen.data    import get_cifar10_loaders, get_ood_loaders
from goen.metrics import compute_ece, compute_nll, compute_brier, compute_ood_metrics
from goen.metrics import mutual_information, ensemble_variance, edl_vacuity
from goen.utils   import set_seed, ResultsBook

from baselines import (
    ResNet18, MCDropoutNet, DeepEnsemble, TemperatureScaler,
    EDLNet, edl_loss, EpiNet, MoENet,
    energy_score, odin_score,
)
from goen.detectors import MahalanobisDetector, KNNDetector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# Generic training loop
# ─────────────────────────────────────────────────────────────

def _train(model, train_l, val_l, cfg, is_edl=False, seed=42):
    set_seed(seed)
    model  = model.to(DEVICE)
    opt    = optim.SGD(model.parameters(), lr=cfg["lr"],
                       momentum=0.9, weight_decay=5e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    ce     = nn.CrossEntropyLoss()
    best_vl= float("inf"); best_sd=None; pat=0

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        for x, y in train_l:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out  = model(x)
            if is_edl:
                loss = edl_loss(out, y, cfg["num_classes"], ep, cfg["epochs"], DEVICE)
            else:
                loss = ce(out, y)
            loss.backward(); opt.step()
        sched.step()

        model.eval(); vl_sum = vl_n = 0
        with torch.no_grad():
            for x, y in val_l:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out  = model(x)
                if is_edl:
                    l = edl_loss(out, y, cfg["num_classes"], ep, cfg["epochs"], DEVICE)
                else:
                    l = ce(out, y)
                vl_sum += l.item() * x.size(0); vl_n += x.size(0)
        vl = vl_sum / vl_n

        if vl < best_vl: best_vl=vl; best_sd=deepcopy(model.state_dict()); pat=0
        else: pat += 1
        if ep % 25 == 0 or ep == 1:
            print(f"    ep {ep:3d}  val_loss={vl:.4f}  (best={best_vl:.4f}  pat={pat})")
        if pat >= cfg.get("patience", 10): break

    model.load_state_dict(best_sd)
    return model


@torch.no_grad()
def _softmax_probs(model, loader):
    model.eval()
    ps, ls = [], []
    for x, y in loader:
        ps.append(F.softmax(model(x.to(DEVICE)), -1).cpu().numpy())
        ls.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ls)


@torch.no_grad()
def _feats_logits(model, loader):
    model.eval()
    fs, lgs, ls = [], [], []
    for x, y in loader:
        lg, f = model(x.to(DEVICE), return_features=True)
        fs.append(f.cpu().numpy()); lgs.append(lg.cpu().numpy())
        ls.append(y.numpy())
    return np.concatenate(fs), np.concatenate(lgs), np.concatenate(ls)


def _log_id(book, probs, labels, name):
    m = dict(Accuracy=float((probs.argmax(1)==labels).mean()),
             ECE=compute_ece(probs,labels),
             NLL=compute_nll(probs,labels),
             Brier=compute_brier(probs,labels))
    book.record(name, "ID-CIFAR10", m)
    print(f"  [ID] {name}: Acc={m['Accuracy']:.4f}  ECE={m['ECE']:.4f}  "
          f"NLL={m['NLL']:.4f}  Brier={m['Brier']:.4f}")


def _log_ood(book, id_sc, ood_dict, name):
    for ood_name, ood_sc in ood_dict.items():
        m = compute_ood_metrics(id_sc, ood_sc)
        book.record(name, f"OOD-{ood_name}", m)
        print(f"  [OOD-{ood_name}] {name}: "
              f"AUROC={m['AUROC']:.4f}  FPR95={m['FPR95']:.4f}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  default="./data")
    parser.add_argument("--output_dir", default="./results/baselines")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--method",     default="all",
        help="Which baseline to run (all, standard_nn, mc_dropout, deep_ensemble, "
             "edl, epinet, moe, energy, odin, mahalanobis, knn)")
    args = parser.parse_args()

    cfg = dict(
        data_root   = args.data_root,
        output_dir  = args.output_dir,
        batch_size  = 128,
        num_classes = 10,
        val_size    = 5000,
        seed        = args.seed,
        epochs      = args.epochs,
        lr          = 0.1,
        patience    = 10,
    )

    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    book = ResultsBook()
    t0   = time.time()

    set_seed(cfg["seed"])
    train_l, val_l, feat_l, test_l = get_cifar10_loaders(cfg)
    ood_loaders = get_ood_loaders(cfg)

    # ── A1: Standard NN ──────────────────────────────────────
    if args.method in ("all", "standard_nn"):
        print("\n═══ A1: Standard NN ═══")
        std = _train(ResNet18(), train_l, val_l, cfg)
        torch.save(std.state_dict(), out / "standardnn.pt")

        probs, labels = _softmax_probs(std, test_l)
        _log_id(book, probs, labels, "StandardNN")

        # Extract features for post-hoc methods
        tr_feats, tr_logits, tr_labels = _feats_logits(std, feat_l)
        te_feats, te_logits, _         = _feats_logits(std, test_l)

        id_sc = 1.0 - probs.max(1)
        ood_sc = {}
        for n, dl in ood_loaders.items():
            p, _ = _softmax_probs(std, dl); ood_sc[n] = 1.0 - p.max(1)
        _log_ood(book, id_sc, ood_sc, "StandardNN")

        # A1b: Temperature Scaling
        print("\n─── A1b: Temperature Scaling ───")
        ts = TemperatureScaler(std).calibrate(val_l, DEVICE)
        ts.eval()
        ts_ps = []
        with torch.no_grad():
            for x, _ in test_l:
                ts_ps.append(F.softmax(ts(x.to(DEVICE)), -1).cpu().numpy())
        ts_probs = np.concatenate(ts_ps)
        _log_id(book, ts_probs, labels, "TempScaling")
        id_ts_sc  = 1.0 - ts_probs.max(1)
        ood_ts_sc = {}
        for n, dl in ood_loaders.items():
            p_list = []
            with torch.no_grad():
                for x, _ in dl: p_list.append(F.softmax(ts(x.to(DEVICE)),-1).cpu().numpy())
            ood_ts_sc[n] = 1.0 - np.concatenate(p_list).max(1)
        _log_ood(book, id_ts_sc, ood_ts_sc, "TempScaling")

    # ── A2: MC Dropout ───────────────────────────────────────
    if args.method in ("all", "mc_dropout"):
        print("\n═══ A2: MC Dropout ═══")
        mc = _train(MCDropoutNet(dropout_rate=0.5), train_l, val_l, cfg)
        torch.save(mc.state_dict(), out / "mcdropout.pt")
        mc_pss, mc_labels = [], []
        for x, y in test_l:
            mc_pss.append(mc.mc_forward(x.to(DEVICE), 20).cpu().numpy())
            mc_labels.append(y.numpy())
        mc_stack  = np.concatenate(mc_pss, axis=1)
        mc_labels = np.concatenate(mc_labels)
        _log_id(book, mc_stack.mean(0), mc_labels, "MCDropout")
        id_mc_sc = mutual_information(mc_stack)
        ood_mc_sc = {}
        for n, dl in ood_loaders.items():
            pss = [mc.mc_forward(x.to(DEVICE), 20).cpu().numpy() for x,_ in dl]
            st  = np.concatenate(pss, axis=1)
            ood_mc_sc[n] = mutual_information(st)
        _log_ood(book, id_mc_sc, ood_mc_sc, "MCDropout")

    # ── A3: Deep Ensemble ─────────────────────────────────────
    if args.method in ("all", "deep_ensemble"):
        print("\n═══ A3: Deep Ensemble (5×) ═══")
        seeds   = [42, 123, 2024, 777, 314]
        members = []
        for i, s in enumerate(seeds):
            print(f"  Training member {i+1}/5  (seed={s})")
            m = _train(ResNet18(), train_l, val_l, cfg, seed=s)
            torch.save(m.state_dict(), out / f"ensemble_m{i+1}.pt")
            members.append(m)
        ens = DeepEnsemble(members)
        ens_stack, ens_mean, ens_labels = ens.predict(test_l, DEVICE)
        _log_id(book, ens_mean, ens_labels, "DeepEnsemble")
        id_ens_sc  = ensemble_variance(ens_stack)
        ood_ens_sc = {}
        for n, dl in ood_loaders.items():
            st, _, _ = ens.predict(dl, DEVICE); ood_ens_sc[n] = ensemble_variance(st)
        _log_ood(book, id_ens_sc, ood_ens_sc, "DeepEnsemble")

    # ── A5: EDL ──────────────────────────────────────────────
    if args.method in ("all", "edl"):
        print("\n═══ A5: EDL ═══")
        edl = _train(EDLNet(), train_l, val_l, cfg, is_edl=True)
        torch.save(edl.state_dict(), out / "edl.pt")
        edl.eval()
        alphas, edl_labels = [], []
        with torch.no_grad():
            for x, y in test_l:
                alphas.append(edl(x.to(DEVICE)).cpu().numpy())
                edl_labels.append(y.numpy())
        alpha_arr = np.concatenate(alphas); edl_labels = np.concatenate(edl_labels)
        edl_probs = alpha_arr / alpha_arr.sum(1, keepdims=True)
        _log_id(book, edl_probs, edl_labels, "EDL")
        id_edl_sc = edl_vacuity(alpha_arr)
        ood_edl_sc = {}
        for n, dl in ood_loaders.items():
            a_l = []
            with torch.no_grad():
                for x, _ in dl: a_l.append(edl(x.to(DEVICE)).cpu().numpy())
            ood_edl_sc[n] = edl_vacuity(np.concatenate(a_l))
        _log_ood(book, id_edl_sc, ood_edl_sc, "EDL")

    # ── B1: Energy ───────────────────────────────────────────
    if args.method in ("all", "energy") and "te_logits" in dir():
        print("\n═══ B1: Energy Score ═══")
        id_en = energy_score(te_logits)
        ood_en = {}
        for n, dl in ood_loaders.items():
            _, lg, _ = _feats_logits(std, dl); ood_en[n] = energy_score(lg)
        _log_ood(book, id_en, ood_en, "Energy")

    # ── B2: ODIN ─────────────────────────────────────────────
    if args.method in ("all", "odin") and "std" in dir():
        print("\n═══ B2: ODIN ═══")
        id_od = odin_score(std, test_l, DEVICE)
        ood_od = {n: odin_score(std, dl, DEVICE) for n, dl in ood_loaders.items()}
        _log_ood(book, id_od, ood_od, "ODIN")

    # ── B3: Mahalanobis ──────────────────────────────────────
    if args.method in ("all", "mahalanobis") and "tr_feats" in dir():
        print("\n═══ B3: Mahalanobis ═══")
        maha = MahalanobisDetector().fit(tr_feats, tr_labels)
        id_ma = maha.score(te_feats)
        ood_ma = {}
        for n, dl in ood_loaders.items():
            ft, _, _ = _feats_logits(std, dl); ood_ma[n] = maha.score(ft)
        _log_ood(book, id_ma, ood_ma, "Mahalanobis")

    # ── B4: KNN ──────────────────────────────────────────────
    if args.method in ("all", "knn") and "tr_feats" in dir():
        print("\n═══ B4: KNN (k=5) ═══")
        knn = KNNDetector(k=5).fit(tr_feats)
        id_kn = knn.score(te_feats)
        ood_kn = {}
        for n, dl in ood_loaders.items():
            ft, _, _ = _feats_logits(std, dl); ood_kn[n] = knn.score(ft)
        _log_ood(book, id_kn, ood_kn, "KNN")

    # ── Save results ─────────────────────────────────────────
    res_path = out / "all_results.json"
    book.save(res_path)
    book.print_summary()
    print(f"\n[Done] Total time: {(time.time()-t0)/3600:.2f} h")
    print(f"[Done] Results → {res_path}")


if __name__ == "__main__":
    main()
