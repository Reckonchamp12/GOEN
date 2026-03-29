# Baseline Methods

All baselines are trained on the same CIFAR-10 45k/5k train/val split (seed=42)
with a ResNet-18 backbone (adapted for 32×32), SGD + cosine LR, 100 epochs,
early stopping on val loss (patience=10).

---

## A-Group: Predictive Uncertainty Methods

### A1 — Standard Neural Network (MSP)
**Uncertainty:** `u(x) = 1 − max_c p(c|x)`  
**Reference:** Hendrycks & Gimpel, *A Baseline for Detecting Misclassified and
Out-of-Distribution Examples in Neural Networks*, ICLR 2017.  
**Notes:** Simplest possible baseline. Overconfident on OOD inputs by design.

### A1b — Temperature Scaling
**Uncertainty:** Same as A1 but logits divided by learned temperature T.  
**Reference:** Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.  
**Notes:** Post-hoc; does not change accuracy. T fitted on val set via LBFGS.
CIFAR-10 result: T = 1.10. Improves ECE 0.0223→0.0149 with no accuracy change.

### A2 — MC Dropout
**Uncertainty:** Mutual information `I[y; ω | x]` over T=20 forward passes.  
**Reference:** Gal & Ghahramani, *Dropout as a Bayesian Approximation*, ICML 2016.  
**Notes:** Dropout rate = 0.5 applied before the final FC layer. Training is
identical to A1. Test time requires T forward passes → T× slower.

### A3 — Deep Ensemble (K=5)
**Uncertainty:** Total variance `Σ_c Var_k[p_c(x)]` over K=5 members.  
**Reference:** Lakshminarayanan et al., *Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles*, NeurIPS 2017.  
**Notes:** 5 independent ResNet-18s trained from different seeds. Best single
baseline for ID accuracy (0.953). 5× inference cost.

### A4 — Evidential Deep Learning (EDL)
**Uncertainty:** Vacuity `u(x) = K / S` where `S = Σ_c alpha_c`.  
**Reference:** Sensoy et al., *Evidential Deep Learning to Quantify
Classification Uncertainty*, NeurIPS 2018.  
**Notes:** Network outputs evidence `e_c ≥ 0`; `alpha_c = e_c + 1` parameterises
a Dirichlet. Loss = multinomial NLL + KL-annealed regularisation. Worst
calibration (ECE 0.194) due to the Dirichlet parameterisation.

### A5 — EpiNet
**Uncertainty:** Mutual information via index samples `z ~ N(0,I)`.  
**Reference:** Osband et al., *Epistemic Neural Networks*, NeurIPS 2023.  
**Notes:** Adds trainable epinet + fixed random prior MLP on top of the base
network's features. Prior is randomly initialised and never updated.

### A6 — Mixture of Experts (MoE, K=5)
**Uncertainty:** `1 − max_k g_k(x)` where `g` is the soft gate.  
**Notes:** K=5 expert heads + a linear gating network on top of shared
ResNet-18 features. Weakest OOD baseline; gate weights collapse to one
dominant expert on ID data, giving poor OOD separation.

---

## B-Group: Post-Hoc OOD Detectors (applied to A1 features/logits)

### B1 — Energy Score
**Score:** `E(x; T) = −T log Σ_c exp(f_c(x)/T)`. Higher = more OOD.  
**Reference:** Liu et al., *Energy-based Out-of-distribution Detection*, NeurIPS 2020.  
**Notes:** Uses the full logit distribution; provably more informative than MSP.
T=1.0 used. Strong on synthetic (0.948) and CIFAR-100 (0.855).

### B2 — ODIN
**Score:** `1 − max ODIN-softmax`. Temperature T=1000, ε=0.0014.  
**Reference:** Liang et al., *Enhancing the Reliability of OOD Image Detection
in Neural Networks*, ICLR 2018.  
**Notes:** Input pre-processing (sign gradient step) combined with temperature
sharpening. Best non-ensemble post-hoc method on CIFAR-100 and synthetic.
Requires backprop at test time → slower inference.

### B3 — Mahalanobis Distance
**Score:** `min_c (z−μ_c)^T Σ^{-1} (z−μ_c)`. Higher = more OOD.  
**Reference:** Lee et al., *A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks*, NeurIPS 2018.  
**Notes:** Weakest baseline despite theoretical strength. CE features are
not compact (ratio 0.43) → covariance is ill-conditioned → poor separation.
**GOEN fixes this** by training compact features first (ratio 7–12×).

### B4 — KNN Distance
**Score:** Cosine distance to k-th nearest training neighbour (k=5).  
**Reference:** Sun et al., *Out-of-Distribution Detection with Deep Nearest
Neighbors*, ICML 2022.  
**Notes:** Previous best single baseline (avg AUROC 0.897). Instance-based,
requires storing all training features at inference time (~45k × 512 floats ≈ 90 MB).
Excellent on synthetic (0.991) but weaker on SVHN (0.850).

---

## Comparison Summary

| Method | Type | Avg AUROC | ID Accuracy | Inference Cost |
|--------|------|-----------|-------------|----------------|
| MSP | Predictive | 0.867 | 0.871 | 1× |
| Temp Scaling | Post-hoc cal. | 0.868 | 0.871 | 1× |
| MC Dropout | Bayesian | 0.873 | 0.930 | T× |
| Deep Ensemble | Ensemble | 0.883 | **0.953** | K× |
| EDL | Evidential | 0.858 | 0.820 | 1× |
| EpiNet | Epistemic | 0.875 | 0.890 | S× |
| MoE-K5 | Routing | 0.705 | 0.884 | 1× |
| Energy | Post-hoc | 0.880 | 0.871 | 1× |
| ODIN | Post-hoc | 0.887 | 0.871 | 1× (+ grad) |
| Mahalanobis | Post-hoc | 0.679 | 0.871 | 1× |
| KNN | Post-hoc | 0.897 | 0.871 | 1× |
| **GOEN** | Geometric | **0.934** | 0.934 | **1×** |
