# Changelog

All notable changes to GOEN are documented here.
Format: [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2024-XX-XX

### Added
- Initial release of **GOEN** (Geometry-Optimised Epistemic Network).
- Three-phase training: CE + CenterLoss → Mahalanobis fitting → CalibHead.
- Multi-scale backbone: ResNet-18 with Layer2 + Layer4 feature concatenation.
- Calibration head trained on real SVHN hard-OOD + Gaussian noise.
- Spherical Mahalanobis scoring on L2-normalised features.
- Full benchmark against 11 baselines (A1–A7, B1–B4).
- Ablation study (4 variants × seed 42).
- Seeding study (5 seeds, mean ± std).
- CLI scripts: `train_goen`, `ablation`, `seeding`, `predict`, `train_baselines`.
- Unit tests for model, metrics, and data loading.
- GitHub Actions CI pipeline.

### Results
- **Avg OOD AUROC: 0.9337 ± 0.0085** (mean ± std over 5 seeds).
- Beats all 11 baselines including Deep Ensemble (0.883) and KNN (0.897).
- ID Accuracy: 0.934 ± 0.001.

---

## Planned

- [ ] ImageNet-scale experiments (ResNet-50 backbone).
- [ ] CIFAR-10-C robustness evaluation.
- [ ] Multi-dataset OOD evaluation (OpenImage-O, iNaturalist).
- [ ] Pretrained model weights on Hugging Face Hub.
- [ ] Streamlit demo application.
