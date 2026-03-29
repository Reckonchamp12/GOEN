# Quickstart Guide

## Installation

```bash
git clone https://github.com/your-org/goen.git
cd goen
pip install -r requirements.txt
pip install -e .
```

## Minimum working example (5 lines)

```python
from goen import GOEN, Trainer, get_default_config

cfg     = get_default_config()          # loads sensible defaults
trainer = Trainer(cfg)
model   = trainer.train()               # ~18 min on a T4 GPU
results = trainer.evaluate(model)
print(f"Avg OOD AUROC: {results['avg_auroc']:.4f}")
```

## Custom configuration

```python
cfg = get_default_config()
cfg["data_root"]    = "/path/to/data"
cfg["output_dir"]   = "/path/to/checkpoints"
cfg["seed"]         = 123
cfg["p1_epochs"]    = 40        # faster run for debugging
cfg["center_alpha"] = 0.01      # CenterLoss weight
cfg["single_scale"] = False     # True = Layer4 only (ablation)
cfg["svhn_ood"]     = True      # False = noise-only Phase 3 (ablation)
```

## Load from YAML

```python
from goen import load_config, Trainer

cfg     = load_config("configs/goen_default.yaml")
trainer = Trainer(cfg)
model   = trainer.train()
```

## Load a pretrained checkpoint

```python
from goen import load_pretrained
import torch

model = load_pretrained("checkpoints/goen_seed42.pt")
model.eval()

# Run inference
images = torch.randn(4, 3, 32, 32)  # your CIFAR-10 normalised images
with torch.no_grad():
    out = model.predict(images)

print(out["probs"])        # (4, 10) class probabilities
print(out["pred"])         # (4,)    predicted class indices
print(out["uncertainty"])  # (4,)    calibrated uncertainty in (0, 1)
```

## Predict on a folder of images

```bash
python scripts/predict.py \
    --checkpoint checkpoints/goen_seed42.pt \
    --image_dir  path/to/my/images/ \
    --output     predictions.csv \
    --threshold  0.6   # flag images with u > 0.6 as OOD
```

Output CSV columns: `filename, pred_class, confidence, uncertainty, flag`.

## Fast debug run (CPU, ~3 min)

```bash
python scripts/train_goen.py \
    --config configs/goen_fast.yaml \
    --seed 42
```

## Reproduce all paper results

```bash
# 1. All baselines (~6 h on GPU)
python scripts/train_baselines.py --data_root ./data --output_dir ./results/baselines

# 2. GOEN default (seed=42, ~18 min)
python scripts/train_goen.py --config configs/goen_default.yaml --seed 42

# 3. Ablation study (~72 min)
python scripts/ablation.py --config configs/goen_default.yaml

# 4. Seeding study (~90 min)
python scripts/seeding.py --seeds 42 123 2024 777 314

# 5. Generate all figures
python scripts/make_plots.py --results_dir ./results --output_dir ./figures
```

## Run tests

```bash
pytest tests/ -v
```

## Expected CIFAR-10 class labels

```
0: airplane   1: automobile  2: bird    3: cat    4: deer
5: dog        6: frog        7: horse   8: ship   9: truck
```

## Interpreting uncertainty scores

| u(x) range | Interpretation |
|------------|----------------|
| 0.00 – 0.20 | High confidence in-distribution |
| 0.20 – 0.50 | Moderate uncertainty |
| 0.50 – 0.80 | High uncertainty — likely near decision boundary or domain shift |
| 0.80 – 1.00 | Very high uncertainty — likely OOD |

These thresholds are approximate and depend on the calibration of Phase 3.
Tune using a held-out set with known ID/OOD labels.
