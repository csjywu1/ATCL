# ACTP: GeoLife Reproducibility Package

Clean reference implementation of Adaptive Contrastive Learning for Trajectory Prediction (ACTP) on GeoLife. The repository separates data processing, model definition, training, and inference into independent modules and uses a single JSON configuration.

## Project structure

```text
ACTP_GeoLife_reproducible/
├── configs/
│   └── geolife.json
├── data/
│   ├── dataset.py
│   ├── prepare.py
│   ├── raw/
│   └── processed/
├── model/
│   ├── actp.py
│   ├── geometry.py
│   └── losses.py
├── train/
│   ├── engine.py
│   └── train.py
├── inference/
│   ├── evaluate.py
│   └── predict.py
├── checkpoints/
├── results/
├── CITATION.cff
├── DATA_LICENSE.md
├── LICENSE
├── SHA256SUMS
└── requirements.txt
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data preparation

Place the official GeoLife archive at `data/raw/Geolife_Trajectories_1.3.zip`, then run:

```bash
python -m data.prepare \
  --input data/raw/Geolife_Trajectories_1.3.zip \
  --output data/processed/geolife_20k.pkl \
  --limit 20000 --min-points 36
```

The preprocessing step extracts valid `.plt` files, removes discontinuities larger than 2 km, and retains trajectories long enough for a historical prefix and five future locations.

## Training

All experiment settings are stored in `configs/geolife.json`.

```bash
python -m train.train --config configs/geolife.json
```

The best validation checkpoint and its JSON report are written to `results/runs/`. Change `seed` while keeping `split_seed` fixed to evaluate initialization stability on the same train/validation/test split.

## Evaluation

Evaluate the released checkpoint on the fixed GeoLife split:

```bash
python -m inference.evaluate \
  --config configs/geolife.json \
  --checkpoint checkpoints/actp_geolife_seed3.pt
```

The released fixed-split runs report **27.17 ± 0.18 m MAE** and **105.13 ± 0.58 m RMSE** over three initialization seeds. The machine-readable summary is in `results/fixed_split_summary.json`.

Verify the released data and checkpoint with `shasum -a 256 -c SHA256SUMS`.

## Inference on one trajectory

Create a JSON file containing a list of observed `[latitude, longitude]` points, then run:

```bash
python -m inference.predict \
  --trajectory example_trajectory.json \
  --config configs/geolife.json \
  --checkpoint checkpoints/actp_geolife_seed3.pt
```

The command returns five predicted metric offsets relative to the last observed point.

## Reproducibility notes

- Prediction horizon: five recorded GPS locations.
- Split: trajectory-level 80/10/10 split with `split_seed=3`.
- Model: three scale encoders at 50 m, 150 m, and 500 m; Poincaré cross-scale contrast; radial scale order; adaptive scale fusion.
- Released configuration: hidden size 2048, batch size 32, SmoothL1 loss with beta 50, and full-history context projection.
- Dataset redistribution is governed by the original GeoLife terms; see `DATA_LICENSE.md` before publishing the archive.

## Legacy backup

The earlier monolithic scripts, exploratory logs, and redundant checkpoints were moved outside this repository to `../ACTP_GeoLife_reproducible_legacy_20260902/`. They are retained only for audit and are not part of the public interface.
