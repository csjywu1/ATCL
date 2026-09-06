# ACTP: Adaptive Contrastive Learning for Trajectory Prediction

PyTorch implementation and reproducibility package for the audited ACTP trajectory-prediction experiments. The code predicts the next five recorded GPS locations from an observed trajectory prefix and implements coarse-to-fine scale encoding, learnable Poincare geometry, hyperbolic cross-scale contrast, radial ordering, and configurable decoding.

## Structure

```text
actp/                 exact audited experiment runner
data/                 loaders, preprocessing, and fixed trajectory splits
model/                ACTP, Poincare operations, and objectives
train/                configuration-driven training and reusable engine API
inference/            frozen-checkpoint evaluation and single-route prediction
configs/              WorldTrace, Chengdu, and GeoLife configurations
scripts/              reproducibility entry points
tests/                CPU smoke tests
results/              released aggregate metrics and checkpoint manifest
checkpoints/          user-downloaded or trained checkpoints
```

## Install

```bash
git clone https://github.com/csjywu1/ATCL.git
cd ATCL
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The repository does not redistribute trajectory datasets. Place authorized, processed files at:

```text
data/processed/worldtrace_3s_20k.pkl
data/processed/chengdu_coordinate_20000.pkl
data/processed/geolife_20k.pkl
```

Raw GeoLife, WorldTrace, and Chengdu sources can also be normalized with `python -m data.prepare`. See [`data/README.md`](data/README.md) and [`DATA_LICENSE.md`](DATA_LICENSE.md).

## Train

Each released configuration fixes the data split independently of model initialization and keeps the test split sealed during checkpoint selection.

```bash
python -m train.train --config configs/worldtrace.json
python -m train.train --config configs/chengdu.json
python -m train.train --config configs/geolife.json
```

Use another initialization without editing the configuration:

```bash
bash scripts/reproduce.sh geolife 101
```

## Evaluate a frozen checkpoint

```bash
python -m inference.evaluate \
  --report results/runs/geolife_seed101.json \
  --checkpoint results/runs/geolife_seed101.json.pt \
  --data-path data/processed/geolife_20k.pkl
```

The evaluation command never trains or selects a checkpoint.

## Predict one trajectory

Store observed points as a JSON array of `[latitude, longitude]` pairs, then run:

```bash
python -m inference.predict \
  --trajectory observed.json \
  --report results/runs/geolife_seed101.json \
  --checkpoint results/runs/geolife_seed101.json.pt
```

The output contains five future east/north offsets in meters relative to the final observed location.

## Released results

Three model-initialization seeds are evaluated on one fixed trajectory split per dataset. Values are mean +/- sample standard deviation in meters.

| Dataset | MAE | RMSE |
|---|---:|---:|
| WorldTrace | **12.20 +/- 0.21** | **22.38 +/- 0.17** |
| Chengdu | **13.02 +/- 0.19** | **22.90 +/- 0.39** |
| GeoLife | **18.58 +/- 0.38** | **45.72 +/- 0.73** |

The machine-readable per-seed records and aggregate statistics are in [`results/`](results/). Chengdu applies a 300 m maximum consecutive-displacement quality-control rule before splitting; use the same filtered subset for matched comparisons.

## Reproducibility policy

- Prediction horizon: five recorded future GPS locations.
- Split: trajectory-level 80/10/10 with `split_seed=20260906`.
- Checkpoint selection: validation RMSE only.
- Test policy: sealed during tuning, followed by one frozen-checkpoint evaluation.
- Spatial views: local, neighborhood, and regional scales at 50, 150, and 500 m.
- Integrity: released results include source and checkpoint SHA-256 hashes.

`actp/runner.py` is the exact source used for the released runs. The surrounding modules expose the same implementation through standard data, model, training, and inference entry points.

## Test

```bash
python -m unittest discover -s tests -v
```

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Source code is released under the MIT License; datasets remain subject to their original terms.
