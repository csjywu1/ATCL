# ACTP: Adaptive Contrastive Learning for Trajectory Prediction

Official reproducibility package for **ACTP** on the GeoLife trajectory-prediction benchmark. ACTP combines multi-scale spatial supervision with hyperbolic cross-scale contrastive learning to model local, neighborhood, and regional movement patterns. The repository provides separate modules for data preparation, model definition, training, evaluation, and single-trajectory inference.

Repository: [https://github.com/csjywu1/ATCL](https://github.com/csjywu1/ATCL)

## Repository layout

```text
.
├── configs/
│   └── geolife.json               # Reproducible GeoLife configuration
├── data/
│   ├── dataset.py                 # Dataset, split, and forecast samples
│   ├── prepare.py                 # Raw GeoLife preprocessing
│   ├── raw/                       # User-provided raw archive
│   └── processed/                 # Generated trajectory file
├── model/
│   ├── actp.py                    # ACTP network
│   ├── geometry.py                # Poincare-ball operations
│   └── losses.py                  # Prediction and auxiliary objectives
├── train/
│   ├── engine.py                  # Training and evaluation loops
│   └── train.py                   # Training entry point
├── inference/
│   ├── evaluate.py                # Fixed-split evaluation
│   └── predict.py                 # Single-trajectory prediction
├── checkpoints/                   # Released checkpoint (Git LFS)
├── results/                       # Reproducibility summary
├── CITATION.cff
├── DATA_LICENSE.md
├── LICENSE
├── SHA256SUMS
└── requirements.txt
```

## Installation

Python 3.10 or newer is recommended. CUDA is used automatically when available; CPU execution is also supported.

```bash
git clone https://github.com/csjywu1/ATCL.git
cd ATCL

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The released checkpoint is stored with Git LFS. Install Git LFS before cloning, or run `git lfs pull` inside an existing clone if the checkpoint is represented by a pointer file.

## Data preparation

Download the official **GeoLife GPS Trajectories 1.3** archive and place it at:

```text
data/raw/Geolife_Trajectories_1.3.zip
```

Then preprocess the trajectories:

```bash
python -m data.prepare \
  --input data/raw/Geolife_Trajectories_1.3.zip \
  --output data/processed/geolife_20k.pkl \
  --limit 20000 \
  --min-points 36
```

The preprocessing script parses valid `.plt` files, removes trajectories containing spatial discontinuities larger than 2 km, and retains trajectories long enough to form a historical prefix and five recorded future locations. GeoLife is not redistributed by this repository; consult [DATA_LICENSE.md](DATA_LICENSE.md) before using or sharing the dataset.

## Evaluate the released model

After preparing the data, evaluate the released checkpoint on the fixed trajectory-level test split:

```bash
python -m inference.evaluate \
  --config configs/geolife.json \
  --checkpoint checkpoints/actp_geolife_seed3.pt
```

The released fixed-split experiments use three initialization seeds while keeping the data split fixed.

| Dataset | MAE (m) | RMSE (m) |
|---|---:|---:|
| GeoLife | **27.17 +/- 0.18** | **105.13 +/- 0.58** |

The machine-readable result summary is available in [`results/fixed_split_summary.json`](results/fixed_split_summary.json).

## Train ACTP

All model and optimization settings are defined in [`configs/geolife.json`](configs/geolife.json):

```bash
python -m train.train --config configs/geolife.json
```

The training script saves the best validation checkpoint and a JSON report under `results/runs/`. To measure initialization stability, change `seed` while keeping `split_seed` fixed. The released configuration uses a 2048-dimensional hidden state, full-history context projection, SmoothL1 prediction loss, and a five-location prediction horizon.

## Predict one trajectory

Create a JSON file containing the observed GPS points as an array of `[latitude, longitude]` pairs:

```json
[
  [39.984702, 116.318417],
  [39.984683, 116.318450],
  [39.984686, 116.318474]
]
```

Run inference with:

```bash
python -m inference.predict \
  --trajectory example_trajectory.json \
  --config configs/geolife.json \
  --checkpoint checkpoints/actp_geolife_seed3.pt
```

The command returns five future metric offsets relative to the final observed point.

## Reproducibility details

- Prediction horizon: five recorded future GPS locations.
- Data split: trajectory-level 80/10/10 split with `split_seed=3`.
- Spatial scales: local, neighborhood, and regional states at 50 m, 150 m, and 500 m.
- Representation objective: Poincare cross-scale contrast with radial scale ordering.
- Released configuration: hidden size 2048, batch size 32, SmoothL1 beta 50, and full-history context projection.
- Integrity check: run `shasum -a 256 -c SHA256SUMS` after preparing the referenced files.

## Citation

If this repository contributes to your research, please cite the ACTP paper. Citation metadata is provided in [`CITATION.cff`](CITATION.cff) and is also exposed through GitHub's **Cite this repository** function.

## License

The source code is released under the [MIT License](LICENSE). The GeoLife dataset remains subject to its original terms of use; see [DATA_LICENSE.md](DATA_LICENSE.md).
