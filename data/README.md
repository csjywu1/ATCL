# Data

ACTP supports WorldTrace, Chengdu, and GeoLife. Dataset files are not committed.

Expected processed paths are:

- `data/processed/worldtrace_3s_20k.pkl`
- `data/processed/chengdu_coordinate_20000.pkl`
- `data/processed/geolife_20k.pkl`

Create a portable pickle from the project root:

```bash
python -m data.prepare \
  --dataset geolife \
  --input data/raw/Geolife_Trajectories_1.3.zip \
  --output data/processed/geolife_20k.pkl \
  --max-step-m 300
```

The released configurations apply a fixed trajectory-level split with seed `20260906`. Chengdu and GeoLife use a 300 m consecutive-displacement quality-control threshold in the reported final runs. Check each provider's terms before downloading, processing, or sharing data.
