# Data

- `raw/Geolife_Trajectories_1.3.zip`: original GeoLife archive. Check the provider's redistribution terms before publishing it.
- `processed/geolife_20k.pkl`: 20,000 contiguous trajectories used by the released experiment.
- `prepare.py`: converts the archive to the processed pickle and removes discontinuities larger than 2 km.

Rebuild the processed file from the project root:

```bash
python -m data.prepare --input data/raw/Geolife_Trajectories_1.3.zip --output data/processed/geolife_20k.pkl
```
