"""Prepare the GeoLife archive for ACTP."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from data.dataset import load_geolife


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20_000)
    parser.add_argument("--min-points", type=int, default=36)
    args = parser.parse_args()

    trajectories = load_geolife(args.input, args.limit, args.min_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pickle.dumps(trajectories, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"saved {len(trajectories)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
