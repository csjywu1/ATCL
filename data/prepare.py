"""Prepare a supported raw trajectory dataset as a portable pickle."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from data.dataset import read_chengdu, read_geolife, read_worldtrace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("worldtrace", "chengdu", "geolife"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20_000)
    parser.add_argument("--min-points", type=int, default=36)
    parser.add_argument("--worldtrace-step", type=int, default=3)
    parser.add_argument("--max-step-m", type=float, default=2_000.0)
    args = parser.parse_args()

    if args.dataset == "worldtrace":
        trajectories = read_worldtrace(args.input, args.limit, args.worldtrace_step)
    elif args.dataset == "chengdu":
        trajectories = read_chengdu(args.input, args.limit, args.min_points, args.max_step_m)
    else:
        trajectories = read_geolife(args.input, args.limit, args.min_points, args.max_step_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pickle.dumps(trajectories, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"saved {len(trajectories)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
