"""Evaluate a released ACTP checkpoint on the fixed GeoLife split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import ForecastDataset, load_geolife, split_trajectories
from model import ACTP
from train.engine import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/geolife.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/actp_geolife_seed3.pt"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    trajectories = load_geolife(config["data_path"], config.get("limit"), config.get("min_points", 36))
    _, _, test_rows = split_trajectories(trajectories, int(config["split_seed"]))
    dataset = ForecastDataset(test_rows, int(config["max_history"]))
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.get("cpu", False) else "cpu")
    model = ACTP(int(config["hidden"]), int(config["bins"]), bool(config["raw_context"])).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    print(json.dumps(evaluate(model, loader, device), indent=2))


if __name__ == "__main__":
    main()
