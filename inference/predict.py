"""Predict five future offsets for one latitude/longitude trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.dataset import ForecastDataset
from model import ACTP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True, help="JSON array of [latitude, longitude] points")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/actp_geolife_seed3.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/geolife.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    points = np.asarray(json.loads(args.trajectory.read_text(encoding="utf-8")), dtype=np.float64)
    if len(points) < 6:
        raise ValueError("trajectory must contain at least six points")
    # Append five copies only to reuse the public prefix formatter; they are never passed to the model.
    padded = np.concatenate([points, np.repeat(points[-1:], 5, axis=0)], axis=0)
    sample = ForecastDataset([padded], int(config["max_history"]))[0]
    device = torch.device("cuda" if torch.cuda.is_available() and not config.get("cpu", False) else "cpu")
    model = ACTP(int(config["hidden"]), int(config["bins"]), bool(config["raw_context"])).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    model.eval()
    with torch.no_grad():
        prediction = model(sample["x"].unsqueeze(0).to(device), sample["mask"].unsqueeze(0).to(device))["pred"][0]
    print(json.dumps({"future_offsets_m": prediction.cpu().tolist()}, indent=2))


if __name__ == "__main__":
    main()
