"""Predict five future metric offsets for one observed trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from actp import runner as core
from inference.evaluate import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    points = np.asarray(json.loads(args.trajectory.read_text(encoding="utf-8")), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
        parser.error("trajectory must contain at least two [latitude, longitude] points")
    config = json.loads(args.report.read_text(encoding="utf-8"))["config"]
    history = core.local_xy(points[:, :2])
    history -= history[-1]
    history = history[-config["max_history"]:]
    x = np.zeros((1, config["max_history"], 3), dtype=np.float32)
    mask = np.zeros((1, config["max_history"]), dtype=np.bool_)
    start = config["max_history"] - len(history)
    x[0, start:, :2] = history
    x[0, start:, 2] = np.arange(-len(history) + 1, 1, dtype=np.float32)
    mask[0, start:] = True
    anchor = torch.tensor(points[-1, :2], dtype=torch.float32).unsqueeze(0)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device),
                           anchor=anchor.to(device))["pred"][0].cpu().tolist()
    print(json.dumps({"unit": "meters from final observation", "prediction_xy": prediction}, indent=2))


if __name__ == "__main__":
    main()
