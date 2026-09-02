"""Train ACTP from a JSON configuration."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import ForecastDataset, load_geolife, split_trajectories
from model import ACTP
from train.engine import evaluate, train_epoch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/geolife.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))

    trajectories = load_geolife(config["data_path"], config.get("limit"), config.get("min_points", 36))
    train_rows, val_rows, test_rows = split_trajectories(trajectories, int(config["split_seed"]))
    if config.get("refit_trainval", False):
        train_rows = train_rows + val_rows

    datasets = {
        "train": ForecastDataset(train_rows, int(config["max_history"])),
        "val": ForecastDataset(val_rows, int(config["max_history"])),
        "test": ForecastDataset(test_rows, int(config["max_history"])),
    }
    loaders = {
        name: DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=name == "train", num_workers=0)
        for name, dataset in datasets.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() and not config.get("cpu", False) else "cpu")
    model = ACTP(int(config["hidden"]), int(config["bins"]), bool(config["raw_context"])).to(device)
    resume = config.get("resume")
    if resume:
        model.load_state_dict(torch.load(resume, map_location=device), strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"actp_seed{config['seed']}.pt"
    history, best = [], float("inf")
    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = train_epoch(model, loaders["train"], optimizer, device, config)
        validation = evaluate(model, loaders["val"], device)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["rmse_m"] < best:
            best = validation["rmse_m"]
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    report = {
        "config": config,
        "counts": {name: len(dataset) for name, dataset in datasets.items()},
        "device": str(device),
        "test": evaluate(model, loaders["test"], device),
        "history": history,
        "checkpoint": str(checkpoint),
    }
    report_path = output / f"actp_seed{config['seed']}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
