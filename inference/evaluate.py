"""Evaluate frozen ACTP checkpoints without training or model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from actp import runner as core


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(config: dict, road_payload=None):
    if config["method"] == "gru":
        return core.GRUPredictor(config["hidden"])
    return core.ACTPPredictor(
        config["hidden"], config["bins"], config["delta_decode"],
        config["autoregressive_decode"], config["encoder_layers"],
        config["velocity_context"], config["activation"], config["decoder_type"],
        config["pooling"], config["velocity_residual"], config["endpoint_blend"],
        config["kinematic_gate"], config["velocity_input"], config["input_transform"],
        config["kinematic_mixture"], config["raw_context"], config["absolute_context"],
        road_payload["road_features"] if road_payload else None,
        config.get("velocity_gap", 1), config.get("fusion_mode", "transition"),
    )


def evaluate_one(report_path: Path, checkpoint: Path | None, data_path: Path | None,
                 device: torch.device) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = report["config"]
    source = data_path or Path(config["path"])
    if config["dataset"] == "worldtrace":
        trajectories = core.read_worldtrace(source, config["limit"], config["worldtrace_step"])
    elif config["dataset"] == "geolife":
        trajectories = core.read_geolife(source, config["limit"], max_step_m=config["max_step_m"])
    else:
        trajectories = core.read_chengdu(source, config["limit"],
                                         max_step_m=config.get("chengdu_max_step_m", 0.0))
    _, _, test_raw = core.split_trajectories(trajectories, config["split_seed"])
    road_path = config.get("road_cache")
    road_payload = pickle.loads(Path(road_path).read_bytes()) if road_path else None
    road_lookup = road_payload["lookup"] if road_payload else None
    dataset = core.ForecastDataset(test_raw, config["max_history"],
                                   history_stride=config["history_stride"], road_lookup=road_lookup)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    model = build_model(config, road_payload)
    checkpoint = checkpoint or Path(str(report_path) + ".pt")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device)
    return {
        "dataset": config["dataset"],
        "seed": config["seed"],
        "split_seed": config["split_seed"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "selected_validation": report["best_validation"],
        "best_epoch": report["best_epoch"],
        "test": core.evaluate(model, loader, device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    result = evaluate_one(args.report, args.checkpoint, args.data_path, device)
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
