"""Reusable training and evaluation loops."""

from __future__ import annotations

import math

import torch

from model.losses import actp_loss, negative_views


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    absolute, squared, count = 0.0, 0.0, 0
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        future = batch["future"].to(device)
        error = model(x, mask)["pred"] - future
        absolute += error.abs().sum().item()
        squared += error.square().sum().item()
        count += error.numel()
    return {"mae_m": absolute / count, "rmse_m": math.sqrt(squared / count), "coordinates": count}


def train_epoch(model, loader, optimizer, device: torch.device, config: dict) -> dict[str, float]:
    model.train()
    totals = {name: 0.0 for name in ("total", "prediction", "occupancy", "contrast", "order")}
    examples = 0
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        future = batch["future"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(x, mask)
        detour, switch = negative_views(x, mask)
        negative_ball = [model(detour, mask)["z"], model(switch, mask)["z"]]
        loss, parts = actp_loss(model, output, negative_ball, future, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()
        batch_size = x.shape[0]
        totals["total"] += float(loss.detach()) * batch_size
        for name, value in parts.items():
            totals[name] += float(value.detach()) * batch_size
        examples += batch_size
    return {name: value / max(examples, 1) for name, value in totals.items()}
