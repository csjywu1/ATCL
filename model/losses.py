"""Training objectives for ACTP."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from data.dataset import HORIZON
from model.actp import SCALES
from model.geometry import poincare_distance


def negative_views(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    detour, switch = x.clone(), x.clone()
    length = x.shape[1]
    for i in range(x.shape[0]):
        valid = int(mask[i].sum())
        if valid < 6:
            continue
        start = length - valid
        left = start + valid // 3
        right = start + max(valid // 3 + 1, 2 * valid // 3)
        detour[i, left:right, :2] += x.new_tensor([35.0, -25.0])
        switch[i, start:start + valid, :2] = torch.flip(switch[i, start:start + valid, :2], dims=[0])
    return detour, switch


def occupancy_targets(future: torch.Tensor, bins: int, cell: float) -> torch.Tensor:
    indices = torch.floor(future / cell + bins / 2).long().clamp(0, bins - 1)
    return indices[..., 0] * bins + indices[..., 1]


def actp_loss(model, output, negative_ball, future, config):
    pred = output["pred"]
    prediction = F.smooth_l1_loss(pred, future, beta=float(config["smoothl1_beta"]))

    occupancy = pred.new_zeros(())
    for logits, scale in zip(output["occ"], SCALES):
        target = occupancy_targets(future, model.bins, scale).reshape(-1)
        occupancy += F.cross_entropy(logits.reshape(-1, model.bins * model.bins), target)
    occupancy /= len(SCALES)

    contrast = pred.new_zeros(())
    for local, other in ((0, 1), (1, 2), (0, 2)):
        positive = -poincare_distance(output["z"][local], output["z"][other]) / config["temperature"]
        candidates = [positive]
        for negative in negative_ball:
            candidates.append(-poincare_distance(output["z"][other], negative[local]) / config["temperature"])
        contrast += (-torch.log_softmax(torch.stack(candidates), dim=0)[0]).mean()
    contrast /= 3.0

    origin = torch.zeros_like(output["z"][0])
    radii = [poincare_distance(origin, state) for state in output["z"]]
    order = F.relu(config["margin"] + radii[2] - radii[1]).mean()
    order += F.relu(config["margin"] + radii[1] - radii[0]).mean()

    total = prediction + config["lambda_occ"] * occupancy
    total += config["lambda_hyp"] * contrast + config["lambda_ord"] * order
    return total, {"prediction": prediction, "occupancy": occupancy, "contrast": contrast, "order": order}
