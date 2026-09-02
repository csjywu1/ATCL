"""Poincaré-ball operations used by HCSC."""

from __future__ import annotations

import math

import torch


def expmap0(u: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    root = math.sqrt(curvature)
    return torch.tanh(root * norm) * u / (root * norm)


def mobius_add(x: torch.Tensor, y: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    xx = (x * x).sum(-1, keepdim=True)
    yy = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    numerator = (1 + 2 * curvature * xy + curvature * yy) * x + (1 - curvature * xx) * y
    denominator = 1 + 2 * curvature * xy + curvature**2 * xx * yy
    return numerator / denominator.clamp_min(1e-8)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    delta = mobius_add(-x, y, curvature)
    norm = (math.sqrt(curvature) * delta.norm(dim=-1)).clamp(max=1 - 1e-5)
    return 2.0 / math.sqrt(curvature) * torch.atanh(norm)
