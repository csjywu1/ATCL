"""Adaptive Contrastive Learning for Trajectory Prediction."""

from __future__ import annotations

import torch
from torch import nn

from data.dataset import HORIZON
from model.geometry import expmap0

SCALES = (50.0, 150.0, 500.0)


class ScaleEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.gru = nn.GRU(3, hidden, batch_first=True)
        self.pool = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU())

    def forward(self, x: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
        values = x.clone()
        values[..., :2] /= scale
        values[..., 2] /= 32.0
        lengths = mask.sum(1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(values, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])
        valid = mask.unsqueeze(-1)
        mean = (output * valid).sum(1) / lengths.to(output.device).unsqueeze(-1)
        maximum = output.masked_fill(~valid, -1e4).max(1).values
        return self.pool(torch.cat([mean, maximum], dim=-1))


class ACTP(nn.Module):
    """Released GeoLife architecture compatible with the published checkpoint."""

    def __init__(self, hidden: int = 2048, bins: int = 21, raw_context: bool = True):
        super().__init__()
        self.hidden, self.bins, self.raw_context = hidden, bins, raw_context
        self.scales = SCALES
        self.encoders = nn.ModuleList([ScaleEncoder(hidden) for _ in SCALES])
        self.scale_code = nn.Parameter(torch.zeros(3, hidden))
        self.to_ball = nn.ModuleList([nn.Linear(hidden, hidden) for _ in SCALES])
        self.occ_heads = nn.ModuleList([nn.Linear(hidden, HORIZON * bins * bins) for _ in SCALES])
        self.score = nn.Linear(hidden, 1)
        self.endpoint_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, HORIZON * 2))
        if raw_context:
            self.raw_proj = nn.Sequential(nn.Linear(32 * 3, hidden), nn.LayerNorm(hidden), nn.GELU())

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        states = [encoder(x, mask, scale) + self.scale_code[i]
                  for i, (encoder, scale) in enumerate(zip(self.encoders, self.scales))]
        ball = [0.90 * expmap0(layer(state)) for layer, state in zip(self.to_ball, states)]
        return states, ball

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        states, ball = self.encode(x, mask)
        logits = torch.cat([self.score(state) for state in states], dim=-1)
        weights = torch.softmax(logits, dim=-1)
        fused = sum(weights[:, i:i + 1] * states[i] for i in range(3))
        if self.raw_context:
            raw = x.clone()
            raw[..., :2] = torch.sign(raw[..., :2]) * torch.log1p(raw[..., :2].abs() / 100.0)
            raw[..., 2] /= 32.0
            fused = fused + self.raw_proj(raw.reshape(raw.shape[0], -1))
        prediction = self.decoder(fused).view(-1, HORIZON, 2)
        occupancy = [head(state).view(-1, HORIZON, self.bins * self.bins)
                     for head, state in zip(self.occ_heads, states)]
        return {
            "h": states,
            "z": ball,
            "weights": weights,
            "pred": prediction,
            "endpoint": self.endpoint_head(fused),
            "occ": occupancy,
        }
