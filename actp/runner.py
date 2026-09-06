#!/usr/bin/env python3
"""A small, reproducible ACTP trajectory-prediction implementation.

The benchmark follows UniTraj's forward-prediction target: the final five
recorded GPS points are targets and the preceding history is the input.  This
script is intentionally self-contained so that the experiment can be audited
on the public WorldTrace sample, the GeoLife archive, and the coordinate-level
Chengdu release.  It implements three scale encoders, occupancy supervision,
Poincare cross-scale contrast, radial ordering, and attention fusion.

It is an experiment runner, not a replacement for the paper's formal model.
The source paths, split seed, and all metrics are written to JSON together
with the checkpoint configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import pickle
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


HORIZON = 5
SCALES = (50.0, 150.0, 500.0)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def local_xy(points: np.ndarray) -> np.ndarray:
    """Convert [lat, lon] points to meters relative to the first point."""
    points = np.asarray(points, dtype=np.float64)
    lat0 = float(points[0, 0])
    lon0 = float(points[0, 1])
    x = (points[:, 1] - lon0) * 111320.0 * math.cos(math.radians(lat0))
    y = (points[:, 0] - lat0) * 110540.0
    return np.stack([x, y], axis=1)


def read_worldtrace(path: Path, limit: int | None, sample_step: int = 3) -> list[np.ndarray]:
    """Read WorldTrace and apply the three-second evaluation sampling."""
    sample_step = max(1, int(sample_step))
    if path.suffix.lower() == ".zip":
        out = []
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                pts = []
                try:
                    with zf.open(name) as raw:
                        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="ignore"))
                        for row in reader:
                            try:
                                pts.append((float(row["latitude"]), float(row["longitude"])))
                            except (KeyError, TypeError, ValueError):
                                continue
                except (OSError, zipfile.BadZipFile):
                    continue
                sampled = np.asarray(pts, dtype=np.float64)[::sample_step]
                if len(sampled) >= 36:
                    out.append(sampled)
                    if limit and len(out) >= limit:
                        break
        return out
    obj = pickle.loads(path.read_bytes())
    rows = obj["trajectory"].tolist() if hasattr(obj, "columns") else obj
    out = []
    for row in rows:
        a = np.asarray(row, dtype=np.float64)
        if a.ndim == 2 and a.shape[1] >= 2:
            sampled = a[::sample_step, :2]
            if len(sampled) < 36:
                continue
            out.append(sampled)
            if limit and len(out) >= limit:
                break
    return out


def read_geolife(path: Path, limit: int | None, min_points: int = 36,
                 max_step_m: float = 2000.0) -> list[np.ndarray]:
    out = []

    if path.suffix.lower() in {".pkl", ".pickle"}:
        rows = pickle.loads(path.read_bytes())
        for row in rows:
            arr = np.asarray(row, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            arr = arr[:, :2]
            if len(arr) < min_points:
                continue
            lat0 = arr[0, 0]
            xx = (arr[:, 1] - arr[0, 1]) * 111320.0 * np.cos(np.deg2rad(lat0))
            yy = (arr[:, 0] - arr[0, 0]) * 110540.0
            cuts = np.flatnonzero(np.sqrt(np.diff(xx) ** 2 + np.diff(yy) ** 2) > max_step_m) + 1
            start = 0
            for end in np.r_[cuts, len(arr)]:
                if end - start >= min_points:
                    out.append(arr[start:end])
                    if limit and len(out) >= limit:
                        return out
                start = int(end)
        return out

    def emit_segments(points):
        if len(points) < min_points:
            return
        arr = np.asarray(points, dtype=np.float64)
        # A PLT file can contain disconnected activities.  Keep only
        # contiguous portions; otherwise a single discontinuity can create a
        # multi-thousand-kilometre training target.
        lat0 = arr[:, 0]
        xx = (arr[:, 1] - arr[:, 1][0]) * 111320.0 * np.cos(np.deg2rad(lat0[0]))
        yy = (arr[:, 0] - arr[:, 0][0]) * 110540.0
        jumps = np.sqrt(np.diff(xx) ** 2 + np.diff(yy) ** 2)
        cuts = np.flatnonzero(jumps > max_step_m) + 1
        start = 0
        for end in np.r_[cuts, len(arr)]:
            if end - start >= min_points:
                out.append(arr[start:end])
            start = int(end)

    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".plt"):
                continue
            pts = []
            try:
                with zf.open(name) as raw:
                    for line in io.TextIOWrapper(raw, encoding="utf-8", errors="ignore"):
                        parts = line.strip().split(",")
                        if len(parts) >= 2:
                            pts.append((float(parts[0]), float(parts[1])))
            except (OSError, ValueError, UnicodeError):
                continue
            emit_segments(pts)
            if limit and len(out) >= limit:
                break
    return out


def read_chengdu(path: Path, limit: int | None, min_points: int = 36,
                 max_step_m: float = 0.0) -> list[np.ndarray]:
    try:
        import pandas as pd
        frame = pd.read_pickle(path)
    except Exception:
        frame = None
    out = []
    if frame is not None and hasattr(frame, "columns"):
        if "lat_list" not in frame.columns or "lng_list" not in frame.columns:
            raise ValueError("Chengdu pickle must contain lat_list and lng_list")
        rows = zip(frame["lat_list"], frame["lng_list"])
    else:
        # Portable coordinate-only pickles are used on machines whose pandas
        # version cannot unpickle the original DataFrame representation.
        rows = ((np.asarray(p)[:, 0], np.asarray(p)[:, 1]) for p in pickle.loads(path.read_bytes()))
    for lat, lon in rows:
        n = min(len(lat), len(lon))
        if n >= min_points:
            arr = np.stack([np.asarray(lat[:n], dtype=np.float64),
                            np.asarray(lon[:n], dtype=np.float64)], axis=1)
            if max_step_m > 0:
                lat0 = float(arr[0, 0])
                dx = np.diff(arr[:, 1]) * 111320.0 * np.cos(np.deg2rad(lat0))
                dy = np.diff(arr[:, 0]) * 110540.0
                if np.sqrt(dx * dx + dy * dy).max(initial=0.0) > max_step_m:
                    continue
            out.append(arr)
            if limit and len(out) >= limit:
                break
    return out


class ForecastDataset(Dataset):
    def __init__(self, trajectories: list[np.ndarray], max_history: int = 64,
                 augment_windows: int = 0, rotate: bool = False, history_stride: int = 1,
                 road_lookup=None):
        self.items = []
        self.max_history = max_history
        self.rotate = rotate
        self.history_stride = max(1, int(history_stride))
        for raw in trajectories:
            if len(raw) <= HORIZON:
                continue
            xy = local_xy(raw)
            # The final five recorded points form the evaluation target.  For
            # training, optional earlier windows expose the same trajectory
            # under different cut points and improve coverage of rare turns.
            end = len(xy) - HORIZON
            ends = [end]
            if augment_windows > 0 and end > 8:
                # Keep auxiliary windows close to the evaluated cut point.
                # Very early cuts have a different prediction distribution
                # and can dilute the five-step continuation objective.
                extra = [end - j for j in range(1, augment_windows + 1)]
                ends.extend(e for e in extra if e >= 8 and e < end)
            for cut in ends:
                hist, fut = xy[:cut], xy[cut:cut + HORIZON]
                hist = hist[-max_history * self.history_stride::self.history_stride]
                # Re-center on the last observed point, which is the decoder start.
                origin = hist[-1].copy()
                hist = hist - origin
                fut = fut - origin
                dt = np.arange(-len(hist) + 1, 1, dtype=np.float32)
                anchor = np.asarray(raw[cut - 1, :2], dtype=np.float32)
                road_ids = None
                if road_lookup is not None and cut == end and self.history_stride == 1:
                    key = hashlib.sha256(np.asarray(raw[:, :2], dtype=np.float64).tobytes()).hexdigest()
                    road_ids = road_lookup.get(key)
                self.items.append((hist.astype(np.float32), fut.astype(np.float32), dt, anchor, road_ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        hist, fut, dt, anchor, road_ids = self.items[i]
        if self.rotate:
            angle = np.random.uniform(-math.pi, math.pi)
            co, si = math.cos(angle), math.sin(angle)
            R = np.asarray([[co, -si], [si, co]], dtype=np.float32)
            hist = hist @ R.T
            fut = fut @ R.T
        x = np.zeros((self.max_history, 3), dtype=np.float32)
        m = np.zeros(self.max_history, dtype=np.bool_)
        start = self.max_history - len(hist)
        x[start:, :2] = hist
        x[start:, 2] = dt
        m[start:] = True
        road = np.zeros(self.max_history, dtype=np.int64)
        if road_ids is not None:
            road[-min(len(road_ids), self.max_history):] = road_ids[-self.max_history:]
        return {"x": torch.from_numpy(x), "mask": torch.from_numpy(m),
                "future": torch.from_numpy(fut), "anchor": torch.from_numpy(anchor),
                "road": torch.from_numpy(road)}


def split_trajectories(trajectories: list[np.ndarray], seed: int) -> tuple[list, list, list]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(trajectories))
    n = len(order)
    a, b = int(0.8 * n), int(0.9 * n)
    return ([trajectories[i] for i in order[:a]],
            [trajectories[i] for i in order[a:b]],
            [trajectories[i] for i in order[b:]])


def make_activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    return nn.GELU()


class ScaleEncoder(nn.Module):
    def __init__(self, hidden: int, layers: int = 1, activation: str = "gelu",
                 pooling: str = "meanmax", velocity_input: bool = False,
                 input_transform: str = "none"):
        super().__init__()
        self.layers = layers
        self.pooling = pooling
        self.velocity_input = velocity_input
        self.input_transform = input_transform
        input_dim = 5 if velocity_input else 3
        self.gru = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=0.1 if layers > 1 else 0.0)
        # A parent scale conditions both every child input and its initial
        # recurrent state, matching the coarse-to-fine computation in Eq. 9.
        self.parent_input = nn.Linear(hidden, input_dim, bias=False)
        self.parent_state = nn.Linear(hidden, hidden, bias=False)
        self.parent_gate = nn.Parameter(torch.tensor(-3.0))
        if pooling == "attention":
            self.attn = nn.Linear(hidden, 1)
            self.pool = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden),
                                      make_activation(activation))
        else:
            self.pool = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden),
                                      make_activation(activation))

    def forward(self, x, mask, scale, parent=None):
        y = x.clone()
        y[..., :2] = y[..., :2] / scale
        if self.input_transform == "log":
            y[..., :2] = torch.sign(y[..., :2]) * torch.log1p(y[..., :2].abs())
        # Time is scaled mildly so that very long histories do not dominate.
        y[..., 2] = y[..., 2] / 32.0
        if self.velocity_input:
            dv = torch.zeros_like(y[..., :2])
            dv[:, 1:] = y[:, 1:, :2] - y[:, :-1, :2]
            y = torch.cat([y[..., :2], dv, y[..., 2:3]], dim=-1)
        initial = None
        if parent is not None:
            gate = torch.sigmoid(self.parent_gate)
            y = y + gate * self.parent_input(parent).unsqueeze(1)
            initial = (gate * self.parent_state(parent)).unsqueeze(0).expand(self.layers, -1, -1).contiguous()
        lengths = mask.sum(1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed, initial)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        mf = mask.unsqueeze(-1)
        if self.pooling == "attention":
            logits = self.attn(out).squeeze(-1).masked_fill(~mask, -1e4)
            alpha = torch.softmax(logits, dim=1).unsqueeze(-1)
            weighted = (out * alpha).sum(1)
            last = out[torch.arange(out.shape[0], device=out.device), lengths.to(out.device) - 1]
            return self.pool(torch.cat([weighted, last], dim=-1)), out
        mean = (out * mf).sum(1) / lengths.to(out.device).unsqueeze(-1)
        mx = out.masked_fill(~mf, -1e4).max(1).values
        return self.pool(torch.cat([mean, mx], dim=-1)), out


class ACTPPredictor(nn.Module):
    def __init__(self, hidden: int = 128, bins: int = 21, delta_decode: bool = False,
                 autoregressive_decode: bool = False, encoder_layers: int = 1,
                 velocity_context: bool = False, activation: str = "gelu",
                 decoder_type: str = "direct", pooling: str = "meanmax",
                 velocity_residual: bool = False, endpoint_blend: float = 0.0,
                 kinematic_gate: bool = False, velocity_input: bool = False,
                 input_transform: str = "none", kinematic_mixture: bool = False,
                 raw_context: bool = False, absolute_context: bool = False,
                 road_features=None, velocity_gap: int = 1,
                 fusion_mode: str = "transition"):
        super().__init__()
        self.hidden = hidden
        self.bins = bins
        self.delta_decode = delta_decode
        self.autoregressive_decode = autoregressive_decode
        self.velocity_context = velocity_context
        self.activation = activation
        self.decoder_type = decoder_type
        self.pooling = pooling
        self.velocity_residual = velocity_residual
        self.velocity_gap = max(1, int(velocity_gap))
        self.endpoint_blend = endpoint_blend
        self.kinematic_gate = kinematic_gate
        self.velocity_input = velocity_input
        self.input_transform = input_transform
        self.kinematic_mixture = kinematic_mixture
        self.raw_context = raw_context
        self.absolute_context = absolute_context
        self.has_road_context = road_features is not None
        self.fusion_mode = fusion_mode
        self.rho = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        if raw_context:
            self.raw_proj = nn.Sequential(nn.Linear(32 * 3, hidden), nn.LayerNorm(hidden),
                                          make_activation(activation))
        if absolute_context:
            self.register_buffer("geo_scales", torch.tensor([.001, .003, .01, .03, .1, 1., 10.]))
            self.geo_proj = nn.Sequential(nn.Linear(30, hidden), nn.LayerNorm(hidden),
                                          make_activation(activation))
        if road_features is not None:
            road_tensor = torch.as_tensor(road_features, dtype=torch.float32)
            self.register_buffer("road_features", road_tensor)
            self.road_proj = nn.Sequential(nn.Linear(road_tensor.shape[1], hidden), nn.LayerNorm(hidden),
                                           make_activation(activation))
            self.road_gru = nn.GRU(hidden, hidden, batch_first=True)
            self.road_gate = nn.Parameter(torch.tensor(-3.0))
        self.kinematic_expert = False
        self.scales = SCALES
        self.encoders = nn.ModuleList([ScaleEncoder(hidden, encoder_layers, activation, pooling, velocity_input, input_transform) for _ in self.scales])
        self.scale_code = nn.Parameter(torch.zeros(3, hidden))
        self.to_ball = nn.ModuleList([nn.Linear(hidden, hidden) for _ in self.scales])
        self.occ_heads = nn.ModuleList([nn.Linear(hidden, HORIZON * bins * bins) for _ in self.scales])
        self.score = nn.Linear(hidden, 1)
        self.hyper_query = nn.Linear(hidden, hidden, bias=False)
        self.hyper_key = nn.Linear(hidden, hidden, bias=False)
        self.hyper_value = nn.Linear(hidden, hidden, bias=False)
        self.hyper_gate = nn.Parameter(torch.tensor(-3.0))
        self.hierarchy_gate = nn.Parameter(torch.tensor(-3.0))
        self.endpoint_head = nn.Sequential(nn.Linear(hidden, hidden), make_activation(activation),
                                           nn.Linear(hidden, 2))
        if kinematic_gate:
            self.kinematic_gate_net = nn.Sequential(
                nn.Linear(hidden + 2, max(32, hidden // 4)), make_activation(activation),
                nn.Linear(max(32, hidden // 4), 1))
            nn.init.constant_(self.kinematic_gate_net[-1].bias, -2.0)
        if kinematic_mixture:
            self.kinematic_mix = nn.Sequential(
                nn.Linear(hidden + 8, max(32, hidden // 4)), make_activation(activation),
                nn.Linear(max(32, hidden // 4), 5))
            with torch.no_grad():
                self.kinematic_mix[-1].bias.zero_()
                self.kinematic_mix[-1].bias[0] = 2.0
        if decoder_type == "horizon":
            self.horizon_code = nn.Parameter(torch.zeros(HORIZON, hidden))
            self.decoder = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden),
                                         make_activation(activation), nn.Linear(hidden, 2))
        else:
            self.decoder = nn.Sequential(nn.Linear(hidden, hidden), make_activation(activation),
                                         nn.Linear(hidden, HORIZON * 2))
        if velocity_context:
            self.velocity_proj = nn.Sequential(nn.Linear(2, hidden), nn.LayerNorm(hidden),
                                               make_activation(activation))
        if autoregressive_decode:
            self.step_code = nn.Parameter(torch.zeros(HORIZON, hidden))
            self.ar_input = nn.Linear(hidden + 2, hidden)
            self.ar_cell = nn.GRUCell(hidden, hidden)
            self.ar_head = nn.Sequential(nn.Linear(hidden, hidden), make_activation(activation),
                                         nn.Linear(hidden, 2))

    def curvature(self):
        return F.softplus(self.rho).clamp(0.02, 5.0)

    def encode(self, x, mask, anchor=None, road_ids=None):
        # Scale order in storage is local/neighbourhood/regional.  Encoding is
        # deliberately reversed so each finer state receives its parent.
        h, sequences, parent = [None] * 3, [None] * 3, None
        for i in (2, 1, 0):
            state, seq = self.encoders[i](x, mask, self.scales[i], parent)
            if i == 2 and self.absolute_context:
                if anchor is None:
                    raise ValueError("absolute_context requires the last observed latitude/longitude")
                phase = anchor.unsqueeze(-1) / self.geo_scales
                geo = torch.cat([anchor / anchor.new_tensor([90., 180.]),
                                 torch.sin(phase).flatten(1), torch.cos(phase).flatten(1)], dim=-1)
                state = state + self.geo_proj(geo)
            if i == 2 and self.has_road_context:
                if road_ids is None:
                    raise ValueError("road context requires matched road ids")
                road_seq = self.road_proj(self.road_features[road_ids])
                _, road_state = self.road_gru(road_seq)
                state = state + torch.sigmoid(self.road_gate) * road_state[-1]
            h[i] = state + self.scale_code[i]
            sequences[i] = seq
            parent = h[i]
        u = [layer(v) for layer, v in zip(self.to_ball, h)]
        # Keep every point safely inside the Poincare ball.  Without this
        # factor a large norm can land numerically on the boundary, where the
        # distance is clipped and the contrastive gradient vanishes.
        z = [0.90 * expmap0(v, self.curvature()) for v in u]
        return h, z

    def hyperbolic_scale_context(self, h, z):
        """On-manifold attention followed by an Einstein midpoint."""
        c = self.curvature()
        q = self.hyper_query(h[0])
        keys = torch.stack([self.hyper_key(v) for v in h], dim=1)
        weights = torch.softmax((q.unsqueeze(1) * keys).sum(-1) / math.sqrt(self.hidden), dim=1)
        values = [0.90 * expmap0(self.hyper_value(v), c) for v in h]
        klein = torch.stack([2 * v / (1 + c * (v * v).sum(-1, keepdim=True)) for v in values], dim=1)
        gamma = (1 - c * (klein * klein).sum(-1)).clamp_min(1e-6).rsqrt()
        wg = weights * gamma
        midpoint = (wg.unsqueeze(-1) * klein).sum(1) / wg.sum(1, keepdim=True).clamp_min(1e-6)
        ball = midpoint / (1 + (1 - c * (midpoint * midpoint).sum(-1, keepdim=True)).clamp_min(1e-6).sqrt())
        return logmap0(ball, c), weights

    def forward(self, x, mask, future=None, teacher_forcing: float = 0.0, anchor=None, road_ids=None):
        h, z = self.encode(x, mask, anchor, road_ids)
        manifold_context, hyper_weights = self.hyperbolic_scale_context(h, z)
        # Decoding starts from the local state, which already contains both
        # parent states.  The on-manifold aggregate is a gated residual.
        flat_scores = torch.cat([self.score(v) for v in h], dim=-1)
        w = torch.softmax(flat_scores, dim=-1)
        flat = sum(w[:, i:i + 1] * h[i] for i in range(3))
        hierarchical = h[0] + torch.sigmoid(self.hyper_gate) * manifold_context
        fused = (hierarchical if self.fusion_mode == "hierarchical" else
                 flat + torch.sigmoid(self.hierarchy_gate) * (hierarchical - flat))
        if self.raw_context:
            raw = x.clone()
            raw[..., :2] = torch.sign(raw[..., :2]) * torch.log1p(raw[..., :2].abs() / 100.0)
            raw[..., 2] = raw[..., 2] / 32.0
            fused = fused + self.raw_proj(raw.reshape(raw.shape[0], -1))
        if self.velocity_context:
            # The final two valid observations provide a local motion trend;
            # this explicit cue helps long-range extrapolation without using
            # any future coordinates.
            velocity = x[:, -1, :2] - x[:, -2, :2]
            velocity_feat = torch.sign(velocity) * torch.log1p(velocity.abs() / 50.0)
            fused = fused + self.velocity_proj(velocity_feat)
        if self.autoregressive_decode:
            state = fused
            cur = fused.new_zeros((fused.shape[0], 2))
            preds = []
            for t in range(HORIZON):
                decoder_input = torch.cat([cur, self.step_code[t].expand(fused.shape[0], -1)], dim=-1)
                state = self.ar_cell(self.ar_input(decoder_input), state)
                cur = cur + self.ar_head(state)
                preds.append(cur)
                if (future is not None and self.training and teacher_forcing > 0 and
                        (teacher_forcing >= 1.0 or torch.rand((), device=x.device) < teacher_forcing)):
                    cur = future[:, t]
            pred = torch.stack(preds, dim=1)
        else:
            if self.decoder_type == "horizon":
                fc = fused.unsqueeze(1).expand(-1, HORIZON, -1)
                hc = self.horizon_code.unsqueeze(0).expand(fused.shape[0], -1, -1)
                pred = self.decoder(torch.cat([fc, hc], dim=-1))
            else:
                pred = self.decoder(fused).view(-1, HORIZON, 2)
            if self.delta_decode:
                pred = torch.cumsum(pred, dim=1)
            if self.velocity_residual:
                gap = min(self.velocity_gap, x.shape[1] - 1)
                velocity = (x[:, -1, :2] - x[:, -1-gap, :2]) / float(gap)
                steps = torch.arange(1, HORIZON + 1, device=x.device, dtype=x.dtype).view(1, HORIZON, 1)
                pred = pred + steps * velocity.unsqueeze(1)
        if self.kinematic_gate:
            # Blend the learned decoder with a bounded constant-velocity
            # extrapolation.  The gate is learned from the observed context,
            # so long-range windows can use the kinematic branch without
            # forcing it on short, irregular movements.
            velocity = x[:, -1, :2] - x[:, -2, :2]
            velocity = velocity.clamp(-500.0, 500.0)
            velocity_feat = torch.sign(velocity) * torch.log1p(velocity.abs() / 50.0)
            gate = torch.sigmoid(self.kinematic_gate_net(torch.cat([fused, velocity_feat], dim=-1)))
            steps = torch.arange(1, HORIZON + 1, device=x.device, dtype=x.dtype).view(1, HORIZON, 1)
            base = steps * velocity.unsqueeze(1)
            pred = (1.0 - gate.unsqueeze(1)) * pred + gate.unsqueeze(1) * base
        if self.kinematic_mixture:
            velocities = [(x[:, -1, :2] - x[:, -1-k, :2]) / float(k) for k in (1, 3, 5, 8)]
            vf = torch.cat([torch.sign(v) * torch.log1p(v.abs() / 50.0) for v in velocities], dim=-1)
            mix = torch.softmax(self.kinematic_mix(torch.cat([fused, vf], dim=-1)), dim=-1)
            bases = [pred]
            steps = torch.arange(1, HORIZON + 1, device=x.device, dtype=x.dtype).view(1, HORIZON, 1)
            for v in velocities:
                bases.append(steps * v.unsqueeze(1))
            pred = sum(mix[:, i:i+1, None] * bases[i] for i in range(5))
        endpoint = self.endpoint_head(fused)
        if self.endpoint_blend > 0:
            pred = pred.clone()
            pred[:, -1] = (1.0 - self.endpoint_blend) * pred[:, -1] + self.endpoint_blend * endpoint
        occ = [head(v).view(-1, HORIZON, self.bins * self.bins) for head, v in zip(self.occ_heads, h)]
        return {"h": h, "z": z, "weights": w, "hyper_weights": hyper_weights,
                "pred": pred, "endpoint": endpoint, "occ": occ}


class GRUPredictor(nn.Module):
    """Single-scale recurrent predictor used as a same-protocol baseline."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(3, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, HORIZON * 2))

    def forward(self, x, mask, future=None, teacher_forcing: float = 0.0, anchor=None, road_ids=None):
        lengths = mask.sum(1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, state = self.gru(packed)
        return {"pred": self.head(state[-1]).view(-1, HORIZON, 2)}


def expmap0(u, c=1.0):
    c = torch.as_tensor(c, dtype=u.dtype, device=u.device)
    root = c.sqrt()
    norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.tanh(root * norm) * u / (root * norm)


def logmap0(z, c=1.0):
    c = torch.as_tensor(c, dtype=z.dtype, device=z.device)
    root = c.sqrt(); norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.atanh((root * norm).clamp(max=1 - 1e-5)) * z / (root * norm)


def mobius_add(x, y, c=1.0):
    xx, yy = (x * x).sum(-1, keepdim=True), (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * yy) * x + (1 - c * xx) * y
    den = 1 + 2 * c * xy + c * c * xx * yy
    return num / den.clamp_min(1e-8)


def poincare_distance(x, y, c=1.0):
    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    q = mobius_add(-x, y, c)
    root = c.sqrt(); norm = (root * q.norm(dim=-1)).clamp(max=1 - 1e-5)
    return 2.0 / root * torch.atanh(norm)


def negative_views(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create detour and switch alternatives without changing future targets."""
    detour, switch = x.clone(), x.clone()
    b, l, _ = x.shape
    for i in range(b):
        valid = int(mask[i].sum())
        if valid < 6:
            continue
        s = max(0, l - valid)
        # Detour: bend a contiguous middle segment away from the observed path.
        a, z = s + valid // 3, s + max(valid // 3 + 1, 2 * valid // 3)
        offset = torch.tensor([35.0, -25.0], device=x.device, dtype=x.dtype)
        detour[i, a:z, :2] = detour[i, a:z, :2] + offset
        # Switch: reverse the observed prefix around its origin; it remains a
        # distinct representation-level alternative while retaining the target.
        switch[i, s:s + valid, :2] = torch.flip(switch[i, s:s + valid, :2], dims=[0])
    return detour, switch


def real_route_negative_indices(x: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select two recorded prefixes with nearby observed start/end locations.

    This is the legal-route fallback when a dataset has no road-transition
    graph.  Every negative is an actually recorded trajectory rather than a
    coordinate perturbation.  Chengdu additionally supplies matched road ids.
    """
    if len(x) < 3:
        ids = torch.arange(len(x), device=x.device)
        return ids.roll(1), ids.roll(2)
    lat = anchor[:, 0]; coslat = torch.cos(torch.deg2rad(lat)).clamp_min(.2)
    end_xy = torch.stack([anchor[:, 1] * 111320. * coslat, lat * 110540.], -1)
    start_xy = end_xy + x[:, 0, :2]
    signature = torch.cat([start_xy, end_xy], -1) / 1000.
    dist = torch.cdist(signature, signature)
    dist.fill_diagonal_(float('inf'))
    near = dist.topk(k=2, largest=False).indices
    return near[:, 0], near[:, 1]


def occupancy_targets(future: torch.Tensor, bins: int, cell: float) -> torch.Tensor:
    # Fixed local grid centered at the last observed point.  Values outside
    # the grid are clipped rather than discarded, preserving one target per
    # future time step.
    idx = torch.floor(future / cell + bins / 2).long()
    idx = idx.clamp(0, bins - 1)
    return idx[..., 0] * bins + idx[..., 1]


def losses(model, out, x, mask, future, args):
    curvature = model.curvature()
    pred = out["pred"]
    horizon_weight = pred.new_tensor(args.horizon_gamma ** np.arange(HORIZON, dtype=np.float32))
    horizon_weight = horizon_weight / horizon_weight.mean()
    horizon_weight = horizon_weight.view(1, HORIZON, 1)
    if args.pred_loss == "smoothl1":
        # A robust coordinate objective is useful for datasets with occasional
        # long GPS jumps; evaluation still uses the unmodified coordinates.
        robust_loss = F.smooth_l1_loss(pred, future, beta=args.smoothl1_beta, reduction="none")
        robust_loss = robust_loss * horizon_weight
        if args.tail_weight > 0:
            # Give rare long-range targets a controlled increase in influence
            # without letting a single extreme jump dominate the batch.
            extent = future.norm(dim=-1).amax(dim=1)
            sample_weight = 1.0 + args.tail_weight * torch.log1p(extent / 250.0)
            sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-6)
            robust_loss = robust_loss.mean(dim=(1, 2)) * sample_weight
            robust_loss = robust_loss.mean()
        else:
            robust_loss = robust_loss.mean()
        # A small squared-error component keeps the robust objective from
        # under-penalising the long jumps that dominate RMSE on GeoLife.
        mse_loss = (pred - future).square() * horizon_weight
        if args.tail_weight > 0:
            mse_loss = mse_loss.mean(dim=(1, 2)) * sample_weight
            mse_loss = mse_loss.mean()
        else:
            mse_loss = mse_loss.mean()
        pred_loss = (1.0 - args.mse_mix) * robust_loss + args.mse_mix * mse_loss
    else:
        pred_loss = F.mse_loss(pred, future)
    if args.endpoint_weight > 0:
        endpoint_loss = F.smooth_l1_loss(out["endpoint"], future[:, -1], beta=args.smoothl1_beta)
        pred_loss = pred_loss + args.endpoint_weight * endpoint_loss
    occ_loss = pred_loss.new_zeros(())
    for logits, scale in zip(out["occ"], SCALES):
        target = occupancy_targets(future, model.bins, scale).reshape(-1)
        occ_loss = occ_loss + F.cross_entropy(logits.reshape(-1, model.bins * model.bins), target)
    occ_loss = occ_loss / len(SCALES)
    pair_loss = pred_loss.new_zeros(())
    # Caller supplies negative outputs through attributes to avoid a second
    # public forward signature.
    if "neg" in out:
        for k, l in ((0, 1), (1, 2), (0, 2)):
            pos = -poincare_distance(out["z"][k], out["z"][l], curvature) / args.temperature
            terms = [pos]
            for nz in out["neg"]:
                terms.append(-poincare_distance(out["z"][l], nz[k], curvature) / args.temperature)
            pair_loss = pair_loss + (-torch.log_softmax(torch.stack(terms, 0), dim=0)[0]).mean()
        pair_loss = pair_loss / 3.0
    else:
        pair_loss = pair_loss / 3.0
    origin = torch.zeros_like(out["z"][0])
    radii = [poincare_distance(origin, z, curvature) for z in out["z"]]
    # Scale order is regional (inner) -> neighborhood -> local (outer).
    # The list order is local, neighborhood, regional.
    order = F.relu(args.margin + radii[2] - radii[1]).mean() + F.relu(args.margin + radii[1] - radii[0]).mean()
    curvature_loss = (torch.log(curvature) - math.log(args.curvature_reference)).square()
    total = (pred_loss + args.lambda_occ * occ_loss + args.lambda_hyp * pair_loss +
             args.lambda_ord * order + args.lambda_curvature * curvature_loss)
    return total, {"pred": pred_loss, "occ": occ_loss, "hyp": pair_loss,
                   "order": order, "curvature": curvature_loss}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    abs_sum = sq_sum = count = 0
    for batch in loader:
        x, m, fut = batch["x"].to(device), batch["mask"].to(device), batch["future"].to(device)
        anchor = batch["anchor"].to(device); road = batch["road"].to(device)
        pred = model(x, m, anchor=anchor, road_ids=road)["pred"]
        err = pred - fut
        abs_sum += err.abs().sum().item()
        sq_sum += (err * err).sum().item()
        count += err.numel()
    return {"MAE_m": abs_sum / count, "RMSE_m": math.sqrt(sq_sum / count), "coordinates": count}


def run(args):
    seed_all(args.seed)
    if args.dataset == "worldtrace":
        trajectories = read_worldtrace(args.path, args.limit, args.worldtrace_step)
    elif args.dataset == "geolife":
        trajectories = read_geolife(args.path, args.limit, max_step_m=args.max_step_m)
    else:
        trajectories = read_chengdu(args.path, args.limit, max_step_m=args.chengdu_max_step_m)
    if len(trajectories) < 30:
        raise RuntimeError(f"only {len(trajectories)} usable trajectories")
    # Keep the data partition independent from the model initialization seed.
    # This makes multi-seed runs comparable and prevents seed search from
    # silently changing the validation/test examples.
    train_raw, val_raw, test_raw = split_trajectories(trajectories, args.split_seed)
    if args.refit_trainval:
        # Final refit mode uses the held-in validation trajectories as extra
        # training data while retaining the validation view for checkpoint
        # selection; the test trajectories remain untouched.
        train_raw = train_raw + val_raw
    road_payload = pickle.loads(args.road_cache.read_bytes()) if args.road_cache else None
    road_lookup = road_payload["lookup"] if road_payload else None
    train = ForecastDataset(train_raw, args.max_history, args.augment_windows, args.rotate_train, args.history_stride, road_lookup)
    val = ForecastDataset(val_raw, args.max_history, history_stride=args.history_stride, road_lookup=road_lookup)
    test = ForecastDataset(test_raw, args.max_history, history_stride=args.history_stride, road_lookup=road_lookup)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = (GRUPredictor(args.hidden) if args.method == "gru" else
             ACTPPredictor(args.hidden, args.bins, args.delta_decode,
                           args.autoregressive_decode, args.encoder_layers,
                           args.velocity_context, args.activation,
                           args.decoder_type, args.pooling,
                           args.velocity_residual, args.endpoint_blend,
                           args.kinematic_gate, args.velocity_input,
                           args.input_transform, args.kinematic_mixture,
                           args.raw_context, args.absolute_context,
                           road_payload["road_features"] if road_payload else None,
                           args.velocity_gap, args.fusion_mode)).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        # Older checkpoints predate the optional endpoint head; its output is
        # unused unless endpoint loss/blending is enabled.
        model.load_state_dict(state, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.tail_sampler > 0:
        ext = np.asarray([float(np.linalg.norm(item[1], axis=1).max()) for item in train.items], dtype=np.float64)
        weights = 1.0 + args.tail_sampler * np.log1p(ext / 250.0)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    history = []
    best = float("inf")
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {k: 0.0 for k in ("total", "pred", "occ", "hyp", "order", "curvature")}
        n = 0
        for batch in train_loader:
            x, m, fut = batch["x"].to(device), batch["mask"].to(device), batch["future"].to(device)
            anchor = batch["anchor"].to(device); road = batch["road"].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x, m, fut, args.teacher_forcing, anchor, road)
            if args.method == "gru":
                horizon_weight = out["pred"].new_tensor(args.horizon_gamma ** np.arange(HORIZON, dtype=np.float32))
                horizon_weight = (horizon_weight / horizon_weight.mean()).view(1, HORIZON, 1)
                if args.pred_loss == "smoothl1":
                    robust_loss = F.smooth_l1_loss(out["pred"], fut, beta=args.smoothl1_beta, reduction="none")
                    robust_loss = (robust_loss * horizon_weight).mean()
                    total = ((1.0 - args.mse_mix) * robust_loss +
                             args.mse_mix * ((out["pred"] - fut).square() * horizon_weight).mean())
                else:
                    total = ((out["pred"] - fut).square() * horizon_weight).mean()
                parts = {"pred": total, "occ": total.new_zeros(()),
                         "hyp": total.new_zeros(()), "order": total.new_zeros(()),
                         "curvature": total.new_zeros(())}
            else:
                if args.lambda_hyp > 0 and epoch > args.aux_warmup:
                    if args.negative_mode == "real":
                        id_d, id_s = real_route_negative_indices(x, anchor)
                        out["neg"] = [model(x[id_d], m[id_d], anchor=anchor[id_d], road_ids=road[id_d])["z"],
                                      model(x[id_s], m[id_s], anchor=anchor[id_s], road_ids=road[id_s])["z"]]
                    else:
                        xd, xs = negative_views(x, m)
                        out["neg"] = [model(xd, m, anchor=anchor, road_ids=road)["z"],
                                      model(xs, m, anchor=anchor, road_ids=road)["z"]]
                total, parts = losses(model, out, x, m, fut, args)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            bs = x.shape[0]
            totals["total"] += float(total.detach()) * bs
            for k, v in parts.items(): totals[k] += float(v.detach()) * bs
            n += bs
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, **{k: v / max(n, 1) for k, v in totals.items()}, "val": val_metrics}
        if args.method == "actp":
            row["curvature"] = float(model.curvature().detach())
        history.append(row)
        score = val_metrics["RMSE_m"] if args.select == "rmse" else val_metrics["MAE_m"]
        if score < best:
            best = score
            torch.save(model.state_dict(), str(args.output) + ".pt")
        print(json.dumps(row), flush=True)
    best_row = min(history, key=lambda row: row["val"]["RMSE_m" if args.select == "rmse" else "MAE_m"])
    if args.test_policy == "evaluate":
        model.load_state_dict(torch.load(str(args.output) + ".pt", map_location=device))
        test_metrics = evaluate(model, test_loader, device)
    else:
        test_metrics = {"status": "sealed", "reason": "candidate selection uses validation only"}
    report = {"dataset": args.dataset, "source": str(args.path), "seed": args.seed,
              "split_seed": args.split_seed, "test_policy": args.test_policy,
              "method": args.method, "horizon": HORIZON, "max_history": args.max_history,
              "counts": {"all": len(trajectories), "train": len(train), "val": len(val), "test": len(test)},
              "device": str(device), "config": {k: str(v) if isinstance(v, Path) else v
                                                   for k, v in vars(args).items()}, "test": test_metrics,
              "best_validation": best_row["val"], "best_epoch": best_row["epoch"], "history": history}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"test": test_metrics, "output": str(args.output)}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("worldtrace", "geolife", "chengdu"), required=True)
    p.add_argument("--method", choices=("actp", "gru"), default="actp")
    p.add_argument("--path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None,
                   help="initialize from a compatible checkpoint before fine-tuning")
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--worldtrace-step", type=int, default=3,
                   help="WorldTrace sampling stride; UniTraj evaluates at 3 seconds")
    p.add_argument("--max-step-m", type=float, default=2000.0,
                   help="split GeoLife segments at physically disconnected consecutive points")
    p.add_argument("--chengdu-max-step-m", type=float, default=0.0,
                   help="reject Chengdu trajectories with a larger adjacent GPS jump; 0 disables")
    p.add_argument("--max-history", type=int, default=64)
    p.add_argument("--history-stride", type=int, default=1,
                   help="spacing between retained history points")
    p.add_argument("--augment-windows", type=int, default=0,
                   help="number of additional earlier training cut points per trajectory")
    p.add_argument("--rotate-train", action="store_true",
                   help="apply a random planar rotation to each training window")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--bins", type=int, default=21)
    p.add_argument("--delta-decode", action="store_true",
                   help="predict per-step displacements and accumulate them over the horizon")
    p.add_argument("--autoregressive-decode", action="store_true",
                   help="decode future displacements recurrently across prediction steps")
    p.add_argument("--teacher-forcing", type=float, default=0.0,
                   help="teacher-forcing probability for autoregressive decoding")
    p.add_argument("--encoder-layers", type=int, default=1,
                   help="number of recurrent layers in each scale encoder")
    p.add_argument("--velocity-context", action="store_true",
                   help="add the last observed displacement as an explicit context cue")
    p.add_argument("--activation", choices=("gelu", "silu", "relu", "leaky_relu"), default="gelu")
    p.add_argument("--decoder-type", choices=("direct", "horizon"), default="direct")
    p.add_argument("--pooling", choices=("meanmax", "attention"), default="meanmax")
    p.add_argument("--velocity-residual", action="store_true",
                   help="predict a residual around constant-velocity extrapolation")
    p.add_argument("--velocity-gap", type=int, default=1,
                   help="history interval used by the velocity residual baseline")
    p.add_argument("--endpoint-blend", type=float, default=0.0,
                   help="blend a dedicated endpoint head into the final prediction")
    p.add_argument("--endpoint-weight", type=float, default=0.0,
                   help="auxiliary loss weight for the final future location")
    p.add_argument("--kinematic-gate", action="store_true",
                   help="learn a bounded blend between the decoder and constant-velocity extrapolation")
    p.add_argument("--velocity-input", action="store_true",
                   help="append per-step displacement channels to each scale encoder")
    p.add_argument("--input-transform", choices=("none", "log"), default="none",
                   help="signed-log transform of scale-normalized coordinates")
    p.add_argument("--kinematic-mixture", action="store_true",
                   help="learn a mixture of the decoder and multiple velocity extrapolators")
    p.add_argument("--raw-context", action="store_true",
                   help="add a projection of the full fixed-length history to the fused state")
    p.add_argument("--absolute-context", action="store_true",
                   help="encode the last observed global position at nested spatial frequencies")
    p.add_argument("--road-cache", type=Path, default=None,
                   help="optional map-matched road ids and two-hop structural road features")
    p.add_argument("--fusion-mode", choices=("transition", "hierarchical"), default="transition",
                   help="learned transition fusion or direct coarse-to-fine hierarchy")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-occ", type=float, default=0.5)
    p.add_argument("--lambda-hyp", type=float, default=0.2)
    p.add_argument("--lambda-ord", type=float, default=0.1)
    p.add_argument("--lambda-curvature", type=float, default=1e-4)
    p.add_argument("--curvature-reference", type=float, default=1.0)
    p.add_argument("--aux-warmup", type=int, default=0,
                   help="prediction-only epochs before enabling HCSC negatives")
    p.add_argument("--negative-mode", choices=("real", "synthetic"), default="real",
                   help="recorded route alternatives or legacy coordinate perturbations")
    p.add_argument("--pred-loss", choices=("mse", "smoothl1"), default="mse")
    p.add_argument("--smoothl1-beta", type=float, default=25.0)
    p.add_argument("--mse-mix", type=float, default=0.0,
                   help="weight of MSE mixed into SmoothL1 (0 keeps the robust loss)")
    p.add_argument("--tail-weight", type=float, default=0.0,
                   help="controlled sample weighting based on future displacement extent")
    p.add_argument("--tail-sampler", type=float, default=0.0,
                   help="oversample long-displacement training windows")
    p.add_argument("--horizon-gamma", type=float, default=1.0,
                   help="geometric weight applied to later prediction steps")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=20260906,
                   help="fixed trajectory partition seed, independent of model initialization")
    p.add_argument("--test-policy", choices=("sealed", "evaluate"), default="sealed",
                   help="keep test sealed during search; evaluate only after freezing a candidate")
    p.add_argument("--refit-trainval", action="store_true",
                   help="include validation trajectories in the training pool")
    p.add_argument("--select", choices=("mae", "rmse"), default="mae",
                   help="validation metric used to select the checkpoint")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
