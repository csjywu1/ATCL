"""Dataset utilities for five-step GeoLife trajectory prediction."""

from __future__ import annotations

import io
import math
import pickle
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HORIZON = 5


def local_xy(points: np.ndarray) -> np.ndarray:
    """Convert latitude/longitude to local metric coordinates."""
    points = np.asarray(points, dtype=np.float64)
    lat0, lon0 = float(points[0, 0]), float(points[0, 1])
    x = (points[:, 1] - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    y = (points[:, 0] - lat0) * 110_540.0
    return np.stack([x, y], axis=1)


def _split_contiguous(points: list[tuple[float, float]], min_points: int) -> list[np.ndarray]:
    if len(points) < min_points:
        return []
    arr = np.asarray(points, dtype=np.float64)
    xy = local_xy(arr)
    cuts = np.flatnonzero(np.linalg.norm(np.diff(xy, axis=0), axis=1) > 2_000.0) + 1
    out, start = [], 0
    for end in np.r_[cuts, len(arr)]:
        if end - start >= min_points:
            out.append(arr[start:end, :2])
        start = int(end)
    return out


def load_geolife(path: str | Path, limit: int | None = None, min_points: int = 36) -> list[np.ndarray]:
    """Load a processed pickle or the official GeoLife ZIP archive."""
    path = Path(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        rows = pickle.loads(path.read_bytes())
        out = []
        for row in rows:
            arr = np.asarray(row, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= min_points:
                out.append(arr[:, :2])
                if limit and len(out) >= limit:
                    break
        return out

    out: list[np.ndarray] = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".plt"):
                continue
            points: list[tuple[float, float]] = []
            try:
                with archive.open(name) as raw:
                    for line in io.TextIOWrapper(raw, encoding="utf-8", errors="ignore"):
                        parts = line.strip().split(",")
                        if len(parts) >= 2:
                            points.append((float(parts[0]), float(parts[1])))
            except (OSError, UnicodeError, ValueError):
                continue
            out.extend(_split_contiguous(points, min_points))
            if limit and len(out) >= limit:
                return out[:limit]
    return out


def split_trajectories(trajectories: list[np.ndarray], seed: int = 3) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Create deterministic 80/10/10 trajectory-level splits."""
    order = np.random.default_rng(seed).permutation(len(trajectories))
    train_end, val_end = int(0.8 * len(order)), int(0.9 * len(order))
    return (
        [trajectories[i] for i in order[:train_end]],
        [trajectories[i] for i in order[train_end:val_end]],
        [trajectories[i] for i in order[val_end:]],
    )


class ForecastDataset(Dataset):
    """Historical prefixes paired with the final five recorded locations."""

    def __init__(self, trajectories: list[np.ndarray], max_history: int = 32):
        self.max_history = max_history
        self.items: list[tuple[np.ndarray, np.ndarray]] = []
        for raw in trajectories:
            if len(raw) <= HORIZON:
                continue
            xy = local_xy(raw)
            cut = len(xy) - HORIZON
            history, future = xy[:cut][-max_history:], xy[cut:]
            origin = history[-1].copy()
            history, future = history - origin, future - origin
            dt = np.arange(-len(history) + 1, 1, dtype=np.float32)
            x = np.zeros((max_history, 3), dtype=np.float32)
            mask = np.zeros(max_history, dtype=np.bool_)
            start = max_history - len(history)
            x[start:, :2], x[start:, 2], mask[start:] = history, dt, True
            self.items.append((x, mask, future.astype(np.float32)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        x, mask, future = self.items[index]
        return {
            "x": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "future": torch.from_numpy(future),
        }
