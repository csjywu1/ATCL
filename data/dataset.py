"""Dataset API used by ACTP training and inference."""

from actp.runner import (
    HORIZON,
    ForecastDataset,
    local_xy,
    read_chengdu,
    read_geolife,
    read_worldtrace,
    split_trajectories,
)

LOADERS = {
    "worldtrace": read_worldtrace,
    "chengdu": read_chengdu,
    "geolife": read_geolife,
}

__all__ = [
    "HORIZON",
    "ForecastDataset",
    "LOADERS",
    "local_xy",
    "read_chengdu",
    "read_geolife",
    "read_worldtrace",
    "split_trajectories",
]
