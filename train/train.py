"""Run ACTP training from a JSON configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def as_cli(config: dict) -> list[str]:
    command = [sys.executable, "-m", "actp.runner"]
    for key, value in config.items():
        if key.startswith("_") or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            command.append(flag)
        else:
            command.extend((flag, str(value)))
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("path", "output", "seed"):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    if args.cpu:
        config["cpu"] = True
    if "path" not in config or "output" not in config:
        parser.error("path and output must be present in the config or supplied as overrides")
    subprocess.run(as_cli(config), check=True)


if __name__ == "__main__":
    main()
