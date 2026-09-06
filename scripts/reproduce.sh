#!/usr/bin/env bash
set -euo pipefail

dataset=${1:?usage: scripts/reproduce.sh worldtrace|chengdu|geolife [seed]}
seed=${2:-100}
python -m train.train \
  --config "configs/${dataset}.json" \
  --seed "$seed" \
  --output "results/runs/${dataset}_seed${seed}.json"

echo "Training keeps the test split sealed. Evaluate the frozen checkpoint with:"
echo "python -m inference.evaluate --report results/runs/${dataset}_seed${seed}.json"
