#!/usr/bin/env bash
set -euo pipefail

baseline="${1:-Byte-Transformer}"
device="${2:-cuda:0}"
dataset="${3:-all}"

if [[ "$dataset" == "all" ]]; then
    datasets=(ios android)
else
    datasets=("$dataset")
fi

for current_dataset in "${datasets[@]}"; do
    echo "==================== ${baseline} - ${current_dataset} ===================="
    python3 main.py --dataset "$current_dataset" --baseline "$baseline" --device "$device"
    python3 test.py --dataset "$current_dataset" --baseline "$baseline" --device "$device"
done
