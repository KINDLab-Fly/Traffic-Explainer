#!/usr/bin/env bash
set -euo pipefail

baseline="${1:-Byte-Transformer}"
device="${2:-cuda:0}"
dataset="${3:-all}"
method="${4:-all}"
mask_budget="${5:-0.1}"
budgets="${6:-0.01,0.05,0.1}"
if [[ "$dataset" == "all" ]]; then
    datasets=(iscx-vpn iscx-nonvpn iscx-tor iscx-nontor)
else
    datasets=("$dataset")
fi

if [[ "$method" == "all" ]]; then
    methods=(random saliency traffic-explainer lime att-rollout)
else
    methods=("$method")
fi

for current_dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        echo "==================== ${baseline} - ${current_dataset} - ${method} ===================="
        python3 explain.py \
            --dataset "$current_dataset" \
            --baseline "$baseline" \
            --explanation "$method" \
            --device "$device" \
            --mask_budget "$mask_budget" \
            --budgets "$budgets"
    done
done
