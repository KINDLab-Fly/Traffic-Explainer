#!/usr/bin/env bash
set -euo pipefail

baseline="${1:-Byte-Transformer}"
device="${2:-cuda:0}"
dataset="${3:-all}"
method="${4:-all}"
budgets="${5:-0.01,0.05,0.1}"
mask_budget="${6:-0.1}"

if [[ "$dataset" == "all" ]]; then
    datasets=(iscx-vpn iscx-nonvpn iscx-tor iscx-nontor)
else
    datasets=("$dataset")
fi

if [[ "$method" == "all" ]]; then
    methods=(random saliency traffic-explainer lime)
else
    methods=("$method")
fi

for current_dataset in "${datasets[@]}"; do
    for current_method in "${methods[@]}"; do
        echo "==================== class-level ${baseline} - ${current_dataset} - ${current_method} ===================="
        python3 explain_class.py \
            --dataset "$current_dataset" \
            --baseline "$baseline" \
            --explanation "$current_method" \
            --device "$device" \
            --budgets "$budgets" \
            --mask_budget "$mask_budget"
    done
done
