#!/usr/bin/env bash
set -euo pipefail

device="${1:-cuda:0}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

run_id="$(date +%Y%m%d_%H%M%S)"
log_dir="backend_logs/$run_id"
mkdir -p "$log_dir"/{train,classification,instance,class,table3,markers}
manifest="$log_dir/manifest.tsv"

iscx_datasets=(iscx-vpn iscx-nonvpn iscx-tor iscx-nontor)
iscx_baselines=("Byte-Transformer")
table3_datasets=(ios android)
all_datasets=(iscx-vpn iscx-nonvpn iscx-tor iscx-nontor ios android)
instance_methods=(random saliency traffic-explainer lime)
byte_transformer_extra_methods=(att-rollout)
class_methods=(random saliency traffic-explainer lime)
table3_methods=(random saliency traffic-explainer lime)
budgets="0.01,0.05,0.1"
mask_budget="0.1"

echo -e "status\tkind\tbaseline\tdataset\tmethod\tlog" > "$manifest"

slug() {
    printf '%s' "$*" | tr ' /' '__'
}

run_job() {
    local kind="$1"
    local baseline="$2"
    local dataset="$3"
    local method="$4"
    shift 4
    local name
    name="$(slug "$kind" "$baseline" "$dataset" "$method")"
    local marker="$log_dir/markers/$name.done"
    local log="$log_dir/$kind/$name.log"

    if [[ -f "$marker" ]]; then
        echo -e "SKIP\t$kind\t$baseline\t$dataset\t$method\t$log" | tee -a "$manifest"
        return 0
    fi

    echo "[$(date '+%F %T')] START $kind | $baseline | $dataset | $method" | tee "$log"
    if "$@" >> "$log" 2>&1; then
        touch "$marker"
        echo "[$(date '+%F %T')] DONE $kind | $baseline | $dataset | $method" >> "$log"
        echo -e "DONE\t$kind\t$baseline\t$dataset\t$method\t$log" | tee -a "$manifest"
    else
        echo "[$(date '+%F %T')] FAIL $kind | $baseline | $dataset | $method" >> "$log"
        echo -e "FAIL\t$kind\t$baseline\t$dataset\t$method\t$log" | tee -a "$manifest"
        return 0
    fi
}

dataset_config_name() {
    case "$1" in
        iscx-vpn) echo "ISCX-VPN-2016" ;;
        iscx-nonvpn) echo "ISCX-NonVPN-2016" ;;
        iscx-tor) echo "ISCX-Tor-2017" ;;
        iscx-nontor) echo "ISCX-NonTor-2017" ;;
        ios) echo "IOS_Cross_Plat" ;;
        android) echo "Android_Cross_Plat" ;;
        *) return 1 ;;
    esac
}

checkpoint_dir_for() {
    local baseline="$1"
    local dataset="$2"
    printf 'model/%s/%s' "$baseline" "$(dataset_config_name "$dataset")"
}

has_local_checkpoint() {
    local checkpoint_dir="$1"
    [[ -f "$checkpoint_dir/model.pth" && \
       -f "$checkpoint_dir/model_pack.pth" && \
       -f "$checkpoint_dir/model_header.pth" && \
       -f "$checkpoint_dir/model_header_pack.pth" ]]
}

run_with_checkpoint() {
    local kind="$1"
    local baseline="$2"
    local dataset="$3"
    local method="$4"
    shift 4
    local checkpoint_dir
    checkpoint_dir="$(checkpoint_dir_for "$baseline" "$dataset")"
    if ! has_local_checkpoint "$checkpoint_dir"; then
        local log="$log_dir/$kind/$(slug "$kind" "$baseline" "$dataset" "$method").log"
        echo "Missing reproduced local checkpoint: $checkpoint_dir" > "$log"
        echo -e "SKIP_NO_LOCAL_CKPT\t$kind\t$baseline\t$dataset\t$method\t$log" | tee -a "$manifest"
        return 0
    fi
    run_job "$kind" "$baseline" "$dataset" "$method" "$@"
}

train_model() {
    local baseline="$1"
    local dataset="$2"
    local checkpoint_dir
    checkpoint_dir="$(checkpoint_dir_for "$baseline" "$dataset")"
    rm -rf "$checkpoint_dir"
    python3 main.py --dataset "$dataset" --baseline "$baseline" --device "$device"
    has_local_checkpoint "$checkpoint_dir"
}

for baseline in "${iscx_baselines[@]}"; do
    for dataset in "${all_datasets[@]}"; do
        run_job train "$baseline" "$dataset" fresh train_model "$baseline" "$dataset"
    done
done

python3 scripts/repro_integrity_check.py "$device" | tee "$log_dir/integrity.log"

for baseline in "${iscx_baselines[@]}"; do
    for dataset in "${all_datasets[@]}"; do
        run_with_checkpoint classification "$baseline" "$dataset" test \
            python3 test.py --dataset "$dataset" --baseline "$baseline" --device "$device"
    done
done

for baseline in "${iscx_baselines[@]}"; do
    for dataset in "${iscx_datasets[@]}"; do
        for method in "${instance_methods[@]}"; do
            run_with_checkpoint instance "$baseline" "$dataset" "$method" \
                python3 explain.py --dataset "$dataset" --baseline "$baseline" --explanation "$method" \
                    --device "$device" --mask_budget "$mask_budget" --budgets "$budgets"
        done

        if [[ "$baseline" == "Byte-Transformer" ]]; then
            for method in "${byte_transformer_extra_methods[@]}"; do
                run_with_checkpoint instance "$baseline" "$dataset" "$method" \
                    python3 explain.py --dataset "$dataset" --baseline "$baseline" --explanation "$method" \
                        --device "$device" --mask_budget "$mask_budget" --budgets "$budgets"
            done
        fi
    done
done

for baseline in "${iscx_baselines[@]}"; do
    for dataset in "${iscx_datasets[@]}"; do
        for method in "${class_methods[@]}"; do
            run_with_checkpoint class "$baseline" "$dataset" "$method" \
                python3 explain_class.py --dataset "$dataset" --baseline "$baseline" --explanation "$method" \
                    --device "$device" --mask_budget "$mask_budget" --budgets "$budgets"
        done
    done
done

for baseline in "${iscx_baselines[@]}"; do
    for dataset in "${table3_datasets[@]}"; do
        for method in "${table3_methods[@]}"; do
            run_with_checkpoint table3 "$baseline" "$dataset" "$method" \
                python3 explain.py --dataset "$dataset" --baseline "$baseline" --explanation "$method" \
                    --device "$device" --mask_budget "$mask_budget" --budgets "$budgets"
        done
    done
done

python3 scripts/summarize_backend_results.py "$log_dir" | tee "$log_dir/summary.txt"
echo "Backend reproduction complete: $log_dir"
