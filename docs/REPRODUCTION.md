# Reproduction Guide

Run all commands from the repository root.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA driver if the default wheel is not appropriate for your machine.

Optional SHAP support:

```bash
pip install -r requirements-optional.txt
```

## Rebuild IOS/Android Data

If the processed localization pickles are absent, rebuild them from the raw CSV files:

```bash
python3 scripts/preprocess_localization.py --dataset ios --overwrite
python3 scripts/preprocess_localization.py --dataset android --overwrite
```

The preprocessing is deterministic by default (`--seed 32`) and writes `train.pkl`, `val.pkl`, `test.pkl`, and `country2id.pkl` under each localization dataset folder. The default `--split_strategy combined-random` matches the old preprocessing logic; use `--split_strategy source` to map CSV `dev` rows to validation and CSV `test` rows to test directly.

## Rebuild ISCX Data

If per-session ISCX `.npz` files already exist under `dataset/<name>/process_file/<class>/`, rebuild the arrays used by training/testing with:

```bash
python3 scripts/preprocess_iscx.py --dataset iscx-vpn --stage build --overwrite
python3 scripts/preprocess_iscx.py --dataset iscx-nonvpn --stage build --overwrite
python3 scripts/preprocess_iscx.py --dataset iscx-tor --stage build --overwrite
python3 scripts/preprocess_iscx.py --dataset iscx-nontor --stage build --overwrite
```

To rebuild from raw pcaps, install `mono` for `SplitCap.exe`, install optional Python preprocessing dependencies, and use `--stage all`:

```bash
pip install -r requirements-optional.txt
python3 scripts/preprocess_iscx.py --dataset iscx-vpn --stage all --overwrite
```

## Train And Test Classification

Single dataset:

```bash
python3 main.py --dataset iscx-vpn --baseline Byte-Transformer --device cuda:0
python3 test.py --dataset iscx-vpn --baseline Byte-Transformer --device cuda:0
```

All ISCX datasets:

```bash
./run_classification.sh Byte-Transformer cuda:0 all
```

Cross-platform localization datasets:

```bash
./run_classification_anomaly.sh Byte-Transformer cuda:0 all
```

If checkpoints are stored outside this repository, point the evaluator to them with:

```bash
export TRAFFIC_EXPLAINER_CHECKPOINT_ROOT=/path/to/Traffic-Explainer-checkpoints
```

## Instance-Level Explanations

```bash
python3 explain.py \
  --dataset iscx-vpn \
  --baseline Byte-Transformer \
  --explanation traffic-explainer \
  --device cuda:0 \
  --mask_budget 0.1 \
  --budgets 0.01,0.05,0.1
```

Run all default instance-level methods for all ISCX datasets:

```bash
./run_explanations.sh Byte-Transformer cuda:0 all all 0.1 0.01,0.05,0.1
```

Supported instance-level methods:

```text
random
saliency
traffic-explainer
lime
att-rollout
shap
```

`shap` is available in `explain.py`, but it is intentionally excluded from default shell runners because it is slow.

## Class-Level Explanations

```bash
./run_explanations_class.sh Byte-Transformer cuda:0 iscx-vpn traffic-explainer 0.01,0.05,0.1 0.1
```

Default class-level methods:

```text
random
saliency
traffic-explainer
lime
```

## Full Backend Run

The backend driver trains, tests, runs explanations, and records a manifest:

```bash
bash scripts/run_backend_all.sh cuda:0
```

Logs are written under:

```text
backend_logs/<run_id>/
```

Summarize a completed run:

```bash
python3 scripts/summarize_backend_results.py backend_logs/<run_id>
```
