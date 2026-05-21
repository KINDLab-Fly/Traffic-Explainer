# Traffic-Explainer

This repository contains the reproduction code for Traffic-Explainer experiments. It includes traffic classification, instance-level explanations, class-level explanations, and cross-platform localization runs.

## Repository Layout

```text
config.py                     Dataset and training configuration
dataset.py                    Dataset loaders for processed traffic data
model.py                      Byte-Transformer model and explainer modules
main.py                       Train classifiers
test.py                       Evaluate classifiers
explain.py                    Instance-level byte explanations
explain_class.py              Class-level byte explanations
run_classification.sh         ISCX training and testing runner
run_classification_anomaly.sh IOS/Android localization training and testing runner
run_explanations.sh           ISCX instance-level explanation runner
run_explanations_anomaly.sh   IOS/Android localization explanation runner
run_explanations_class.sh     ISCX class-level explanation runner
scripts/                      Backend orchestration and result summaries
docs/                         Data, reproduction, and release notes
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your machine needs a CUDA-specific PyTorch wheel, install the matching PyTorch build before running the experiments.

For SHAP explanations:

```bash
pip install -r requirements-optional.txt
```

## Data

The code expects processed datasets under:

```text
dataset/
dataset_localization/
```

For local reproduction these may be symlinks:

```bash
ln -s /path/to/Traffic-Explainer-data/dataset dataset
ln -s /path/to/Traffic-Explainer-data/dataset_localization dataset_localization
```

For step-by-step data reproduction, start here:

[docs/DATA_REPRODUCTION.md](docs/DATA_REPRODUCTION.md)

Source links:

- Localization data: [WM-JayLab/NetBench](https://github.com/WM-JayLab/NetBench), package `Level` folder from the [NetBench Google Drive dataset link](https://drive.google.com/drive/u/0/folders/1dYGHKKJR5WS4cXISk9AtfW_gB9mdC2ED)
- ISCX data: follow [ViktorAxelsen/TFE-GNN](https://github.com/ViktorAxelsen/TFE-GNN) preprocessing; raw data from [UNB/CIC VPN 2016](https://www.unb.ca/cic/datasets/vpn.html) and [UNB/CIC Tor 2016](https://www.unb.ca/cic/datasets/tor.html)

Supporting references:

- [docs/DATA.md](docs/DATA.md): expected file layout and preprocessing commands
- [docs/DATASETS.md](docs/DATASETS.md): per-dataset labels, shapes, and split counts
- [docs/SPLITCAP.md](docs/SPLITCAP.md): what SplitCap does and how to run the tracked wrapper

## Classification

Train and test one ISCX dataset:

```bash
python3 main.py --dataset iscx-vpn --baseline Byte-Transformer --device cuda:0
python3 test.py --dataset iscx-vpn --baseline Byte-Transformer --device cuda:0
```

Run all ISCX classification jobs:

```bash
./run_classification.sh Byte-Transformer cuda:0 all
```

Run IOS/Android localization classification jobs:

```bash
./run_classification_anomaly.sh Byte-Transformer cuda:0 all
```

Supported dataset aliases:

```text
iscx-vpn
iscx-nonvpn
iscx-tor
iscx-nontor
ios
android
```

Checkpoints are written under `model/<baseline>/<dataset>/`.

If checkpoints are stored outside the repository, set:

```bash
export TRAFFIC_EXPLAINER_CHECKPOINT_ROOT=/path/to/Traffic-Explainer-checkpoints
```

## Explanations

Run Traffic-Explainer for one dataset:

```bash
python3 explain.py \
  --dataset iscx-vpn \
  --baseline Byte-Transformer \
  --explanation traffic-explainer \
  --device cuda:0 \
  --mask_budget 0.1 \
  --budgets 0.01,0.05,0.1
```

Run all default instance-level explanation methods for ISCX:

```bash
./run_explanations.sh Byte-Transformer cuda:0 all all 0.1 0.01,0.05,0.1
```

Run all default class-level explanation methods for ISCX:

```bash
./run_explanations_class.sh Byte-Transformer cuda:0 all all 0.01,0.05,0.1 0.1
```

Run IOS/Android localization explanations:

```bash
./run_explanations_anomaly.sh Byte-Transformer cuda:0 all all 0.1 0.01,0.05,0.1
```

Default ISCX instance-level explanation methods:

```text
random
saliency
traffic-explainer
lime
att-rollout
```

Default IOS/Android localization explanation methods are `random`, `saliency`, `traffic-explainer`, and `lime`. `att-rollout` is available in `explain.py` for Byte-Transformer instance-level runs. `shap` is also implemented in `explain.py`, but it is excluded from default runners because it is slow; run it explicitly when needed.

Explanation outputs are written under `res/<baseline>/<dataset>/`.

## Full Backend Reproduction

The backend runner executes the main reproduction sequence and records logs plus a manifest:

```bash
bash scripts/run_backend_all.sh cuda:0
```

Summarize a run:

```bash
python3 scripts/summarize_backend_results.py backend_logs/<run_id>
```

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for more detailed commands.
