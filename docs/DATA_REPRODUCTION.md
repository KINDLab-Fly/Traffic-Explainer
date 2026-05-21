# Data Reproduction

This guide explains how to reproduce the datasets used by Traffic-Explainer.

The training and explanation code does not read raw PCAPs directly. It reads processed arrays:

```text
dataset/
dataset_localization/
```

Choose the section that matches the files you already have.

## 0. Data Sources

Traffic-Explainer uses two source families.

Localization datasets:

- Source repository: [WM-JayLab/NetBench](https://github.com/WM-JayLab/NetBench)
- Download location: [NetBench Google Drive dataset folder](https://drive.google.com/drive/u/0/folders/1dYGHKKJR5WS4cXISk9AtfW_gB9mdC2ED)
- Folder to use: package `Level`
- Tasks used here: Cross Platform Android/iOS `Country_Detection`
- Files expected by this repo:

```text
dataset_localization/IOS_Cross_Plat/Cross_Platform_iOS_pkt.csv
dataset_localization/Android_Cross_Plat/Cross_Platform_Android_pkt.csv
```

ISCX datasets:

- Preprocessing reference: [ViktorAxelsen/TFE-GNN](https://github.com/ViktorAxelsen/TFE-GNN)
- VPN/NonVPN raw data: [UNB/CIC VPN 2016 dataset](https://www.unb.ca/cic/datasets/vpn.html)
- Tor/NonTor raw data: [UNB/CIC Tor 2016 dataset](https://www.unb.ca/cic/datasets/tor.html)
- Workflow used here: raw PCAPs -> SplitCap bidirectional TCP sessions -> per-session `.npz` -> final arrays
- Datasets expected by this repo:

```text
dataset/ISCX-VPN-2016/
dataset/ISCX-NonVPN-2016/
dataset/ISCX-Tor-2017/
dataset/ISCX-NonTor-2017/
```

## 1. Final Layout Expected By Code

From the repository root, the expected layout is:

```text
dataset/
  ISCX-VPN-2016/
    train_pyg.npz
    val_pyg.npz
    test_pyg.npz
    header_train_pyg.npz
    header_val_pyg.npz
    header_test_pyg.npz
  ISCX-NonVPN-2016/
  ISCX-Tor-2017/
  ISCX-NonTor-2017/

dataset_localization/
  IOS_Cross_Plat/
    Cross_Platform_iOS_pkt.csv
    train.pkl
    val.pkl
    test.pkl
    country2id.pkl
  Android_Cross_Plat/
    Cross_Platform_Android_pkt.csv
    train.pkl
    val.pkl
    test.pkl
    country2id.pkl
```

If the data is stored outside the repository, symlink it:

```bash
ln -s /path/to/Traffic-Explainer-data/dataset dataset
ln -s /path/to/Traffic-Explainer-data/dataset_localization dataset_localization
```

## 2. Reproduce IOS/Android Localization Data

Starting files:

```text
dataset_localization/IOS_Cross_Plat/Cross_Platform_iOS_pkt.csv
dataset_localization/Android_Cross_Plat/Cross_Platform_Android_pkt.csv
```

Download these from [WM-JayLab/NetBench](https://github.com/WM-JayLab/NetBench), using the package `Level` folder in the [NetBench Google Drive dataset folder](https://drive.google.com/drive/u/0/folders/1dYGHKKJR5WS4cXISk9AtfW_gB9mdC2ED).

Install base dependencies:

```bash
pip install -r requirements.txt
```

Build the pickles:

```bash
python3 scripts/preprocess_localization.py --dataset ios --overwrite
python3 scripts/preprocess_localization.py --dataset android --overwrite
```

Default behavior:

- Reads CSV columns `text`, `dataset_type`, and `Country_Detection`.
- Converts the hex text into integer byte sequences.
- Writes `head`, `pkt`, and `label`.
- Pads/truncates `head` and `pkt` to length `64`.
- Uses padding token `65536`.
- Maps labels in first-seen CSV order: `india -> 0`, `china -> 1`, `us -> 2`.
- Keeps CSV `train` rows as training data.
- Combines non-train rows, shuffles with seed `32`, and splits them equally into validation/test.

To preserve the CSV `dev` and `test` labels directly:

```bash
python3 scripts/preprocess_localization.py --dataset ios --split_strategy source --overwrite
python3 scripts/preprocess_localization.py --dataset android --split_strategy source --overwrite
```

## 3. Reproduce ISCX Data From Per-Session NPZ Files

Use this path if you already have files under:

```text
dataset/<ISCX-name>/process_file/<class>/*.npz
```

Build the arrays used by training/testing:

```bash
python3 scripts/preprocess_iscx.py --dataset iscx-vpn --stage build --overwrite
python3 scripts/preprocess_iscx.py --dataset iscx-nonvpn --stage build --overwrite
python3 scripts/preprocess_iscx.py --dataset iscx-tor --stage build --overwrite
python3 scripts/preprocess_iscx.py --dataset iscx-nontor --stage build --overwrite
```

This writes:

```text
train_pyg.npz
val_pyg.npz
test_pyg.npz
header_train_pyg.npz
header_val_pyg.npz
header_test_pyg.npz
```

Default behavior:

- Payload array shape: `(num_flows, 50, 150)`.
- Header array shape: `(num_flows, 50, 40)`.
- Byte values: `0-255`.
- Padding token: `256`.
- Split per class in file order: first 80% train, next 10% validation, last 10% test.
- Cap each class at `9999` segments.
- For Tor, split each session into non-overlapping 60-second windows before padding/truncation.
- For VPN/NonVPN/NonTor, treat each session as one flow.

Use deterministic filename ordering on a new machine:

```bash
python3 scripts/preprocess_iscx.py --dataset iscx-vpn --stage build --file_order sorted --overwrite
```

The current local reproduced `ISCX-VPN-2016` tensors match the historical filesystem-order build. For full public reproducibility across machines, prefer `--file_order sorted` and report that setting.

## 4. Reproduce ISCX Data From Raw PCAP Files

This path follows the public TFE-GNN preprocessing recipe:

1. Download raw ISCX VPN/NonVPN PCAP datasets from [UNB/CIC VPN 2016](https://www.unb.ca/cic/datasets/vpn.html), and Tor/NonTor PCAP datasets from [UNB/CIC Tor 2016](https://www.unb.ca/cic/datasets/tor.html).
2. Put raw capture files under `dataset/<ISCX-name>/raw/`.
3. Put `SplitCap.exe` under `dataset/<ISCX-name>/`.
4. Install `mono` so Linux can run `SplitCap.exe`.
5. Install optional Python preprocessing dependencies:

```bash
pip install -r requirements-optional.txt
```

Run all preprocessing stages:

```bash
python3 scripts/preprocess_iscx.py --dataset iscx-vpn --stage all --overwrite
python3 scripts/preprocess_iscx.py --dataset iscx-nonvpn --stage all --overwrite
python3 scripts/preprocess_iscx.py --dataset iscx-tor --stage all --overwrite
python3 scripts/preprocess_iscx.py --dataset iscx-nontor --stage all --overwrite
```

The `all` stage performs:

```text
raw PCAP
  -> SplitCap bidirectional TCP session PCAP
  -> per-session .npz
  -> final train/val/test arrays
```

Notes:

- The ISCX workflow follows [ViktorAxelsen/TFE-GNN](https://github.com/ViktorAxelsen/TFE-GNN), which uses SplitCap to obtain bidirectional TCP flows before `.npz` conversion and byte-level preprocessing.
- SplitCap may require `.pcapng` to `.pcap` conversion for some raw files.
- Only TCP sessions are used.
- Dataset partitioning is not standardized by the original public datasets; file order and preprocessing choices can change final splits.

## 5. Verify Processed Data

After preprocessing, run:

```bash
python3 - <<'PY'
import numpy as np

for dataset in ['ISCX-VPN-2016', 'ISCX-NonVPN-2016', 'ISCX-Tor-2017', 'ISCX-NonTor-2017']:
    arr = np.load(f'dataset/{dataset}/train_pyg.npz')
    print(dataset, arr['data'].shape, arr['label'].shape)
PY
```

Then run a quick classification smoke test:

```bash
python3 test.py --dataset iscx-vpn --baseline Byte-Transformer --device cuda:0
```

If you do not have checkpoints yet, train first:

```bash
python3 main.py --dataset iscx-vpn --baseline Byte-Transformer --device cuda:0
```
