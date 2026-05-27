# Data Layout

This repository does not commit datasets or generated model artifacts.

For a step-by-step guide from raw/source files to the arrays used by training and explanation, see [DATA_REPRODUCTION.md](DATA_REPRODUCTION.md).

For per-dataset labels, shapes, split counts, and task definitions, see [DATASETS.md](DATASETS.md).

Expected local paths from the repository root:

```text
dataset/
  ISCX-VPN-2016/
  ISCX-NonVPN-2016/
  ISCX-Tor-2017/
  ISCX-NonTor-2017/

dataset_localization/
  IOS_Cross_Plat/
  Android_Cross_Plat/
```

Ready-to-use processed dataset archives:

- [Traffic-Explainer_dataset.zip](https://www.dropbox.com/scl/fi/mh00t4ycemjkirnjotojs/Traffic-Explainer_dataset.zip?rlkey=v4m5yf4rgfcux7pvz8ulzr7hz&st=bphzy61v&dl=0): final ISCX arrays. Extract at the repository root so the top-level folder is `dataset/`.
- [Traffic-Explainer_dataset_localization.zip](https://www.dropbox.com/scl/fi/bbckc21gericqyiooxq4c/Traffic-Explainer_dataset_localization.zip?rlkey=p0s0z0gcsbg50pglgw4t2oe00&st=bma0fbnd&dl=0): final IOS/Android localization CSVs and pickles. Extract at the repository root so the top-level folder is `dataset_localization/`.

Source links:

- Localization data comes from [WM-JayLab/NetBench](https://github.com/WM-JayLab/NetBench), package `Level` folder in the [NetBench Google Drive dataset folder](https://drive.google.com/drive/u/0/folders/1dYGHKKJR5WS4cXISk9AtfW_gB9mdC2ED).
- ISCX raw PCAPs follow the [ViktorAxelsen/TFE-GNN](https://github.com/ViktorAxelsen/TFE-GNN) preprocessing instructions and are downloaded from [UNB/CIC VPN 2016](https://www.unb.ca/cic/datasets/vpn.html) and [UNB/CIC Tor 2016](https://www.unb.ca/cic/datasets/tor.html).

The code expects the following processed files for ISCX datasets:

```text
train_pyg.npz
val_pyg.npz
test_pyg.npz
header_train_pyg.npz
header_val_pyg.npz
header_test_pyg.npz
```

The cross-platform localization datasets expect:

```text
train.pkl
val.pkl
test.pkl
```

For local reproduction, these paths can be normal directories or symlinks to a shared data location:

```bash
ln -s /path/to/Traffic-Explainer-data/dataset dataset
ln -s /path/to/Traffic-Explainer-data/dataset_localization dataset_localization
```

Generated checkpoints and explanation outputs are written to `model/` and `res/`. Both are ignored by git.

## Rebuild Localization Pickles

The tracked preprocessing entrypoint for IOS/Android localization data is:

```bash
python3 scripts/preprocess_localization.py --dataset ios --overwrite
python3 scripts/preprocess_localization.py --dataset android --overwrite
```

The source CSVs are from [WM-JayLab/NetBench](https://github.com/WM-JayLab/NetBench), package `Level` folder in the [NetBench Google Drive dataset folder](https://drive.google.com/drive/u/0/folders/1dYGHKKJR5WS4cXISk9AtfW_gB9mdC2ED).

By default, this reproduces the representation expected by the current model: 64-byte `head` and `pkt` arrays, padding value `65536`, and country labels in first-seen CSV order. The train split uses CSV rows whose `dataset_type` is `train`; all non-train rows are deterministically shuffled with seed `32` and split equally into validation and test sets. To preserve the CSV `dev` and `test` split labels instead, add `--split_strategy source`.

## Rebuild ISCX Arrays

Our ISCX preprocessing follows the public [TFE-GNN](https://github.com/ViktorAxelsen/TFE-GNN) preprocessing recipe:

- Download the raw ISCX VPN/NonVPN PCAP datasets from [UNB/CIC VPN 2016](https://www.unb.ca/cic/datasets/vpn.html), and Tor/NonTor PCAP datasets from [UNB/CIC Tor 2016](https://www.unb.ca/cic/datasets/tor.html).
- Use SplitCap to split raw captures into bidirectional flow/session pcaps.
- Keep TCP sessions only.
- Convert each session pcap into a per-session `.npz` file.
- Aggregate those per-session `.npz` files into the final arrays consumed by training and explanation.

Reference: [TFE-GNN README](https://github.com/ViktorAxelsen/TFE-GNN#pre-processing).

The model loads the final ISCX arrays directly:

```text
dataset/<name>/train_pyg.npz
dataset/<name>/val_pyg.npz
dataset/<name>/test_pyg.npz
dataset/<name>/header_train_pyg.npz
dataset/<name>/header_val_pyg.npz
dataset/<name>/header_test_pyg.npz
```

The full preprocessing path is:

```text
raw pcap files
  -> SplitCap.exe session pcaps under process_file/<class>/
  -> per-session .npz files with header/payload/time fields
  -> final train/val/test .npz arrays
```

If per-session `.npz` files already exist under `process_file/<class>/`, rebuild the runtime arrays with:

```bash
python3 scripts/preprocess_iscx.py --dataset iscx-vpn --stage build --overwrite
python3 scripts/preprocess_iscx.py --dataset iscx-nonvpn --stage build --overwrite
python3 scripts/preprocess_iscx.py --dataset iscx-tor --stage build --overwrite
python3 scripts/preprocess_iscx.py --dataset iscx-nontor --stage build --overwrite
```

To start from raw pcaps, place the raw files in `dataset/<name>/raw/`, keep `SplitCap.exe` in the dataset folder, install `mono` and optional Python preprocessing dependencies, then run:

```bash
pip install -r requirements-optional.txt
python3 scripts/preprocess_iscx.py --dataset iscx-vpn --stage all --overwrite
```

`--stage all` runs SplitCap, converts session pcaps to session `.npz`, and builds final runtime arrays. Tor uses non-overlapping 60-second windows after session conversion; the other ISCX datasets treat each session as one flow. The release script defaults to the original filesystem file order; use `--file_order sorted` for deterministic ordering on a new machine.

For SplitCap-only guidance, see [SPLITCAP.md](SPLITCAP.md) or run:

```bash
scripts/splitcap_iscx.sh iscx-vpn
```
