# Data Layout

This repository does not commit datasets or generated model artifacts.

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

By default, this reproduces the representation expected by the current model: 64-byte `head` and `pkt` arrays, padding value `65536`, and country labels in first-seen CSV order. The train split uses CSV rows whose `dataset_type` is `train`; all non-train rows are deterministically shuffled with seed `32` and split equally into validation and test sets. To preserve the CSV `dev` and `test` split labels instead, add `--split_strategy source`.

## Rebuild ISCX Arrays

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
