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
