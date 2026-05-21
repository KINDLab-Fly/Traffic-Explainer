# Public Release Checklist

Before pushing this repository to GitHub:

- Choose and add a `LICENSE`.
- Add citation metadata, such as `CITATION.cff`, once the public title and author list are fixed.
- Decide whether checkpoints should be released. If yes, upload them to an artifact host instead of committing them.
- Decide whether processed datasets can be redistributed. If yes, document download URLs and checksums in `docs/DATA.md`.
- Run a fresh smoke test from a clean clone with only `dataset/` and `dataset_localization/` linked locally.
- Confirm `model/`, `res/`, `backend_logs/`, and datasets are not staged by git.
- Update `docs/RESULTS_STATUS.md` with final completed run IDs before reporting numbers.

Recommended smoke test:

```bash
python3 test.py --dataset iscx-vpn --baseline Byte-Transformer --device cuda:0
python3 explain.py --dataset iscx-vpn --baseline Byte-Transformer --explanation random --device cuda:0 --max_samples 8
```
