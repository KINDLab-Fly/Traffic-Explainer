#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?Usage: scripts/splitcap_iscx.sh <iscx-vpn|iscx-nonvpn|iscx-tor|iscx-nontor> [dataset_root]}"
dataset_root="${2:-dataset}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 scripts/preprocess_iscx.py \
  --dataset "$dataset" \
  --root "$dataset_root" \
  --stage splitcap \
  --overwrite
