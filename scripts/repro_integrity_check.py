import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import get_config
from dataset import TrafficDataset
from explain import load_models, predict_logits


ISCX_DATASETS = ['iscx-vpn', 'iscx-nonvpn', 'iscx-tor', 'iscx-nontor']
LOCALIZATION_DATASETS = ['ios', 'android']
ISCX_BASELINES = [
    'Byte-Transformer',
]


def check_forward(dataset_name, baseline, device):
    config = get_config(dataset_name)
    config.baseline = baseline
    config.device = device

    models = load_models(config)
    dataset = Subset(TrafficDataset(config.dataset, 'test', config.PAD_TRUNC_DIGIT), range(2))
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    seq, label, mask, seq_header, seq_header_mask = [
        item.to(config.device) if torch.is_tensor(item) else item for item in batch
    ]
    with torch.no_grad():
        logits = predict_logits(*models, seq, mask, seq_header, seq_header_mask, config)

    expected = (2, config.class_size)
    if tuple(logits.shape) != expected:
        raise RuntimeError(f'{dataset_name}/{baseline}: expected logits {expected}, got {tuple(logits.shape)}')

    print(
        f'OK\t{dataset_name}\t{baseline}\t'
        f'dataset={config.dataset}\tvocab={config.seq_vocab_size}\t'
        f'logits={tuple(logits.shape)}\tpred={logits.argmax(dim=1).detach().cpu().tolist()}\t'
        f'label={label.detach().cpu().tolist()}'
    )


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else ('cuda:0' if torch.cuda.is_available() else 'cpu')

    for dataset_name in ISCX_DATASETS:
        for baseline in ISCX_BASELINES:
            check_forward(dataset_name, baseline, device)

    for dataset_name in LOCALIZATION_DATASETS:
        check_forward(dataset_name, 'Byte-Transformer', device)


if __name__ == '__main__':
    main()
