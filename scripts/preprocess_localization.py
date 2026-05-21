import argparse
import os
import pickle
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


DATASETS = {
    'ios': {
        'directory': 'IOS_Cross_Plat',
        'csv': 'Cross_Platform_iOS_pkt.csv',
        'label_column': 'Country_Detection',
    },
    'android': {
        'directory': 'Android_Cross_Plat',
        'csv': 'Cross_Platform_Android_pkt.csv',
        'label_column': 'Country_Detection',
    },
}


def hex_to_int_list(hex_str):
    return [int(value, 16) for value in hex_str.split()]


def transform(hex_string):
    if '<head>' not in hex_string or '<pkt>' not in hex_string:
        raise ValueError('Expected flow text to contain both <head> and <pkt> markers')

    before_head, after_head = hex_string.split('<head>', 1)
    packet_text, _ = after_head.split('<pkt>', 1)
    return hex_to_int_list(before_head.strip()), hex_to_int_list(packet_text.strip())


def pad_or_truncate(values, max_len, pad_value):
    output = []
    for row in values:
        row = list(row)
        if len(row) < max_len:
            row = row + [pad_value] * (max_len - len(row))
        else:
            row = row[:max_len]
        output.append(row)
    return output


def label_mapping(labels):
    mapping = OrderedDict()
    for label in labels:
        if label not in mapping:
            mapping[label] = len(mapping)
    return mapping


def split_non_train(non_train_data, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(non_train_data['label']))
    rng.shuffle(indices)
    split_at = len(indices) // 2
    return indices[:split_at], indices[split_at:]


def select_rows(data, indices):
    selected = defaultdict(list)
    for key, values in data.items():
        selected[key] = [values[i] for i in indices]
    return selected


def dump_pickle(path, data):
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'wb') as handle:
        pickle.dump(dict(data), handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def summarize(split_name, data):
    counts = dict(sorted(Counter(data['label']).items()))
    pkt_shape = (len(data['pkt']), len(data['pkt'][0]) if data['pkt'] else 0)
    head_shape = (len(data['head']), len(data['head'][0]) if data['head'] else 0)
    print(
        f'{split_name}: n={len(data["label"])}, pkt_shape={pkt_shape}, '
        f'head_shape={head_shape}, label_counts={counts}'
    )


def preprocess_dataset(dataset_key, root, seed, max_len, pad_value, split_strategy, overwrite):
    spec = DATASETS[dataset_key]
    dataset_dir = os.path.join(root, spec['directory'])
    csv_path = os.path.join(dataset_dir, spec['csv'])

    output_paths = [
        os.path.join(dataset_dir, 'train.pkl'),
        os.path.join(dataset_dir, 'val.pkl'),
        os.path.join(dataset_dir, 'test.pkl'),
        os.path.join(dataset_dir, 'country2id.pkl'),
    ]
    if not overwrite:
        existing = [path for path in output_paths if os.path.exists(path)]
        if existing:
            raise FileExistsError(
                'Output files already exist. Use --overwrite to regenerate: '
                + ', '.join(existing)
            )

    dataframe = pd.read_csv(
        csv_path,
        usecols=['text', 'dataset_type', spec['label_column']],
    )
    country2id = label_mapping(dataframe[spec['label_column']])

    train_data = defaultdict(list)
    val_data = defaultdict(list)
    test_data = defaultdict(list)
    non_train_data = defaultdict(list)
    observed_max_len = 0

    rows = dataframe.itertuples(index=False)
    for row in tqdm(rows, total=len(dataframe), desc=f'preprocess {dataset_key}'):
        flow = getattr(row, 'text')
        split = getattr(row, 'dataset_type')
        country = getattr(row, spec['label_column'])
        head, pkt = transform(flow)
        observed_max_len = max(observed_max_len, len(head), len(pkt))

        if split == 'train':
            target = train_data
        elif split_strategy == 'source':
            if split in {'dev', 'val', 'validation'}:
                target = val_data
            elif split == 'test':
                target = test_data
            else:
                raise ValueError(f'Unknown source split: {split}')
        else:
            target = non_train_data
        target['label'].append(country2id[country])
        target['head'].append(head)
        target['pkt'].append(pkt)

    target_len = observed_max_len if max_len == 'auto' else int(max_len)
    for data in (train_data, val_data, test_data, non_train_data):
        if not data:
            continue
        data['head'] = pad_or_truncate(data['head'], target_len, pad_value)
        data['pkt'] = pad_or_truncate(data['pkt'], target_len, pad_value)

    if split_strategy == 'combined-random':
        val_indices, test_indices = split_non_train(non_train_data, seed)
        val_data = select_rows(non_train_data, val_indices)
        test_data = select_rows(non_train_data, test_indices)

    dump_pickle(os.path.join(dataset_dir, 'train.pkl'), train_data)
    dump_pickle(os.path.join(dataset_dir, 'val.pkl'), val_data)
    dump_pickle(os.path.join(dataset_dir, 'test.pkl'), test_data)
    dump_pickle(os.path.join(dataset_dir, 'country2id.pkl'), dict(country2id))

    print(f'dataset={dataset_key}')
    print(f'csv={csv_path}')
    print(f'split_strategy={split_strategy}, seed={seed}')
    print(f'country2id={dict(country2id)}')
    print(f'observed_max_len={observed_max_len}, written_max_len={target_len}')
    summarize('train', train_data)
    summarize('val', val_data)
    summarize('test', test_data)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Traffic-Explainer localization datasets.')
    parser.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    parser.add_argument('--root', default='dataset_localization')
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--max_len', default='64', help='Use an integer length or "auto".')
    parser.add_argument('--pad_value', type=int, default=65536)
    parser.add_argument(
        '--split_strategy',
        choices=['combined-random', 'source'],
        default='combined-random',
        help='combined-random matches the original process.py logic; source maps CSV dev/test directly.',
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    preprocess_dataset(
        dataset_key=args.dataset,
        root=args.root,
        seed=args.seed,
        max_len=args.max_len,
        pad_value=args.pad_value,
        split_strategy=args.split_strategy,
        overwrite=args.overwrite,
    )
