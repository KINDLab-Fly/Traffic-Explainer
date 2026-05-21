import os
import re
import sys


def iter_logs(log_dir):
    for root, _, files in os.walk(log_dir):
        for name in sorted(files):
            if name.endswith('.log'):
                yield os.path.join(root, name)


def last_metrics_table(text):
    matches = list(re.finditer(r'Budget\tFid\tAcc\tC-Fid\tC-Acc\n((?:[0-9.]+\t[0-9.]+\t[0-9.]+\t[0-9.]+\t[0-9.]+\n?)+)', text))
    if not matches:
        return None
    return matches[-1].group(0).strip()


def classification_accuracy(text):
    lines = [line.rstrip() for line in text.splitlines()]
    for line in reversed(lines):
        parts = line.split()
        if len(parts) >= 5 and parts[0] == 'accuracy':
            return parts[1]
    return None


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: summarize_backend_results.py <backend_log_dir>')

    log_dir = sys.argv[1]
    manifest = os.path.join(log_dir, 'manifest.tsv')
    print(f'Log directory: {log_dir}')
    if os.path.exists(manifest):
        with open(manifest, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]
        done = sum(line.startswith('DONE\t') for line in lines)
        fail = sum(line.startswith('FAIL\t') for line in lines)
        skip = sum(line.startswith('SKIP\t') for line in lines)
        print(f'Jobs: DONE={done} FAIL={fail} SKIP={skip}')
        failed = [line for line in lines if line.startswith('FAIL\t')]
        if failed:
            print('\nFailed jobs:')
            for line in failed:
                print(line)

    print('\nClassification accuracy:')
    for path in iter_logs(log_dir):
        if '/classification/' not in path:
            continue
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        acc = classification_accuracy(text)
        if acc is not None:
            print(f'{os.path.relpath(path, log_dir)}\taccuracy={acc}')

    print('\nExplanation metrics:')
    for path in iter_logs(log_dir):
        if not any(part in path for part in ('/instance/', '/class/', '/table3/')):
            continue
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        table = last_metrics_table(text)
        if table is None:
            continue
        print(f'\n{os.path.relpath(path, log_dir)}')
        print(table)


if __name__ == '__main__':
    main()
