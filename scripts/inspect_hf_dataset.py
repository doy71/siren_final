#!/usr/bin/env python
import argparse
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--name', default=None, help='HF dataset config/subset name, e.g. toxicchat0124')
    p.add_argument('--split', default=None, help='Split to inspect. If omitted, only configs/splits are listed when possible.')
    p.add_argument('--limit', type=int, default=3)
    args = p.parse_args()

    try:
        print('configs:', get_dataset_config_names(args.dataset))
    except Exception as e:
        print('configs: <unavailable>', type(e).__name__, e)
    try:
        print('splits:', get_dataset_split_names(args.dataset, args.name) if args.name else get_dataset_split_names(args.dataset))
    except Exception as e:
        print('splits: <unavailable>', type(e).__name__, e)

    if args.split is None:
        return

    ds = load_dataset(args.dataset, args.name, split=args.split) if args.name else load_dataset(args.dataset, split=args.split)
    print(ds)
    print('columns:', ds.column_names)
    print('features:', ds.features)
    for i in range(min(args.limit, len(ds))):
        print('\n--- row', i, '---')
        row = ds[i]
        for k, v in row.items():
            s = repr(v)
            if len(s) > 600:
                s = s[:600] + ' ...'
            print(f'{k}: {s}')


if __name__ == '__main__':
    main()
