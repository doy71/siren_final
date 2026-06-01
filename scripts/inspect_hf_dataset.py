#!/usr/bin/env python
import argparse
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--name', default=None)
    p.add_argument('--split', default='train')
    p.add_argument('--limit', type=int, default=3)
    args = p.parse_args()
    ds = load_dataset(args.dataset, args.name, split=args.split) if args.name else load_dataset(args.dataset, split=args.split)
    print(ds)
    print('columns:', ds.column_names)
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
