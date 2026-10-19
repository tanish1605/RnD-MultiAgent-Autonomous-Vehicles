#!/usr/bin/env python3
"""
split_dataset.py
================
Write train / val / test lists for the labelled dataset.

Expected layout (made by extract_frames.py + convert_masks.py):

    dataset/
      images/<recording>/<frame>.jpg
      masks/<recording>/<frame>.png     <- same stem as the image
      splits/{train,val,test}.txt       <- written by this script

Only images that have a mask are used. Each list line is
"<recording>/<frame stem>".

Why split by recording?
-----------------------
Consecutive frames of one drive are almost identical. A random per-frame
split puts near-copies in train AND test, so test scores look far better
than real performance. Whole recordings are therefore assigned to one
split. With few recordings (< --min-recordings) that gives badly skewed
splits, so each recording is cut into contiguous chunks (--chunk frames)
and chunks are assigned instead - a weaker but still honest split.

Usage
-----
  python3 split_dataset.py --root dataset --val 0.15 --test 0.15
"""

import argparse
import random
import sys
from pathlib import Path


def collect(root):
    """{recording: [stem, ...]} for images that have a mask."""
    groups = {}
    for mask in sorted((root / 'masks').glob('*/*.png')):
        rec = mask.parent.name
        imgs = list((root / 'images' / rec).glob(mask.stem + '.*'))
        if imgs:
            groups.setdefault(rec, []).append(mask.stem)
        else:
            print(f'[warn] mask without image: {mask}')
    return groups


def assign(groups, val, test, seed):
    """Greedy: biggest groups first, each to the split furthest below target."""
    total = sum(len(v) for v in groups.values())
    target = {'train': (1 - val - test) * total, 'val': val * total, 'test': test * total}
    have = {k: 0 for k in target}
    out = {k: [] for k in target}

    keys = list(groups)
    random.Random(seed).shuffle(keys)                 # tie-break randomly
    for key in sorted(keys, key=lambda k: len(groups[k]), reverse=True):
        split = max(target, key=lambda s: (target[s] - have[s]) / max(target[s], 1))
        out[split].extend(groups[key])
        have[split] += len(groups[key])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='dataset')
    ap.add_argument('--val', type=float, default=0.15)
    ap.add_argument('--test', type=float, default=0.15)
    ap.add_argument('--chunk', type=int, default=40,
                    help='frames per chunk when there are too few recordings')
    ap.add_argument('--min-recordings', type=int, default=5,
                    help='split by whole recording only if at least this many')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    by_rec = collect(root)
    if not by_rec:
        sys.exit(f'No image/mask pairs under {root}/images and {root}/masks')

    if len(by_rec) >= args.min_recordings:
        groups = {rec: [f'{rec}/{s}' for s in stems] for rec, stems in by_rec.items()}
        mode = 'by recording'
    else:
        groups = {}
        for rec, stems in by_rec.items():
            for i in range(0, len(stems), args.chunk):
                groups[f'{rec}#{i}'] = [f'{rec}/{s}' for s in stems[i:i + args.chunk]]
        mode = f'by {args.chunk}-frame chunks (only {len(by_rec)} recording(s))'
        print(f'[warn] fewer than {args.min_recordings} recordings - splitting {mode}. '
              'Record more drives for a trustworthy test score.')

    splits = assign(groups, args.val, args.test, args.seed)
    (root / 'splits').mkdir(parents=True, exist_ok=True)
    total = sum(len(v) for v in splits.values())
    print(f'Split {mode}:')
    for name, items in splits.items():
        (root / 'splits' / f'{name}.txt').write_text(
            ''.join(f'{x}\n' for x in sorted(items)), encoding='utf-8')
        print(f'  {name:5s} {len(items):5d}  ({100 * len(items) / total:4.1f} %)')
        if not items:
            print(f'  [warn] {name} split is empty - add more data')


if __name__ == '__main__':
    main()
