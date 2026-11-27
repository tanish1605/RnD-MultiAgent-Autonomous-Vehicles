#!/usr/bin/env python3
"""
prepare_sign_dataset.py.

Turn a Roboflow YOLO export into a clean dataset for training:

  1. gathers every image from every split folder of the export
  2. fixes labels you decided are wrong (--relabel IMAGE=CLASS), for
     example after reviewing the output of audit_sign_labels.py
  3. drops images you decided are unusable (--drop IMAGE), and, in a polygon
     (segmentation) dataset, images labelled with plain boxes, because
     segmentation training fails on mixed labels (--keep-box-labels to keep)
  4. splits by BLOCKS of consecutive video frames (--block-frames, default
     12) with all their augmented copies, stratified by class, so near-copies
     of a photo never end up in both training and test data; where blocks of
     different splits meet, valid/test frames within --gap frames (default 2)
     of a training frame are left out, because neighbouring frames are almost
     identical
  5. keeps all augmented copies in train, but only one copy per image in
     valid and test (identical copies add nothing to an evaluation)
  6. writes <out>/{train,valid,test}/{images,labels} and data.yaml

Images are linked, not copied, unless --copy is given (use --copy on
Windows or to move the dataset to another computer). Fixed label files are
always written as new files; the export itself is never changed.

Usage (from the object_detection package folder)
-----
  python3 tools/prepare_sign_dataset.py EXPORT_DIR OUT_DIR --dry-run
  python3 tools/prepare_sign_dataset.py EXPORT_DIR OUT_DIR --relabel STOP_00000_jpg=19
"""

import argparse
from collections import Counter, defaultdict
import os
import shutil
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from object_detection.dataset_split import (  # noqa: E402
    collect, mixed_box_images, relabel_lines, stratified_split)

SPLITS = ('train', 'valid', 'test')


def parse_relabel(items):
    out = {}
    for item in items or []:
        src, _, cls = item.partition('=')
        if not cls.isdigit():
            sys.exit(f'--relabel expects IMAGE=CLASS, got {item!r}')
        out[src] = int(cls)
    return out


def load_names(export_dir):
    path = os.path.join(export_dir, 'data.yaml')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        names = yaml.safe_load(f).get('names')
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('export', help='Roboflow export folder (contains train/, maybe valid/, test/)')
    ap.add_argument('out', help='output folder for the clean dataset')
    ap.add_argument('--relabel', nargs='*', metavar='IMAGE=CLASS',
                    help='set the class of every object in these images')
    ap.add_argument('--drop', nargs='*', metavar='IMAGE', help='leave these images out')
    ap.add_argument('--val', type=float, default=0.15)
    ap.add_argument('--test', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--block-frames', type=int, default=12,
                    help='consecutive frames kept together in one split (1 = per image)')
    ap.add_argument('--gap', type=int, default=2,
                    help='leave out valid/test frames this close to a training frame (0 = off)')
    ap.add_argument('--copy', action='store_true', help='copy images instead of linking them')
    ap.add_argument('--keep-box-labels', action='store_true',
                    help='keep box-labelled images in a polygon dataset')
    ap.add_argument('--dry-run', action='store_true', help='show the plan, write nothing')
    args = ap.parse_args()

    samples = collect(args.export)
    if not samples:
        sys.exit(f'No <split>/images folders found in {args.export}')
    relabel = parse_relabel(args.relabel)
    drop = set(args.drop or [])
    unknown = (set(relabel) | drop) - {s.source for s in samples}
    if unknown:
        sys.exit(f'Not in the export: {", ".join(sorted(unknown))}')

    for s in samples:
        if s.source in relabel:
            s.classes = [relabel[s.source]] * len(s.classes)
    box_images = [] if args.keep_box_labels else mixed_box_images(samples)
    if box_images:
        print(f'Leaving out {len(box_images)} images labelled with boxes in a polygon dataset '
              f'(segmentation training fails on mixed labels): {", ".join(box_images)}')
        drop |= set(box_images)
    samples = [s for s in samples if s.source not in drop]
    assignment = stratified_split(samples, args.val, args.test, args.seed,
                                  block_frames=args.block_frames, gap=args.gap)
    edge = sum(1 for v in assignment.values() if v == 'drop')

    names = load_names(args.export)
    n_classes = max([c for s in samples for c in s.classes] + [-1]) + 1
    if names is None:
        names = [f'class_{i}' for i in range(n_classes)]
        print('[warn] no data.yaml in the export: using class_0, class_1, ... as names')

    per_split = defaultdict(Counter)
    files = Counter()
    for s in samples:
        split = assignment[s.source]
        if split == 'drop':
            continue
        copies = s.copies if split == 'train' else s.copies[:1]
        files[split] += len(copies)
        for c in s.classes:
            per_split[split][c] += 1

    print(f'{len(samples)} original images ({sum(len(s.copies) for s in samples)} files), '
          f'{len(relabel)} relabelled, {len(drop)} dropped')
    print(f'split by blocks of {args.block_frames} frames; {edge} valid/test frames left out '
          f'because they are within {args.gap} frames of a training frame')
    print(f'\n{"class":>5} {"name":22s} ' + ' '.join(f'{sp:>6s}' for sp in SPLITS))
    for c in range(n_classes):
        name = names[c] if c < len(names) else '?'
        print(f'{c:>5} {name[:22]:22s} ' + ' '.join(f'{per_split[sp][c]:>6d}' for sp in SPLITS))
    print(f'{"images":>28s} ' + ' '.join(
        f'{sum(1 for s in samples if assignment[s.source] == sp):>6d}' for sp in SPLITS))
    print(f'{"files":>28s} ' + ' '.join(f'{files[sp]:>6d}' for sp in SPLITS))
    if args.dry_run:
        print('\n--dry-run: nothing written')
        return

    for sp in SPLITS:
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(args.out, sp, sub), exist_ok=True)
    for s in samples:
        split = assignment[s.source]
        if split == 'drop':
            continue
        for image, label in (s.copies if split == 'train' else s.copies[:1]):
            dst = os.path.join(args.out, split, 'images', os.path.basename(image))
            if not os.path.exists(dst):
                if args.copy:
                    shutil.copy2(image, dst)
                else:
                    os.symlink(os.path.abspath(image), dst)
            stem = os.path.splitext(os.path.basename(image))[0]
            out_label = os.path.join(args.out, split, 'labels', stem + '.txt')
            lines = open(label, encoding='utf-8').readlines() if label else []
            if s.source in relabel:
                lines = relabel_lines(lines, relabel[s.source])
            with open(out_label, 'w', encoding='utf-8') as f:
                f.writelines(lines)

    with open(os.path.join(args.out, 'data.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump({'path': os.path.abspath(args.out), 'train': 'train/images',
                        'val': 'valid/images', 'test': 'test/images',
                        'nc': len(names), 'names': names}, f, sort_keys=False)
    print(f'\nWrote {args.out} (data.yaml included). Train with: '
          f'yolo train data={os.path.join(args.out, "data.yaml")} ...')


if __name__ == '__main__':
    main()
