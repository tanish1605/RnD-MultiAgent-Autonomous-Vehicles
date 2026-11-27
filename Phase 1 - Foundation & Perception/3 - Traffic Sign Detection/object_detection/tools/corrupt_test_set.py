#!/usr/bin/env python3
"""
corrupt_test_set.py.

Job C (robustness test): make corrupted copies of a dataset's test images
and, optionally, evaluate every model on each copy.

For every corruption (dark, low_contrast, glare, motion_blur, noise) and
severity (1 mild .. 3 strong) it writes
    OUT/<corruption>_s<severity>/images/   corrupted test images
    OUT/<corruption>_s<severity>/labels/   the original labels (still valid)
    OUT/<corruption>_s<severity>.yaml      a data file for `yolo val`
plus OUT/clean.yaml for the unchanged test images, as the reference.

With --models it also runs `yolo val` (Ultralytics) for every model on every
copy and writes OUT/robustness.csv: box and mask mAP per model, corruption
and severity, and the drop against the clean test images. Without --models
it prints the commands to run.

Usage (from the object_detection package folder)
-----
  python3 tools/corrupt_test_set.py clean_signs/data.yaml robustness
  python3 tools/corrupt_test_set.py clean_signs/data.yaml robustness --models a/best.pt b/best.pt
"""

import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from object_detection.robustness import corrupt, CORRUPTIONS  # noqa: E402

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def split_dirs(data_yaml, split):
    """Images and labels folder of one split of a YOLO data.yaml."""
    with open(data_yaml, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    root = data.get('path') or os.path.dirname(os.path.abspath(data_yaml))
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(data_yaml)), root)
    images = os.path.join(root, data[split])
    labels = os.path.join(os.path.dirname(images), 'labels')
    return data, images, labels


def write_yaml(path, images_dir, names):
    root = os.path.dirname(os.path.abspath(images_dir))
    sub = os.path.basename(images_dir)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump({'path': root, 'train': sub, 'val': sub, 'test': sub,
                        'names': names}, f, sort_keys=False)


def make_copies(images, labels, out, names, corruptions, severities, seed):
    """Write all corrupted copies; returns [(corruption, severity, yaml path)]."""
    files = sorted(f for f in os.listdir(images) if f.lower().endswith(IMAGE_EXTS))
    if not files:
        sys.exit(f'No images in {images}')
    os.makedirs(out, exist_ok=True)
    clean_yaml = os.path.join(out, 'clean.yaml')
    write_yaml(clean_yaml, images, names)
    sets = [('clean', 0, clean_yaml)]
    for name in corruptions:
        for sev in severities:
            folder = os.path.join(out, f'{name}_s{sev}')
            os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
            os.makedirs(os.path.join(folder, 'labels'), exist_ok=True)
            for i, f in enumerate(files):
                # same seed per image and severity: reproducible, varied across images
                rng = np.random.default_rng([seed, i, sev, CORRUPTIONS.index(name)])
                img = cv2.imread(os.path.join(images, f), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                cv2.imwrite(os.path.join(folder, 'images', f), corrupt(img, name, sev, rng))
                label = os.path.join(labels, os.path.splitext(f)[0] + '.txt')
                if os.path.exists(label):
                    shutil.copy2(label, os.path.join(folder, 'labels'))
            path = os.path.join(out, f'{name}_s{sev}.yaml')
            write_yaml(path, os.path.join(folder, 'images'), names)
            sets.append((name, sev, path))
            print(f'  {name}_s{sev}: {len(files)} images')
    return sets


def evaluate(models, sets, out, imgsz):
    from ultralytics import YOLO
    rows = []
    for model_path in models:
        model = YOLO(model_path)
        clean = None
        for name, sev, data in sets:
            r = model.val(data=data, split='test', imgsz=imgsz, verbose=False, plots=False,
                          project=os.path.join(os.path.abspath(out), 'val_runs'),
                          name=f'{os.path.basename(os.path.dirname(os.path.dirname(model_path)))}'
                               f'_{name}_s{sev}', exist_ok=True)
            seg = getattr(r, 'seg', None)
            row = {'model': model_path, 'corruption': name, 'severity': sev,
                   'box_map50': r.box.map50, 'box_map50_95': r.box.map,
                   'mask_map50': seg.map50 if seg is not None else '',
                   'mask_map50_95': seg.map if seg is not None else ''}
            if name == 'clean':
                clean = row
            key = 'mask_map50_95' if seg is not None else 'box_map50_95'
            row['drop_vs_clean'] = (clean[key] - row[key]) if clean else ''
            rows.append(row)
            drop = f'  drop {row["drop_vs_clean"]:+.3f}' if name != 'clean' else ''
            print(f'  {os.path.basename(model_path)} {name:12s} s{sev}  '
                  f'{key} {row[key]:.3f}{drop}')
    path = os.path.join(out, 'robustness.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nWrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', help='data.yaml of the clean dataset (from prepare_sign_dataset.py)')
    ap.add_argument('out', help='output folder for the corrupted copies')
    ap.add_argument('--split', default='test', help='split to corrupt (default: test)')
    ap.add_argument('--corruptions', nargs='+', default=list(CORRUPTIONS), choices=CORRUPTIONS)
    ap.add_argument('--severities', nargs='+', type=int, default=[1, 2, 3], choices=[1, 2, 3])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--models', nargs='*', help='best.pt files to evaluate on every copy')
    ap.add_argument('--imgsz', type=int, default=640)
    args = ap.parse_args()

    data, images, labels = split_dirs(args.data, args.split)
    print(f'Corrupting {images}')
    sets = make_copies(images, labels, args.out, data['names'], args.corruptions,
                       args.severities, args.seed)
    if args.models:
        evaluate(args.models, sets, args.out, args.imgsz)
    else:
        print('\nEvaluate a model on every copy with, for each yaml in the output folder:')
        print(f'  yolo val model=<best.pt> data={args.out}/<name>.yaml split=test '
              f'imgsz={args.imgsz}')
        print('or rerun this command with --models <best.pt> ... to do all of it at once.')


if __name__ == '__main__':
    main()
