#!/usr/bin/env python3
"""
convert_masks.py
================
Turn exported labels into class-index masks: a single-channel PNG per
image where each pixel value is the class index from
training/config/train.yaml (0 = background, 1 = drivable, ...).

Supported exports
-----------------
cvat      CVAT "Segmentation mask 1.1": SegmentationClass/*.png colour masks
          plus labelmap.txt ("name:r,g,b::"). Colours are mapped by NAME,
          so the CVAT colours do not need to match ours.
roboflow  Roboflow "Semantic Segmentation Masks": *_mask.png already holds
          indices, plus _classes.csv ("Pixel Value, Class"). Re-mapped by
          name to our order.

Labels not listed in train.yaml are reported and set to ignore_index.

Usage
-----
  python3 convert_masks.py cvat     --src cvat_export/ --out dataset/masks/run_01
  python3 convert_masks.py roboflow --src rf_export/train --out dataset/masks/run_01
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

DEFAULT_CFG = Path(__file__).resolve().parent.parent / 'training' / 'config' / 'train.yaml'


def load_classes(cfg_path):
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    names = [c['name'] for c in cfg['classes']]
    return {n: i for i, n in enumerate(names)}, cfg.get('ignore_index', 255)


def norm(name):
    return name.strip().lower().replace(' ', '_').replace('-', '_')


def convert_cvat(src, out, class_idx, ignore):
    labelmap = {}
    with open(Path(src) / 'labelmap.txt', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or ':' not in line:
                continue
            name, rgb = line.split(':')[:2]
            r, g, b = (int(v) for v in rgb.split(','))
            labelmap[norm(name)] = (b, g, r)            # OpenCV loads BGR

    unknown = sorted(n for n in labelmap if n not in class_idx)
    if unknown:
        print(f'[warn] CVAT labels not in train.yaml -> ignored: {unknown}')

    files = sorted((Path(src) / 'SegmentationClass').glob('*.png'))
    for p in files:
        color = cv2.imread(str(p), cv2.IMREAD_COLOR)
        mask = np.full(color.shape[:2], ignore, dtype=np.uint8)
        for name, bgr in labelmap.items():
            hit = np.all(color == np.array(bgr, dtype=np.uint8), axis=-1)
            mask[hit] = class_idx.get(name, ignore)
        cv2.imwrite(str(Path(out) / p.name), mask)
    return len(files)


def convert_roboflow(src, out, class_idx, ignore):
    lut = np.full(256, ignore, dtype=np.uint8)
    with open(Path(src) / '_classes.csv', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    unknown = []
    for row in rows[1:]:
        value, name = int(row[0]), norm(row[1])
        if name in class_idx:
            lut[value] = class_idx[name]
        else:
            unknown.append(name)
    if unknown:
        print(f'[warn] Roboflow classes not in train.yaml -> ignored: {unknown}')

    files = sorted(Path(src).glob('*_mask.png'))
    for p in files:
        mask = lut[cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)]
        stem = p.name[:-len('_mask.png')]
        cv2.imwrite(str(Path(out) / f'{stem}.png'), mask)
    return len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('format', choices=['cvat', 'roboflow'])
    ap.add_argument('--src', required=True, help='export folder')
    ap.add_argument('--out', required=True, help='output mask folder')
    ap.add_argument('--config', default=str(DEFAULT_CFG), help='train.yaml with class list')
    args = ap.parse_args()

    class_idx, ignore = load_classes(args.config)
    class_idx = {norm(k): v for k, v in class_idx.items()}
    os.makedirs(args.out, exist_ok=True)

    fn = convert_cvat if args.format == 'cvat' else convert_roboflow
    n = fn(args.src, args.out, class_idx, ignore)
    if n == 0:
        sys.exit('No mask files found - check --src and the export format.')
    print(f'Converted {n} masks -> {args.out}')
    print('Mask file names must match image names (same stem, .png) for training.')


if __name__ == '__main__':
    main()
