#!/usr/bin/env python3
"""
evaluate.py
===========
Score a trained checkpoint on a dataset split (default: test).

Prints per-class IoU, mIoU, pixel accuracy and inference speed on this
machine, and saves side-by-side images (input | ground truth | prediction)
so you can SEE where it fails.

IoU (intersection over union) per class = correctly labelled pixels of that
class / pixels that are that class in truth OR prediction. 1.0 is perfect.
Thin classes (yellow_line, centre_dash) score lower than big areas for the
same visual quality - compare runs class by class, not just by mIoU.

Usage
-----
  python3 evaluate.py runs/20261019_120000/best.pt
  python3 evaluate.py runs/.../best.pt --split val --save 30
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import (colorize, confusion, iou_from_confusion, load_checkpoint,
                    load_config, overlay, pick_device)
from dataset import SegDataset

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint')
    ap.add_argument('--config', default=str(HERE / 'config' / 'train.yaml'),
                    help='only data.root is used from it')
    ap.add_argument('--split', default='test')
    ap.add_argument('--save', type=int, default=20, help='how many comparison images to save')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    names, colors = ckpt['classes'], ckpt['colors']
    nc, ignore = len(names), ckpt['ignore_index']
    w, h = ckpt['input_size']

    root = load_config(args.config)['data']['root']
    ds = SegDataset(root, args.split, w, h, ignore)
    out_dir = Path(args.checkpoint).parent / f'eval_{args.split}'
    out_dir.mkdir(exist_ok=True)

    cm = torch.zeros(nc, nc, dtype=torch.long)
    times = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            x = x.unsqueeze(0).to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = model(x)['out'].argmax(1)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            pred = pred[0].cpu()
            cm += confusion(pred.flatten(), y.flatten(), nc, ignore)

            if i < args.save:
                img, _ = ds.load_pair(i)
                truth = y.numpy().astype(np.uint8)
                truth[truth == ignore] = 0
                p = pred.numpy().astype(np.uint8)
                panel = np.hstack([img, colorize(truth, colors), overlay(img, p, colors)])
                cv2.imwrite(str(out_dir / f'{i:04d}.jpg'), panel)

    iou = iou_from_confusion(cm)
    acc = cm.diag().sum().item() / max(cm.sum().item(), 1)
    print(f'\n{args.split}: {len(ds)} images, input {w}x{h}\n')
    print('class          IoU')
    for name, v in zip(names, iou.tolist()):
        print(f'  {name:12s} {"  n/a" if np.isnan(v) else f"{v:5.3f}"}')
    print(f'\nmIoU           {float(torch.nanmean(iou)):.3f}')
    print(f'pixel accuracy {acc:.3f}')
    if len(times) > 3:
        ms = 1000 * float(np.median(times[3:]))   # skip warm-up
        print(f'speed on {device}: {ms:.1f} ms/frame ({1000 / ms:.1f} FPS) - '
              'PyTorch only; TensorRT on the Jetson will be faster')
    print(f'\nComparison images (input | truth | prediction): {out_dir}')


if __name__ == '__main__':
    main()
