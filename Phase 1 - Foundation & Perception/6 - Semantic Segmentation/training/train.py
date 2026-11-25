#!/usr/bin/env python3
"""
train.py
========
Train the semantic segmentation model described in config/train.yaml.

Each epoch: train on the `train` split, measure mIoU on the `val` split,
keep the best checkpoint. Output in runs/<timestamp>/:
    best.pt    highest val mIoU  <- use this for evaluate / export
    last.pt    final epoch
    log.csv    epoch, train loss, val mIoU, per-class IoU

The checkpoint stores everything the ROS node needs besides the weights:
class names and colours, input size, normalisation (mean/std).

Usage
-----
  python3 train.py                              # uses config/train.yaml
  python3 train.py --config config/train.yaml --epochs 5   # quick test
"""

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (MEAN, STD, build_model, confusion, iou_from_confusion,
                    load_config, pick_device)
from dataset import SegDataset, class_frequencies

HERE = Path(__file__).resolve().parent


def make_class_weights(train_set, num_classes, mode):
    """'auto': inverse-sqrt pixel frequency, so thin lines are not ignored."""
    if mode == 'none':
        return None
    freq = class_frequencies(train_set, num_classes)
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-4))
    w = w / w[freq > 0].mean()                    # average weight ~1
    w = np.clip(w, 0.2, 10.0)
    print('Class pixel share :', ' '.join(f'{f:.3f}' for f in freq))
    print('Class weights     :', ' '.join(f'{x:.2f}' for x in w))
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def validate(model, loader, num_classes, ignore, device):
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        pred = model(x.to(device))['out'].argmax(1).cpu()
        cm += confusion(pred.flatten(), y.flatten(), num_classes, ignore)
    return iou_from_confusion(cm)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(HERE / 'config' / 'train.yaml'))
    ap.add_argument('--epochs', type=int, help='override train.epochs')
    ap.add_argument('--no-pretrained', action='store_true',
                    help='skip downloading ImageNet backbone weights')
    args = ap.parse_args()

    cfg = load_config(args.config)
    d, t, m = cfg['data'], cfg['train'], cfg['model']
    classes = cfg['classes']
    nc, ignore = len(classes), cfg['ignore_index']
    epochs = args.epochs or t['epochs']

    random.seed(t['seed'])
    np.random.seed(t['seed'])
    torch.manual_seed(t['seed'])
    device = pick_device(t['device'])
    print(f'Device: {device}')

    train_set = SegDataset(d['root'], 'train', d['input_width'], d['input_height'],
                           ignore, augment_cfg=cfg['augment'])
    val_set = SegDataset(d['root'], 'val', d['input_width'], d['input_height'], ignore)
    print(f'Train: {len(train_set)} images   Val: {len(val_set)} images')

    loader_kw = dict(num_workers=t['num_workers'], pin_memory=device.type == 'cuda',
                     persistent_workers=t['num_workers'] > 0)
    train_loader = DataLoader(train_set, batch_size=t['batch_size'], shuffle=True,
                              drop_last=len(train_set) > t['batch_size'], **loader_kw)
    val_loader = DataLoader(val_set, batch_size=t['batch_size'], **loader_kw)

    model = build_model(m['arch'], nc,
                        m['pretrained_backbone'] and not args.no_pretrained).to(device)
    # torchvision's MobileNetV3 uses BatchNorm momentum 0.01: with a small
    # dataset (few hundred steps) the running stats used at inference lag far
    # behind and eval-mode output collapses to "background". 0.1 = PyTorch default.
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.momentum = 0.1
    weights = make_class_weights(train_set, nc, t['class_weights'])
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        ignore_index=ignore)
    opt = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    steps = epochs * max(len(train_loader), 1)
    # cosine decay with a short linear warm-up
    warm = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))

    run_dir = Path(t['out_dir']) / time.strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        'arch': m['arch'],
        'classes': [c['name'] for c in classes],
        'colors': [c['color'] for c in classes],
        'input_size': [d['input_width'], d['input_height']],   # [w, h]
        'mean': MEAN, 'std': STD,
        'ignore_index': ignore,
    }
    names = meta['classes']
    with open(run_dir / 'log.csv', 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_miou'] + names)

    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x)['out'], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * x.size(0)
            n += x.size(0)

        iou = validate(model, val_loader, nc, ignore, device)
        miou = float(torch.nanmean(iou)) if not torch.isnan(iou).all() else 0.0
        per_class = ' '.join(f'{nm}={v:.2f}' for nm, v in zip(names, iou.tolist()))
        print(f'epoch {epoch:3d}/{epochs}  loss {total / max(n, 1):.4f}  '
              f'val mIoU {miou:.3f}  [{per_class}]  {time.time() - t0:.0f}s')

        with open(run_dir / 'log.csv', 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch, f'{total / max(n, 1):.5f}', f'{miou:.4f}']
                                   + [f'{v:.4f}' for v in iou.tolist()])

        ckpt = {**meta, 'state_dict': model.state_dict(),
                'epoch': epoch, 'val_miou': miou}
        torch.save(ckpt, run_dir / 'last.pt')
        if miou > best:
            best = miou
            torch.save(ckpt, run_dir / 'best.pt')

    print(f'\nDone. Best val mIoU {best:.3f}  ->  {run_dir / "best.pt"}')


if __name__ == '__main__':
    main()
