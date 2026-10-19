"""
common.py
=========
Shared by train.py, evaluate.py and export_onnx.py: config loading, model
construction, device choice, metrics and overlay drawing.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import (deeplabv3_mobilenet_v3_large,
                                             lraspp_mobilenet_v3_large)

# ImageNet statistics - the pretrained backbone expects inputs normalised so.
# Stored in every checkpoint / ONNX file so the ROS node uses the same values.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

ARCHS = {
    'lraspp_mobilenet_v3_large': lraspp_mobilenet_v3_large,       # fastest
    'deeplabv3_mobilenet_v3_large': deeplabv3_mobilenet_v3_large,  # more accurate, slower
}


def load_config(path):
    """Load train.yaml; relative paths in it are resolved against training/."""
    path = Path(path).resolve()
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    base = path.parent.parent                       # training/
    for key in ('root',):
        cfg['data'][key] = str((base / cfg['data'][key]).resolve())
    cfg['train']['out_dir'] = str((base / cfg['train']['out_dir']).resolve())
    return cfg


def build_model(arch, num_classes, pretrained_backbone=False):
    if arch not in ARCHS:
        raise ValueError(f'Unknown arch {arch}; choose from {list(ARCHS)}')
    kwargs = {'weights': None, 'num_classes': num_classes}
    if pretrained_backbone:
        kwargs['weights_backbone'] = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    else:
        kwargs['weights_backbone'] = None
    if arch == 'deeplabv3_mobilenet_v3_large':
        kwargs['aux_loss'] = False
    return ARCHS[arch](**kwargs)


class LogitsOnly(torch.nn.Module):
    """torchvision seg models return {'out': logits}; ONNX wants a tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['out']


def pick_device(name='auto'):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_checkpoint(path, device='cpu'):
    """Return (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt['arch'], len(ckpt['classes']))
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval(), ckpt


# ── metrics ────────────────────────────────────────────────────────────────

def confusion(pred, target, num_classes, ignore_index):
    """Accumulate a num_classes x num_classes confusion matrix (rows = truth)."""
    keep = target != ignore_index
    idx = target[keep] * num_classes + pred[keep]
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def iou_from_confusion(cm):
    """Per-class IoU (NaN for classes absent from both truth and prediction)."""
    cm = cm.double()
    tp = cm.diag()
    union = cm.sum(0) + cm.sum(1) - tp
    iou = tp / union
    iou[union == 0] = float('nan')
    return iou


# ── visualisation ──────────────────────────────────────────────────────────

def colorize(mask, colors):
    """Class-index mask (H, W) -> BGR image using the per-class colours."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:len(colors)] = np.array(colors, dtype=np.uint8)
    return lut[mask]


def overlay(bgr, mask, colors, alpha=0.5):
    color = colorize(mask, colors)
    out = bgr.copy()
    painted = mask > 0                               # leave background untouched
    out[painted] = cv2.addWeighted(bgr, 1 - alpha, color, alpha, 0)[painted]
    return out
