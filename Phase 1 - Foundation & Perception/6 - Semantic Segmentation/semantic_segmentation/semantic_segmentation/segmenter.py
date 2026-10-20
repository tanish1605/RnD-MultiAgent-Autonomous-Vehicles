#!/usr/bin/env python3
"""
segmenter.py.

Semantic segmentation model wrapper.

Responsibilities
----------------
1. Load a model exported by ../training (PyTorch .pt checkpoint or .onnx)
2. Read class names / colours / input size / normalisation from the model
   file itself, so the node needs no separate class configuration
3. segment(bgr) -> class-index mask at the input image resolution

No ROS code should exist here (same rule as object_detection/detector.py).
"""

import json
from pathlib import Path

import cv2
import numpy as np


class Segmenter:
    """Run a semantic segmentation model on BGR images."""

    def __init__(self, model_path, device='auto'):
        """
        Load the model.

        model_path : .pt checkpoint (from train.py) or .onnx (from export_onnx.py)
        device     : 'auto', 'cuda', 'cpu' (PyTorch) - for ONNX, 'auto'/'cuda'
                     try the CUDA execution provider first
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f'Model not found: {self.model_path}')

        suffix = self.model_path.suffix.lower()
        if suffix == '.pt':
            self._load_torch(device)
        elif suffix == '.onnx':
            self._load_onnx(device)
        else:
            raise ValueError(f'Unsupported model type {suffix} (use .pt or .onnx)')

        self.input_w, self.input_h = (int(v) for v in self.meta['input_size'])
        self.class_names = list(self.meta['classes'])
        self.colors = [tuple(int(c) for c in col) for col in self.meta['colors']]
        self._mean = np.array(self.meta['mean'], dtype=np.float32)
        self._std = np.array(self.meta['std'], dtype=np.float32)
        lut = np.zeros((256, 3), dtype=np.uint8)
        lut[:len(self.colors)] = np.array(self.colors, dtype=np.uint8)
        self._color_lut = lut

    # ── loading ──────────────────────────────────────────────────────────

    def _load_torch(self, device):
        import torch
        from torchvision.models import segmentation

        ckpt = torch.load(self.model_path, map_location='cpu', weights_only=False)
        kwargs = {'weights': None, 'weights_backbone': None,
                  'num_classes': len(ckpt['classes'])}
        if ckpt['arch'].startswith('deeplabv3'):
            kwargs['aux_loss'] = False
        model = getattr(segmentation, ckpt['arch'])(**kwargs)
        model.load_state_dict(ckpt['state_dict'])

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._torch = torch
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        self.meta = {k: ckpt[k] for k in
                     ('arch', 'classes', 'colors', 'input_size', 'mean', 'std')}
        self.backend = f'torch ({self._device})'

    def _load_onnx(self, device):
        import onnxruntime as ort

        providers = ['CPUExecutionProvider']
        if device in ('auto', 'cuda') and \
                'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

        props = self._session.get_modelmeta().custom_metadata_map
        if 'classes' in props:
            self.meta = {k: json.loads(v) for k, v in props.items()}
        else:   # fall back to the .yaml written next to the .onnx
            import yaml
            with open(self.model_path.with_suffix('.yaml'), encoding='utf-8') as f:
                self.meta = yaml.safe_load(f)
        self.backend = f'onnxruntime ({self._session.get_providers()[0]})'

    # ── inference ────────────────────────────────────────────────────────

    def preprocess(self, bgr):
        """BGR uint8 image -> normalised 1x3xHxW float32 array."""
        img = cv2.resize(bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (rgb - self._mean) / self._std
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def infer(self, x):
        """Run the network: 1x3xHxW input -> HxW class-index mask (uint8)."""
        if hasattr(self, '_session'):
            logits = self._session.run(None, {self._input_name: x})[0]
            return logits[0].argmax(0).astype(np.uint8)
        torch = self._torch
        with torch.inference_mode():
            t = torch.from_numpy(x).to(self._device)
            pred = self._model(t)['out'][0].argmax(0)
        return pred.to(torch.uint8).cpu().numpy()

    def segment(self, bgr, full_resolution=True):
        """
        Segment one BGR image.

        Returns a uint8 mask whose pixel values are class indices. With
        full_resolution=True it has the same size as `bgr`, else the model's
        input size (cheaper to publish; scale coordinates yourself).
        """
        mask = self.infer(self.preprocess(bgr))
        if full_resolution and mask.shape != bgr.shape[:2]:
            mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        return mask

    # ── visualisation ────────────────────────────────────────────────────

    def overlay(self, bgr, mask, alpha=0.5):
        """Blend class colours over the image; background (0) is left as-is."""
        if mask.shape != bgr.shape[:2]:
            mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        color = self._color_lut[mask]
        out = bgr.copy()
        painted = mask > 0
        out[painted] = cv2.addWeighted(bgr, 1 - alpha, color, alpha, 0)[painted]
        return out
