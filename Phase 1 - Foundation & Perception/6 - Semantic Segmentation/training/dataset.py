"""
dataset.py
==========
PyTorch Dataset for dataset/{images,masks,splits}, with augmentations aimed
at what breaks the classic lane pipeline: lighting changes, shadows, glare
and blur.

Every geometric change is applied identically to image and mask; colour
changes only to the image. Masks are always resized with NEAREST so class
indices never blend into invalid values.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import MEAN, STD


def read_split(root, split):
    path = Path(root) / 'splits' / f'{split}.txt'
    if not path.exists():
        raise FileNotFoundError(f'{path} missing - run tools/split_dataset.py first')
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def find_image(root, item):
    """item = '<recording>/<stem>' -> image path (any extension)."""
    rec, stem = item.split('/', 1)
    matches = sorted((Path(root) / 'images' / rec).glob(stem + '.*'))
    if not matches:
        raise FileNotFoundError(f'No image for {item}')
    return matches[0]


def to_tensor(rgb):
    """HxWx3 uint8 RGB -> normalised 3xHxW float tensor."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class Augmenter:
    def __init__(self, cfg, ignore_index, rng=None):
        self.c = cfg
        self.ignore = ignore_index
        self.rng = rng or np.random.default_rng()

    def __call__(self, img, mask):
        img, mask = self._geometric(img, mask)
        img = self._photometric(img)
        return img, mask

    def _geometric(self, img, mask):
        h, w = mask.shape
        r = self.rng
        s = r.uniform(*self.c['scale'])
        tx = r.uniform(-self.c['shift'], self.c['shift']) * w
        ty = r.uniform(-self.c['shift'], self.c['shift']) * h
        M = np.array([[s, 0, (1 - s) * w / 2 + tx],
                      [0, s, (1 - s) * h / 2 + ty]], dtype=np.float32)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        # pixels that came from outside the image are unknown -> ignore
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore)
        return img, mask

    def _photometric(self, img):
        r, c = self.rng, self.c
        out = img.astype(np.float32)

        # brightness / contrast
        alpha = 1 + r.uniform(-c['contrast'], c['contrast'])
        beta = 255 * r.uniform(-c['brightness'], c['brightness'])
        out = (out - 128) * alpha + 128 + beta

        # small hue shift (yellow must stay yellow)
        if c['hue_shift']:
            hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
            hsv[..., 0] = (hsv[..., 0].astype(int)
                           + r.integers(-c['hue_shift'], c['hue_shift'] + 1)) % 180
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

        h, w = out.shape[:2]
        # shadow: darken a random polygon
        if r.random() < c['shadow_prob']:
            pts = np.stack([r.integers(0, w, 5), r.integers(0, h, 5)], 1).astype(np.int32)
            poly = np.zeros((h, w), np.uint8)
            cv2.fillPoly(poly, [cv2.convexHull(pts)], 1)
            poly = cv2.GaussianBlur(poly.astype(np.float32), (31, 31), 0)
            out *= 1 - r.uniform(0.3, 0.6) * poly[..., None]

        # glare: add a soft bright blob
        if r.random() < c['glare_prob']:
            yy, xx = np.mgrid[0:h, 0:w]
            cx, cy = r.uniform(0, w), r.uniform(0, h)
            rad = r.uniform(0.05, 0.2) * w
            blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad ** 2))
            out += r.uniform(80, 180) * blob[..., None]

        # blur: gaussian or horizontal/vertical motion
        if r.random() < c['blur_prob']:
            k = int(r.choice([3, 5, 7]))
            if r.random() < 0.5:
                out = cv2.GaussianBlur(out, (k, k), 0)
            else:
                kernel = np.zeros((k, k), np.float32)
                kernel[k // 2, :] = 1.0 / k
                if r.random() < 0.5:
                    kernel = kernel.T
                out = cv2.filter2D(out, -1, kernel)

        return np.clip(out, 0, 255).astype(np.uint8)


class SegDataset(Dataset):
    def __init__(self, root, split, width, height, ignore_index=255,
                 augment_cfg=None):
        self.root = root
        self.items = read_split(root, split)
        self.size = (width, height)
        self.ignore = ignore_index
        self.augment = Augmenter(augment_cfg, ignore_index) if augment_cfg else None

    def __len__(self):
        return len(self.items)

    def load_pair(self, i):
        """Resized BGR image and mask, before augmentation/normalisation."""
        item = self.items[i]
        img = cv2.imread(str(find_image(self.root, item)), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(Path(self.root) / 'masks' / f'{item}.png'),
                          cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(f'Could not read image/mask for {item}')
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        return img, mask

    def __getitem__(self, i):
        img, mask = self.load_pair(i)
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return to_tensor(rgb), torch.from_numpy(mask.astype(np.int64))


def class_frequencies(dataset, num_classes):
    """Pixel share of each class over a dataset (no augmentation)."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for i in range(len(dataset)):
        _, mask = dataset.load_pair(i)
        valid = mask[mask < num_classes]
        counts += np.bincount(valid, minlength=num_classes)[:num_classes]
    return counts / max(counts.sum(), 1)
