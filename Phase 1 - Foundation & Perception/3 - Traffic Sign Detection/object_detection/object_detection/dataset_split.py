#!/usr/bin/env python3

"""
dataset_split.py.

Clean and re-split a YOLO dataset exported from Roboflow, keeping every
augmented copy of an image in the same split.

Two kinds of near-copies would otherwise leak between training and test data
and make the test score look better than the model is:

- augmented copies: Roboflow exports each original image several times
  (names differ only in '.rf.<hash>') when augmentation is on;
- neighbouring video frames: images named <prefix>_<frame number>_jpg are
  consecutive frames of one recording and look almost identical.

The split is therefore made per block of consecutive frames (block_frames
frames of one prefix; all augmented copies included), stratified by class
so that every class appears in every split.

No ROS code should exist here.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import os
import random
import re

from .label_audit import source_id

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
_FRAME = re.compile(r'^(.*)_(\d+)_(jpg|jpeg|png)$', re.I)


@dataclass
class Sample:
    """One original image with all its augmented copies."""

    source: str
    copies: list = field(default_factory=list)   # [(image_path, label_path), ...]
    classes: list = field(default_factory=list)  # class of every object (first copy)
    n_boxes: int = 0                              # objects labelled as plain boxes

    @property
    def main_class(self):
        """Most common class in the image, or None for an image without labels."""
        return Counter(self.classes).most_common(1)[0][0] if self.classes else None


def collect(export_dir):
    """
    All samples of a Roboflow YOLO export (every split folder found in it).

    Looks for <split>/images and <split>/labels under export_dir; an image
    without a label file counts as an image without objects.
    """
    samples = {}
    for split in sorted(os.listdir(export_dir)):
        img_dir = os.path.join(export_dir, split, 'images')
        lbl_dir = os.path.join(export_dir, split, 'labels')
        if not os.path.isdir(img_dir):
            continue
        for name in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            label = os.path.join(lbl_dir, stem + '.txt')
            src = source_id(stem + '.txt')
            sample = samples.setdefault(src, Sample(src))
            sample.copies.append((os.path.join(img_dir, name),
                                  label if os.path.exists(label) else None))
    for s in samples.values():
        first_label = s.copies[0][1]
        if first_label:
            s.classes, s.n_boxes = read_label_info(first_label)
    return list(samples.values())


def read_classes(label_path):
    return read_label_info(label_path)[0]


def read_label_info(label_path):
    """(class of every object, number of objects labelled as plain boxes)."""
    classes, boxes = [], 0
    with open(label_path, encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if parts:
                classes.append(int(parts[0]))
                boxes += len(parts) == 5          # 'class cx cy w h' = a box
    return classes, boxes


def mixed_box_images(samples):
    """
    Images whose labels are boxes in a dataset that is mostly polygons.

    Segmentation training fails on a mix of box and polygon labels, so such
    images are best left out. Returns [] for a box (detection) dataset.
    """
    boxes = sum(s.n_boxes for s in samples)
    objects = sum(len(s.classes) for s in samples)
    if boxes == 0 or boxes * 2 > objects:
        return []
    return sorted(s.source for s in samples if s.n_boxes)


def stratified_split(samples, val=0.15, test=0.15, seed=42, block_frames=1, gap=0):
    """
    Assign whole samples to train / valid / test.

    Images are first gathered into blocks of block_frames consecutive frames
    of one recording (see frame_block); a whole block always goes to one
    split, so neighbouring, nearly identical frames never end up on both
    sides. block_frames=1 splits per image.

    Blocks are grouped by their main class (images without objects form one
    more group). Within a group, shuffled blocks are handed one by one to the
    split that is furthest below its target share of images, so every class
    is represented in every split when it has at least three blocks.

    Where a valid or test block meets a train block, the frames on either
    side are still almost identical. gap > 0 leaves out ('drop') every valid
    or test image within gap frames of a train image of the same recording.
    Returns {sample.source: 'train' | 'valid' | 'test' | 'drop'}.
    """
    out = _assign_blocks(samples, val, test, seed, block_frames)
    if gap > 0:
        frames = {}
        for s in samples:
            m = _FRAME.match(s.source)
            if m:
                frames[(m.group(1), int(m.group(2)))] = s.source
        for (prefix, n), src in frames.items():
            if out[src] in ('valid', 'test') and any(
                    out.get(frames.get((prefix, n + d))) == 'train'
                    for d in range(-gap, gap + 1) if d):
                out[src] = 'drop'
    return out


def _assign_blocks(samples, val, test, seed, block_frames):
    blocks = defaultdict(list)
    for s in samples:
        blocks[frame_block(s.source, block_frames)].append(s)

    groups = defaultdict(list)
    for key, members in blocks.items():
        classes = [s.main_class for s in members if s.main_class is not None]
        main = Counter(classes).most_common(1)[0][0] if classes else None
        groups[main].append(key)

    rng = random.Random(seed)
    shares = {'test': test, 'valid': val, 'train': 1.0 - val - test}
    out = {}
    for cls in sorted(groups, key=lambda k: (k is None, k)):
        keys = sorted(groups[cls])
        rng.shuffle(keys)
        if len(keys) < 3:                       # too few blocks to evaluate on
            for key in keys:
                for s in blocks[key]:
                    out[s.source] = 'train'
            continue
        total = sum(len(blocks[k]) for k in keys)
        have = {sp: 0 for sp in shares}
        for key in keys:
            # the split furthest below its target share gets the next block
            split = max(shares, key=lambda sp: (shares[sp] * total - have[sp])
                        / max(shares[sp] * total, 1e-9))
            have[split] += len(blocks[key])
            for s in blocks[key]:
                out[s.source] = split
    return out


def frame_block(source, block_frames):
    """
    Block of consecutive frames an image belongs to.

    'STOP_00042_jpg' with block_frames=10 -> 'STOP#4' (frames 40-49). Images
    whose name has no frame number form a block of their own.
    """
    m = _FRAME.match(source)
    if block_frames <= 1 or not m:
        return source
    return f'{m.group(1)}#{int(m.group(2)) // block_frames}'


def relabel_lines(lines, new_class):
    """Replace the class of every object in a label file's lines."""
    out = []
    for line in lines:
        parts = line.split()
        if parts:
            out.append(' '.join([str(new_class)] + parts[1:]) + '\n')
    return out
