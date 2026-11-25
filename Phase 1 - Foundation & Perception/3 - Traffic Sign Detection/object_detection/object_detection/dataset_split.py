#!/usr/bin/env python3

"""
dataset_split.py.

Clean and re-split a YOLO dataset exported from Roboflow, keeping every
augmented copy of an image in the same split.

Roboflow exports each original image several times (names differ only in
'.rf.<hash>') when augmentation is on. If those copies land in different
splits, the test set contains near-copies of training images and the test
score looks better than the model is. Here the split is made per original
image, stratified by class so that every class appears in every split.

No ROS code should exist here.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import os
import random

from .label_audit import source_id

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


@dataclass
class Sample:
    """One original image with all its augmented copies."""

    source: str
    copies: list = field(default_factory=list)   # [(image_path, label_path), ...]
    classes: list = field(default_factory=list)  # class of every object (first copy)

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
            s.classes = read_classes(first_label)
    return list(samples.values())


def read_classes(label_path):
    with open(label_path, encoding='utf-8') as f:
        return [int(line.split()[0]) for line in f if line.strip()]


def stratified_split(samples, val=0.15, test=0.15, seed=42):
    """
    Assign whole samples to train / valid / test.

    Samples are grouped by their main class (images without objects form
    one more group); each group is shuffled and cut in the given shares, so
    every class is represented in every split where it has enough images.
    Returns {sample.source: split}.
    """
    groups = defaultdict(list)
    for s in samples:
        groups[s.main_class].append(s)
    rng = random.Random(seed)
    out = {}
    for key in sorted(groups, key=lambda k: (k is None, k)):
        items = sorted(groups[key], key=lambda s: s.source)
        rng.shuffle(items)
        n = len(items)
        n_test = round(n * test) if n >= 3 else 0
        n_val = round(n * val) if n >= 3 else 0
        for i, s in enumerate(items):
            out[s.source] = ('test' if i < n_test else
                             'valid' if i < n_test + n_val else 'train')
    return out


def relabel_lines(lines, new_class):
    """Replace the class of every object in a label file's lines."""
    out = []
    for line in lines:
        parts = line.split()
        if parts:
            out.append(' '.join([str(new_class)] + parts[1:]) + '\n')
    return out
