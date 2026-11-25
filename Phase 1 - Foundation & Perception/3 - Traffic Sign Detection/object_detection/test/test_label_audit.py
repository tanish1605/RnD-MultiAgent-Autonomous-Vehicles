"""Unit tests for label_audit.py (no ROS needed)."""

import math

import numpy as np
import pytest

from object_detection.label_audit import (
    audit, fit_shape, LabelObject, name_prefix, read_labels, score_against_names,
    source_id)


def polygon(shape, n=10, cx=500, cy=500, r=100, jitter=0.0, seed=0):
    """Make a coarse, hand-drawn-looking outline of a sign shape."""
    rng = np.random.default_rng(seed)
    if shape == 'round':
        a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 0.3)
        pts = np.stack([np.cos(a), np.sin(a)], axis=1) * r
    else:
        corners = {
            'triangle_up': [(0, -1), (0.87, 0.5), (-0.87, 0.5)],
            'triangle_down': [(-0.87, -0.5), (0.87, -0.5), (0, 1)],
            'square': [(-1, -1), (1, -1), (1, 1), (-1, 1)],
            'diamond': [(0, -1.3), (1.3, 0), (0, 1.3), (-1.3, 0)],
        }[shape]
        c = np.array(corners) * r
        # points along the edges, as a labeller clicking around the sign would
        pts = []
        for i in range(len(c)):
            for t in np.linspace(0, 1, 3, endpoint=False):
                pts.append(c[i] + t * (c[(i + 1) % len(c)] - c[i]))
        pts = np.array(pts)
    pts = pts + rng.normal(0, jitter * r, pts.shape)
    return pts + [cx, cy]


@pytest.mark.parametrize('shape', ['round', 'triangle_up', 'triangle_down', 'square', 'diamond'])
def test_fit_shape(shape):
    assert fit_shape(polygon(shape, jitter=0.02)).shape == shape


@pytest.mark.parametrize('shape', ['round', 'triangle_up', 'triangle_down', 'square'])
def test_fit_shape_survives_stretched_image(shape):
    """Exports often stretch images to a square; the shape test must not care."""
    p = polygon(shape) * [0.56, 1.0]
    assert fit_shape(p).shape == shape


def test_degenerate_outline():
    assert fit_shape(np.array([[0, 0], [1, 1], [2, 2]])) is None


def test_names():
    assert source_id('STOP_00012_jpg.rf.4a340c440af0d8af4a89a8f51e3ccc2f.txt') == 'STOP_00012_jpg'
    assert source_id('plain_name.txt') == 'plain_name'
    assert name_prefix('no-left_turn_00065_jpg') == 'no-left_turn'
    assert name_prefix('Right_00016_png') == 'Right'


def make_dataset(n_per_class=20):
    """Three classes (round, triangle, square), each with its own file-name prefix."""
    objs = []
    for cls, (shape, prefix) in enumerate([('round', 'noentry'), ('triangle_up', 'warn'),
                                           ('square', 'park')]):
        for i in range(n_per_class):
            pts = polygon(shape, jitter=0.02, seed=cls * 100 + i)
            objs.append(LabelObject(f'{prefix}_{i:05d}_jpg', [f'{prefix}_{i}.txt'], cls, pts))
    return objs


def test_audit_finds_planted_wrong_class():
    objs = make_dataset()
    # a 'warn' (triangle) image wrongly labelled with the round class 0
    objs.append(LabelObject('warn_00099_jpg', ['warn_99.txt'], 0, polygon('triangle_up')))
    findings, table = audit(objs)
    assert table[0][0] == 'round' and table[1][0] == 'triangle_up' and table[2][0] == 'square'
    bad = [f for f in findings if f.obj.source == 'warn_00099_jpg'][0]
    assert 'shape_mismatch' in bad.flags and 'name_mismatch' in bad.flags
    assert bad.strength == 'strong'
    clean = [f for f in findings if f.obj.source != 'warn_00099_jpg']
    assert not any('shape_mismatch' in f.flags for f in clean)
    s = score_against_names(findings)
    assert s['found'] == 1 and s['recall'] == 1.0


def test_audit_orientation_and_self_crossing():
    objs = make_dataset()
    objs.append(LabelObject('warn_00098_jpg', ['w.txt'], 1, polygon('triangle_down')))
    bowtie = np.array([[400, 400], [600, 600], [600, 400], [400, 600]], float)
    objs.append(LabelObject('park_00097_jpg', ['p.txt'], 2, bowtie))
    findings, _ = audit(objs)
    by = {f.obj.source: f for f in findings}
    assert 'orientation_mismatch' in by['warn_00098_jpg'].flags
    assert 'self_crossing' in by['park_00097_jpg'].flags


def test_class_without_clear_shape_is_not_checked():
    objs = [LabelObject(f'mix_{i:05d}_jpg', ['m.txt'], 0,
                        polygon('round' if i % 2 else 'square', seed=i)) for i in range(20)]
    findings, table = audit(objs)
    assert table[0][0] is None
    assert not any('shape_mismatch' in f.flags for f in findings)


def test_read_labels(tmp_path):
    poly = ' '.join(f'{v:.4f}' for p in polygon('round', r=0.1, cx=0.5, cy=0.5) for v in p)
    for h in ('aaa', 'bbb', 'ccc'):                       # three augmented copies
        (tmp_path / f'sign_00001_jpg.rf.{h}.txt').write_text(f'3 {poly}\n')
    (tmp_path / 'sign_00002_jpg.rf.ddd.txt').write_text('')             # no labels
    (tmp_path / 'sign_00003_jpg.rf.eee.txt').write_text('1 0.5 0.5 0.2 0.2\n')  # a box
    (tmp_path / 'sign_00004_jpg.rf.fff.txt').write_text(f'2 {poly}\n')
    (tmp_path / 'sign_00004_jpg.rf.abc.txt').write_text(f'5 {poly}\n')  # copies disagree
    objs, empty, mismatch = read_labels(str(tmp_path))
    by = {o.source: o for o in objs}
    assert len(by['sign_00001_jpg'].files) == 3 and by['sign_00001_jpg'].class_id == 3
    assert empty == ['sign_00002_jpg']
    assert by['sign_00003_jpg'].is_box
    assert mismatch == ['sign_00004_jpg']
    assert by['sign_00001_jpg'].points.max() <= 1000
    assert math.isclose(np.ptp(by['sign_00003_jpg'].points[:, 0]), 200, rel_tol=1e-6)
