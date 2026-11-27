"""Unit tests for robustness.py and tools/corrupt_test_set.py (no ROS needed)."""

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
import yaml

from object_detection.robustness import corrupt, CORRUPTIONS

TOOL = os.path.join(os.path.dirname(__file__), '..', 'tools', 'corrupt_test_set.py')


def scene(seed=0):
    """Make a textured test image: gradient, edges and a bright patch."""
    rng = np.random.default_rng(seed)
    img = np.tile(np.linspace(40, 200, 160, dtype=np.float32), (120, 1))
    img = np.stack([img, img * 0.8, img * 0.6], axis=2)
    img[40:80, 60:100] = 230
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize('name', CORRUPTIONS)
def test_shape_type_and_reproducible(name):
    img = scene()
    a = corrupt(img, name, 2, np.random.default_rng(1))
    b = corrupt(img, name, 2, np.random.default_rng(1))
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, img)


def test_severity_makes_it_worse():
    img = scene()
    rng = np.random.default_rng

    def measure(name, fn):
        return [fn(corrupt(img, name, s, rng(0))) for s in (1, 2, 3)]

    dark = measure('dark', np.mean)
    assert dark[0] > dark[1] > dark[2]
    contrast = measure('low_contrast', np.std)
    assert contrast[0] > contrast[1] > contrast[2]
    glare = measure('glare', np.mean)
    assert img.mean() < glare[0] < glare[1] < glare[2]

    def sharpness(x):
        return cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

    blur = measure('motion_blur', sharpness)
    assert sharpness(img) > blur[0] > blur[1] > blur[2]
    noise = measure('noise', lambda x: np.std(x.astype(float) - img))
    assert noise[0] < noise[1] < noise[2]


def test_bad_arguments():
    with pytest.raises(ValueError):
        corrupt(scene(), 'fog', 1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        corrupt(scene(), 'dark', 4, np.random.default_rng(0))


def test_tool_writes_copies_and_yamls(tmp_path):
    data_dir = tmp_path / 'clean'
    for sub in ('images', 'labels'):
        (data_dir / 'test' / sub).mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(data_dir / 'test' / 'images' / f'img{i}.jpg'), scene(i))
        (data_dir / 'test' / 'labels' / f'img{i}.txt').write_text('0 0.4 0.4 0.6 0.4 0.5 0.6\n')
    (data_dir / 'data.yaml').write_text(yaml.safe_dump(
        {'path': str(data_dir), 'train': 'train/images', 'val': 'valid/images',
         'test': 'test/images', 'names': ['stop']}))
    out = tmp_path / 'robust'
    run = subprocess.run([sys.executable, TOOL, str(data_dir / 'data.yaml'), str(out),
                          '--corruptions', 'dark', 'noise', '--severities', '1', '3'],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    for name in ('dark_s1', 'dark_s3', 'noise_s1', 'noise_s3'):
        assert len(os.listdir(out / name / 'images')) == 3
        assert len(os.listdir(out / name / 'labels')) == 3
        cfg = yaml.safe_load((out / f'{name}.yaml').read_text())
        assert cfg['names'] == ['stop'] and cfg['test'] == 'images'
        assert os.path.isdir(os.path.join(cfg['path'], cfg['test']))
    clean = yaml.safe_load((out / 'clean.yaml').read_text())
    assert os.path.samefile(os.path.join(clean['path'], clean['test']),
                            data_dir / 'test' / 'images')
    assert 'yolo val' in run.stdout


def test_evaluate_writes_drop_table(tmp_path, monkeypatch):
    """--models path, with a stand-in for Ultralytics that returns known scores."""
    import csv
    import importlib.util
    import types

    scores = {'clean': 0.90, 'dark_s1': 0.80, 'dark_s3': 0.50}

    class Metrics:
        def __init__(self, v):
            self.map50, self.map = v + 0.05, v

    class FakeYOLO:
        def __init__(self, path):
            self.path = path

        def val(self, data, split, imgsz, **kw):
            assert split == 'test' and imgsz == 640
            v = scores[os.path.splitext(os.path.basename(data))[0]]
            return types.SimpleNamespace(box=Metrics(v + 0.01), seg=Metrics(v))

    monkeypatch.setitem(sys.modules, 'ultralytics', types.SimpleNamespace(YOLO=FakeYOLO))
    spec = importlib.util.spec_from_file_location('corrupt_tool', TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    sets = [('clean', 0, 'clean.yaml'), ('dark', 1, 'dark_s1.yaml'), ('dark', 3, 'dark_s3.yaml')]
    tool.evaluate(['runs/yolo26n_seg/weights/best.pt'], sets, str(tmp_path), 640)
    rows = list(csv.DictReader(open(tmp_path / 'robustness.csv')))
    assert [r['corruption'] for r in rows] == ['clean', 'dark', 'dark']
    assert float(rows[2]['mask_map50_95']) == pytest.approx(0.50)
    assert float(rows[2]['drop_vs_clean']) == pytest.approx(0.40)
