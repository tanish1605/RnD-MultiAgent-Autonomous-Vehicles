"""Unit tests for segmenter.py (no ROS needed; skipped if torch is missing)."""

import json

import numpy as np
import pytest

from semantic_segmentation.segmenter import Segmenter

torch = pytest.importorskip('torch')
segmentation = pytest.importorskip('torchvision.models.segmentation')

META = {
    'arch': 'lraspp_mobilenet_v3_large',
    'classes': ['background', 'drivable', 'yellow_line'],
    'colors': [[0, 0, 0], [0, 160, 0], [0, 215, 255]],
    'input_size': [64, 48],
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    """Write a small untrained checkpoint in the format train.py produces."""
    torch.manual_seed(0)
    model = segmentation.lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=len(META['classes']))
    path = tmp_path_factory.mktemp('model') / 'seg.pt'
    torch.save({**META, 'state_dict': model.state_dict(), 'ignore_index': 255}, path)
    return path, model.eval()


def test_mask_shape_and_values(checkpoint):
    seg = Segmenter(checkpoint[0], device='cpu')
    assert seg.class_names == META['classes']
    assert (seg.input_w, seg.input_h) == (64, 48)

    img = np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8)
    full = seg.segment(img)
    small = seg.segment(img, full_resolution=False)
    assert full.shape == (120, 200) and full.dtype == np.uint8
    assert small.shape == (48, 64)
    assert full.max() < len(META['classes'])
    assert seg.overlay(img, small).shape == img.shape


def test_onnx_matches_torch(checkpoint, tmp_path):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    import onnx

    path_pt, model = checkpoint

    class Logits(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)['out']

    onnx_path = tmp_path / 'seg.onnx'
    dummy = torch.randn(1, 3, 48, 64)
    try:
        torch.onnx.export(Logits(model), dummy, str(onnx_path), input_names=['input'],
                          output_names=['logits'], opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(Logits(model), dummy, str(onnx_path), input_names=['input'],
                          output_names=['logits'], opset_version=17)
    m = onnx.load(str(onnx_path))
    for k, v in META.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = k, json.dumps(v)
    onnx.save(m, str(onnx_path))

    img = np.random.default_rng(1).integers(0, 255, (96, 128, 3), dtype=np.uint8)
    a = Segmenter(path_pt, device='cpu').segment(img)
    b = Segmenter(onnx_path, device='cpu').segment(img)
    assert (a == b).mean() > 0.999


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Segmenter(tmp_path / 'nope.onnx')
