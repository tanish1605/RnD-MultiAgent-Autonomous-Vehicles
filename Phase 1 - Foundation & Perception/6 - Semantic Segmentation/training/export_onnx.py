#!/usr/bin/env python3
"""
export_onnx.py
==============
Convert a checkpoint to ONNX for the ROS node / TensorRT.

- fixed input 1x3xHxW (TensorRT is fastest with static shapes)
- output: raw class scores (logits) 1xCxHxW; the node takes the argmax
- class names, colours, input size and mean/std are written INTO the ONNX
  file (metadata) and next to it as <name>.yaml, so the node needs no
  extra configuration to interpret the output
- verifies the ONNX output against PyTorch with onnxruntime (if installed)

On the Jetson, build a TensorRT engine from the ONNX file:
  /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16

Usage
-----
  python3 export_onnx.py runs/20261019_120000/best.pt
  python3 export_onnx.py runs/.../best.pt --out ../semantic_segmentation/models/seg.onnx
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from common import LogitsOnly, load_checkpoint

META_KEYS = ('arch', 'classes', 'colors', 'input_size', 'mean', 'std')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint')
    ap.add_argument('--out', help='output .onnx (default: next to the checkpoint)')
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint, 'cpu')
    w, h = ckpt['input_size']
    out = Path(args.out) if args.out else Path(args.checkpoint).with_suffix('.onnx')
    out.parent.mkdir(parents=True, exist_ok=True)

    net = LogitsOnly(model).eval()
    dummy = torch.randn(1, 3, h, w)
    export_kw = dict(input_names=['input'], output_names=['logits'],
                     opset_version=args.opset)
    try:   # classic TorchScript exporter: most TensorRT-friendly graphs
        torch.onnx.export(net, dummy, str(out), dynamo=False, **export_kw)
    except TypeError:   # older torch without the `dynamo` argument
        torch.onnx.export(net, dummy, str(out), **export_kw)

    import onnx
    m = onnx.load(str(out))
    meta = {k: ckpt[k] for k in META_KEYS}
    for k, v in meta.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = k, json.dumps(v)
    onnx.checker.check_model(m)
    onnx.save(m, str(out))
    with open(out.with_suffix('.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f'Wrote {out}  (+ {out.with_suffix(".yaml").name})  input 1x3x{h}x{w}')

    try:
        import onnxruntime as ort
    except ImportError:
        print('onnxruntime not installed - skipped numerical check')
        return
    sess = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    onnx_out = sess.run(None, {'input': dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = net(dummy).numpy()
    diff = float(np.abs(onnx_out - torch_out).max())
    agree = float((onnx_out.argmax(1) == torch_out.argmax(1)).mean())
    print(f'ONNX vs PyTorch: max |diff| {diff:.2e}, argmax agreement {100 * agree:.2f} %')
    if agree < 0.999:
        raise SystemExit('ONNX output differs from PyTorch - do not deploy this file')


if __name__ == '__main__':
    main()
