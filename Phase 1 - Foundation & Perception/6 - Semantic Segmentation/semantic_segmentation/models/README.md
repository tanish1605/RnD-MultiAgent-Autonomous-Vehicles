# models/

Put the deployed model here as **`seg.onnx`** (plus the `seg.yaml` written
next to it by `training/export_onnx.py`). It is installed with the package
and used when `model_path` is empty in `config/segmentation_params.yaml`.

```bash
python3 ../../training/export_onnx.py ../../training/runs/<run>/best.pt --out seg.onnx
```

Model files are git-ignored by default; add one deliberately with
`git add -f models/seg.onnx` if the team wants it versioned.
