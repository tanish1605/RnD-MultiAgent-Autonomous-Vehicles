# 6 - Semantic Segmentation

Pixel-level scene understanding for the vehicle: a neural network labels every
camera pixel as **background, drivable, yellow_line, centre_dash or obstacle**.

The classic `lane_detection` node finds lane lines with fixed colour and edge
thresholds (HSV range, Canny limits, ...), which break under shadows, glare
and lighting changes. A segmentation model learns what the tape looks like from
labelled examples instead. The goal is to feed its mask into the existing lane
pipeline; the polynomial fit, tracking and path generation stay as they are.

> **Status:** tooling, training and the ROS 2 node are complete and tested on
> synthetic data. No model has been trained on real track data yet, and the
> mask is not yet consumed by `lane_detection`.

---

## Contents

1. [Folder layout](#1-folder-layout)
2. [Setup](#2-setup)
3. [Workflow at a glance](#3-workflow-at-a-glance)
4. [Step by step](#4-step-by-step)
5. [Labelling guide](#5-labelling-guide)
6. [Running on the vehicle](#6-running-on-the-vehicle)
7. [Reading the results](#7-reading-the-results)
8. [Troubleshooting](#8-troubleshooting)
9. [Next steps](#9-next-steps)

---

## 1. Folder layout

```text
6 - Semantic Segmentation/
├── tools/                      # data preparation, run on a laptop or the Jetson
│   ├── lane_failure_report.py  # how often does classic lane detection fail?
│   ├── extract_frames.py       # bag / video -> upright JPEG frames
│   ├── convert_masks.py        # CVAT / Roboflow export -> class-index masks
│   ├── split_dataset.py        # train / val / test lists, split by recording
│   └── bag_io.py               # reads ros2 bags without ROS (rosbags library)
├── training/                   # run on a PC with a GPU, or Colab
│   ├── config/train.yaml       # classes, input size, training + augmentation settings
│   ├── train.py                # trains, keeps best.pt by validation mIoU
│   ├── evaluate.py             # per-class IoU, speed, comparison images
│   ├── export_onnx.py          # checkpoint -> ONNX (+ metadata), verified
│   ├── dataset.py              # data loading + augmentation
│   └── common.py               # model definition, metrics, overlays
├── semantic_segmentation/      # ROS 2 package (colcon)
│   ├── semantic_segmentation/
│   │   ├── segmenter.py        # model wrapper, no ROS code
│   │   └── segmentation_node.py
│   ├── config/segmentation_params.yaml
│   ├── launch/segmentation.launch.py
│   └── models/                 # put seg.onnx / best.pt here (git-ignored)
├── dataset/                    # your data (git-ignored), created by the tools
│   ├── images/<recording>/*.jpg
│   ├── masks/<recording>/*.png
│   └── splits/{train,val,test}.txt
└── requirements.txt
```

The **class list lives in one place**: `training/config/train.yaml`. The
converter reads it, training stores it in the checkpoint, the export writes it
into the ONNX file, and the ROS node reads it from there. The order of the
list defines the pixel values (0 = background, 1 = drivable, ...).

---

## 2. Setup

**Laptop / PC / Colab** (all commands below are run from this folder):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Jetson:** torch, torchvision and OpenCV are already installed. For the tools
add `pip install rosbags`. The ROS package builds like any other:

```bash
cd ~/ros2_ws && colcon build --packages-select semantic_segmentation --symlink-install
```

---

## 3. Workflow at a glance

```text
 record ──► report ──► extract ──► label ──► convert ──► split ──► train ──► evaluate ──► export ──► deploy
 (car)     (laptop)   (laptop)   (CVAT/     (laptop)    (laptop)   (GPU)     (GPU)        (GPU)     (Jetson)
                                  Roboflow)
```

| Step | Command | Output |
|---|---|---|
| Record | `ros2 bag record /oak/rgb/image_raw /lane/status` | `rosbag2_<date>/` |
| Report | `tools/lane_failure_report.py` | failure % + `failures.json` |
| Extract | `tools/extract_frames.py` | `dataset/images/<rec>/` |
| Label | CVAT or Roboflow | export zip |
| Convert | `tools/convert_masks.py` | `dataset/masks/<rec>/` |
| Split | `tools/split_dataset.py` | `dataset/splits/*.txt` |
| Train | `training/train.py` | `training/runs/<time>/best.pt` |
| Evaluate | `training/evaluate.py` | IoU table + images |
| Export | `training/export_onnx.py` | `seg.onnx` + `seg.yaml` |
| Deploy | `ros2 launch semantic_segmentation segmentation.launch.py` | `/segmentation/mask` |

---

## 4. Step by step

### 4.1 Record drives

On the vehicle, with the camera driver and `lane_detection` running:

```bash
ros2 bag record -o run_01 /oak/rgb/image_raw /lane/status
```

Aim for **at least 5 separate recordings** that differ from each other:
morning vs evening light, lights on/off, both driving directions, curves,
obstacles on the track, a dirty or worn tape section. Variety matters far more
than length; 1–2 minutes per recording is plenty.

### 4.2 Measure where the classic pipeline fails

```bash
python3 tools/lane_failure_report.py --bag run_01 --json run_01_failures.json
```

```text
Frames analysed : 3600  (120.0 s)
Healthy frames  :   3150  ( 87.5 %)
Bad frames      :    450  ( 12.5 %)

Category            frames      %
  lane_lost            120    3.3
  no_yellow             40    1.1
  yellow_degraded      210    5.8
  dash_fallback        300    8.3
  dash_degraded         90    2.5

Longest bad stretches (37 total):
   start_s    end_s   dur_s  frames  main causes
     41.20    44.87    3.67     98  dash_fallback:98, yellow_degraded:61, ...
```

(numbers above are illustrative)

| Category | Meaning |
|---|---|
| `lane_lost` | no lane centre at all; the controller gets no error |
| `no_yellow` | the yellow line has never been found |
| `yellow_degraded` | this frame's yellow fit was rejected (`REJ`) or had too few points (`LOW_PTS`) |
| `dash_fallback` | no usable dashes; centre **guessed** as yellow + `l2c_offset` |
| `dash_degraded` | this frame's dash fit was rejected or had too few points |

This number is the baseline segmentation has to beat. Without a bag you can
also use `ros2 topic echo /lane/status > status.txt` and
`--text status.txt --fps 30`.

### 4.3 Extract frames to label

```bash
# a general sample: every 15th frame, near-duplicates skipped
python3 tools/extract_frames.py --bag run_01 --out dataset/images --every 15

# plus extra frames from the moments the classic pipeline struggled
python3 tools/extract_frames.py --bag run_01 --out dataset/images \
        --windows run_01_failures.json --pad 1.0 --every 5
```

Both commands write into the **same** folder, `dataset/images/run_01/`.
Keep it that way: one folder per drive. If the failure frames went into a
separate folder (e.g. with `--name run_01_fail`), the splitter would treat
them as a different recording and could put frames of the same drive into both
train and test. Frames extracted twice simply overwrite each other (the file
name is the frame index).

- Frames are **rotated 180°** by default because the OAK-D is mounted upside
  down; every node rotates before processing, so labels must match that view.
  Use `--no-rotate` only for footage that is already upright.
- `--min-diff` (default 4) skips frames that barely changed, e.g. while the car
  stands still.
- Videos work too: `--video lane_debug.mp4 --name lab_day2`.

**How many?** Start with **200–300 labelled frames** over 5+ recordings,
train, look at the mistakes, then label more frames like the ones it gets
wrong. That loop beats labelling 1000 random frames up front.

### 4.4 Label

See the [labelling guide](#5-labelling-guide). Create **one CVAT task (or
Roboflow batch) per recording folder**, so the exported masks can be placed
in the matching `dataset/masks/<recording>/` folder.

### 4.5 Convert the export

```bash
# CVAT: Export task dataset -> "Segmentation mask 1.1"
python3 tools/convert_masks.py cvat --src cvat_export_run_01 --out dataset/masks/run_01

# Roboflow: Export -> "Semantic Segmentation Masks"
python3 tools/convert_masks.py roboflow --src rf_export/train --out dataset/masks/run_01
```

Labels are matched **by name** (case, spaces and dashes are ignored, so
"Yellow Line" = `yellow_line`). Labels not in `train.yaml` are reported and
become *ignore* pixels, which are not used for training.

### 4.6 Split

```bash
python3 tools/split_dataset.py --root dataset
```

Whole recordings go to one split, so near-identical neighbouring frames never
appear in both train and test (which would make test scores misleadingly
high). With fewer than 5 recordings it falls back to 40-frame chunks and warns.

### 4.7 Train

```bash
python3 training/train.py
```

- Settings: `training/config/train.yaml` (epochs, batch size, input size, ...).
- First run downloads ImageNet weights for the MobileNetV3 backbone (~20 MB).
- Device is picked automatically: CUDA GPU, Apple GPU (MPS), else CPU.
- Output: `training/runs/<timestamp>/best.pt` (best validation mIoU),
  `last.pt`, `log.csv`.
- Quick smoke test: `python3 training/train.py --epochs 2`.

**What the training does for you**

- *Augmentation* simulates what breaks HSV thresholds: brightness/contrast,
  shadows (dark polygons), glare (bright blobs), blur. It does **not** flip
  images horizontally, because the yellow line is always on the left.
- *Class weights* (`class_weights: auto`) give rare thin classes such as
  dashes more weight, so the model does not just predict "background".

**Colab:** upload this folder (with `dataset/`) to Google Drive, open a GPU
notebook, `pip install -r requirements.txt`, then run the same commands.

### 4.8 Evaluate

```bash
python3 training/evaluate.py training/runs/<timestamp>/best.pt
```

Prints IoU per class, mIoU, pixel accuracy and speed, and saves images
`input | ground truth | prediction` into `runs/<timestamp>/eval_test/`.
**Look at those images**; they tell you what to label next.

### 4.9 Export

```bash
python3 training/export_onnx.py training/runs/<timestamp>/best.pt \
        --out semantic_segmentation/models/seg.onnx
```

Writes `seg.onnx` and `seg.yaml` (class names, colours, input size,
normalisation), checks the ONNX output against PyTorch, and refuses if they
disagree.

---

## 5. Labelling guide

Consistency matters more than precision. Agree on these rules before
labelling and keep them identical for every frame.

### Labels to create in CVAT / Roboflow

| Label | What to draw |
|---|---|
| `drivable` | The track surface the car may drive on: **both lanes**, from the yellow line to the outer right edge. Draw it as one large polygon *underneath* the lines. |
| `yellow_line` | The yellow tape itself, full width, as far as you can see it. |
| `centre_dash` | **Each white dash separately.** Do not join dashes; the gaps are `drivable`. |
| `obstacle` | Anything on the track the car must not hit: boxes, other cars, feet, cones. |
| `ignore` *(optional)* | Pixels you genuinely cannot decide (heavy blur, far horizon). They are skipped in training. |
| *(unlabelled)* | Everything else: floor outside the track, walls, signs on poles. Becomes `background`. |

### Rules of thumb

- **Draw lines on top of `drivable`.** In CVAT, set the z-order so lines and
  obstacles are above the drivable polygon, otherwise they get overwritten in
  the export.
- **Label what is visible, not what you know is there.** A dash hidden by an
  obstacle is `obstacle`; don't draw the dash.
- **Follow the edge of the tape**, not a straight approximation, on curves.
- **Shadows and glare do not change the class.** Tape in shadow is still
  `yellow_line`; these frames are exactly the ones the model needs.
- **Don't label far-away pixels you can't see clearly**; leave them unlabelled
  or use `ignore`.
- Traffic signs are handled by the YOLO node. Don't label them unless they
  stand **on** the track, in which case they are `obstacle`.

### CVAT tips

- Use the **polygon** tool; the "AI tools" / SAM interactor speeds up big
  regions such as `drivable` a lot.
- Use the **brush** tool for thin lines if polygons are fiddly.
- Copy annotations to the next frame (CTRL+C / CTRL+V) when frames are similar.
- Export with **"Segmentation mask 1.1"**.

### Adding or changing classes

Edit `classes:` in `training/config/train.yaml` (keep `background` first),
re-run `convert_masks.py`, retrain and re-export. The ROS node picks the new
classes up from the model file automatically.

---

## 6. Running on the vehicle

1. Put the model in the package: `semantic_segmentation/models/seg.onnx`
   (+ `seg.yaml`), or a `best.pt`.
2. Build: `colcon build --packages-select semantic_segmentation --symlink-install`
3. Run:

```bash
ros2 launch depthai_ros_driver_v3 driver.launch.py               # terminal 1
ros2 launch semantic_segmentation segmentation.launch.py         # terminal 2
rqt   # Plugins -> Visualization -> Image View -> /segmentation/overlay
```

### Which model file on the Jetson?

| File | Runtime | Notes |
|---|---|---|
| `best.pt` | PyTorch (already installed, uses the GPU) | **Recommended for now.** Set `model_path` to the `.pt` file. |
| `seg.onnx` | onnxruntime | `pip install onnxruntime` is CPU-only on the Jetson, so slow. GPU onnxruntime for Jetson needs NVIDIA's Jetson-specific wheel. |
| `.engine` | TensorRT | Fastest; **not supported by the node yet** (see [next steps](#9-next-steps)). |

### Topics

| Topic | Type | Content |
|---|---|---|
| `/segmentation/mask` | `sensor_msgs/Image`, `mono8` | pixel value = class index; upright; camera resolution by default; same header/timestamp as the camera frame |
| `/segmentation/overlay` | `sensor_msgs/Image`, `bgr8` | colour overlay; **only computed while something subscribes** |
| `/segmentation/classes` | `std_msgs/String` (latched) | JSON list of class names; index = mask value |

The mask looks almost black in an image viewer because its values are 0–4.
Use the overlay to look at it.

### Parameters (`config/segmentation_params.yaml`)

| Parameter | Default | Meaning |
|---|---|---|
| `model_path` | `""` | empty = `<share>/semantic_segmentation/models/seg.onnx` |
| `device` | `auto` | `auto`, `cuda` or `cpu` |
| `rotate_180` | `true` | camera is mounted upside down |
| `full_resolution_mask` | `true` | `false` publishes the mask at model size (e.g. 512×288), cheaper |
| `process_every_n` | `1` | `2` = segment every second frame |
| `overlay_alpha` | `0.5` | overlay colour strength |
| `stats_period_sec` | `5.0` | how often ms/frame is logged |
| `*_topic` | see file | topic names |

### Checking it works

```bash
ros2 topic hz /segmentation/mask        # frames per second actually published
ros2 topic echo --once /segmentation/classes
```

The node logs `N frames segmented, X ms/frame` every 5 s. If X is larger than
the camera period (33 ms at 30 fps), frames are skipped. That is safe (the
newest frame is always used), but consider `process_every_n: 2`, a smaller
`input_width/height` in `train.yaml`, or TensorRT.

---

## 7. Reading the results

**IoU** (intersection over union) per class = pixels correctly labelled as
that class ÷ pixels that are that class in the truth *or* the prediction.
1.0 is perfect; **mIoU** is the average over classes.

- Thin classes (`yellow_line`, `centre_dash`) always score lower than large
  ones (`drivable`, `background`) for the same visual quality, because a
  1-pixel offset is a large fraction of a thin line. Compare runs
  **class by class**.
- Always check the comparison images in `eval_test/`; a decent mIoU can still
  hide a systematic mistake, such as dashes missed in shadows.
- The number that ultimately matters is **how often lane detection fails**.
  Once the mask is wired into `lane_detection`, re-run
  `lane_failure_report.py` on the same bags and compare with section 4.2.

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Prediction is all background | Too few labelled pixels of other classes, or masks not matching images (check names). Look at a mask with `evaluate.py` comparison images. |
| `splits/train.txt missing` | Run `tools/split_dataset.py` first. |
| `mask without image` warnings | Mask and image names differ; masks must be `<image stem>.png` in the same recording folder name. |
| `Topic ... not in bag` | Check the topic name; the error lists what the bag contains. Compressed topics work too. |
| Labels ignored in `convert_masks.py` | The label name in CVAT/Roboflow doesn't match `train.yaml`. |
| Node: `Model not found` | Put `seg.onnx` into `models/` and rebuild, or set `model_path`. |
| Node: `No module named onnxruntime` | `pip install onnxruntime`, or use a `.pt` model. |
| Prediction upside down | Frames were labelled in the wrong orientation; check `--no-rotate` use and `rotate_180`. |
| ROS rejects a parameter | Type mismatch in YAML, e.g. `0` instead of `0.0` for a float. |

---

## 9. Next steps

1. **Collect and label real data**, train, and evaluate (sections 4–5).
2. **Use the mask in `lane_detection`**: add a `mask_source: classic | segmentation`
   parameter that replaces the HSV/Canny mask extraction with the
   `yellow_line` / `centre_dash` pixels from `/segmentation/mask`, warped with
   the same bird's-eye transform. Fitting, tracking and path generation stay
   unchanged, so the two sources can be compared directly with the failure
   report.
3. **TensorRT on the Jetson**:
   `/usr/src/tensorrt/bin/trtexec --onnx=seg.onnx --saveEngine=seg_fp16.engine --fp16`,
   then add an `.engine` backend to `segmenter.py` and measure the speed-up.
4. **Free space for planning**: combine `drivable` / `obstacle` with LiDAR
   obstacles, which the 2D LiDAR alone can miss when they are below its scan
   plane.
