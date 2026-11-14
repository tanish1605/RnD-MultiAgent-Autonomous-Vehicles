#!/usr/bin/env python3
"""
auto_label.py
=============
Label frames with SAM 2 (Segment Anything 2) instead of drawing polygons.

For each recording in dataset/images/<recording>/ you click a few points on
one frame (road, yellow line, dashes, obstacles). SAM 2 segments what you
clicked and follows it through the other frames of the recording. You then
scrub through the result, add a click where it went wrong, exclude frames
that are beyond repair, and save. Masks are written straight to
dataset/masks/<recording>/<stem>.png in the format train.py expects, so no
CVAT export or convert_masks.py is needed.

Tracking works because the frames come from one continuous drive. The closer
together they are (smaller --every in extract_frames.py), the better SAM 2
follows thin things such as the centre dashes.

Existing masks (e.g. hand-labelled in CVAT) are never overwritten unless
--overwrite is given.

Setup (GPU laptop, inside the training venv)
-----
  pip install "git+https://github.com/facebookresearch/sam2.git" huggingface_hub
  If the install fails while compiling, set SAM2_BUILD_CUDA=0 and run it again
  (the CUDA extension is optional).

Usage (run from the "6 - Semantic Segmentation" folder)
-----
  python tools/auto_label.py                            # every recording
  python tools/auto_label.py --recordings run_01 run_02
  python tools/auto_label.py --model facebook/sam2.1-hiera-large   # slower, better

The controls are printed when it starts (press h to print them again).
"""

import argparse
import contextlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

DEFAULT_CFG = Path(__file__).resolve().parent.parent / 'training' / 'config' / 'train.yaml'
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
WINDOW = 'auto_label'

CONTROLS = """
Controls (click on the image window first so it has keyboard focus)
  1-9          pick class ({classes})
  left click   add a point ON the current object
  right click  add a point NOT on it (removes a wrongly included area)
  n            start a new, separate object of the current class
               (e.g. a second obstacle)
  z            undo the last point of the current object on this frame
  p            track: follow every object through all frames
  a / d        previous / next frame        [ / ]   jump 10 frames
  x            exclude / include this frame (excluded frames are not saved)
  t            hide / show the coloured overlay
  w            write the masks and go to the next recording
  q q          skip this recording without saving (press q twice)
  Esc          quit
Typical use: on frame 1 click the road (1), yellow line (2), a few dashes (3)
and any obstacle (4); press p; step through with d; where a frame is wrong,
add a click there and press p again; then w.
"""


def load_classes(cfg_path):
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    names = [c['name'] for c in cfg['classes']]
    colors = [tuple(int(v) for v in c['color']) for c in cfg['classes']]
    return names, colors


def build_predictor(model_id, device):
    try:
        from sam2.build_sam import build_sam2_video_predictor_hf
    except ImportError:
        sys.exit('SAM 2 is not installed. Run:\n  pip install '
                 '"git+https://github.com/facebookresearch/sam2.git" huggingface_hub')
    print(f'Loading {model_id} on {device} (first run downloads the weights) ...')
    return build_sam2_video_predictor_hf(model_id, device=device)


class Session:
    """
    One SAM 2 tracking session over a list of frames, plus the clicks so far.

    No GUI code here, so it can also be driven from a script or a test.
    Labels are uint8 class-index masks (0 = background), mask_width wide.
    """

    def __init__(self, predictor, frame_paths, mask_width=960, low_memory=False):
        self.predictor = predictor
        self.paths = [Path(p) for p in frame_paths]
        # SAM 2 reads a folder of JPEGs named by frame number (0.jpg, 1.jpg, ...)
        self.tmp = Path(tempfile.mkdtemp(prefix='sam2_frames_'))
        for i, src in enumerate(self.paths):
            dst = self.tmp / f'{i:05d}.jpg'
            if src.suffix.lower() in ('.jpg', '.jpeg'):
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copyfile(src, dst)
            else:
                cv2.imwrite(str(dst), cv2.imread(str(src), cv2.IMREAD_COLOR))
        self.state = predictor.init_state(video_path=str(self.tmp),
                                          offload_video_to_cpu=True,
                                          offload_state_to_cpu=low_memory)
        h, w = self.state['video_height'], self.state['video_width']
        mw = min(mask_width, w)
        self.mask_size = (mw, round(mw * h / w))     # (w, h) for cv2.resize
        self.prompts = {}     # (frame, obj_id) -> [(x, y, 1 = on / 0 = not on), ...]
        self.obj_class = {}   # obj_id -> class index
        self.labels = {}      # frame -> mask from the last tracking run
        self.preview = {}     # frame -> mask from the clicks on that frame alone
        self.tracked = False  # SAM 2 state holds a tracking run
        self.dirty = False    # clicks changed since the last tracking run

    def __len__(self):
        return len(self.paths)

    def label(self, frame):
        """Best mask to show/save for a frame, or None."""
        return self.preview.get(frame, self.labels.get(frame))

    def add_point(self, frame, obj_id, cls, x, y, positive=True):
        self.obj_class[obj_id] = cls
        self.prompts.setdefault((frame, obj_id), []).append(
            (float(x), float(y), int(positive)))
        self.dirty = True
        if self.tracked:
            self._rebuild()     # SAM 2 takes no new objects once tracking ran
        else:
            self._send(frame, obj_id)

    def undo(self, frame, obj_id):
        points = self.prompts.get((frame, obj_id))
        if not points:
            return False
        points.pop()
        if not points:
            del self.prompts[(frame, obj_id)]
        self.dirty = True
        self._rebuild()         # the object may now have no clicks anywhere
        return True

    def track(self):
        """Propagate every object through all frames (forward, then backward)."""
        if not self.prompts:
            return False
        if self.tracked:
            if not self.dirty:
                return True
            self._rebuild()
        for reverse in (False, True):
            for frame, obj_ids, masks in self.predictor.propagate_in_video(
                    self.state, reverse=reverse):
                self.labels[frame] = self._combine(obj_ids, masks)
        self.preview = {}
        self.tracked, self.dirty = True, False
        return True

    def save(self, out_dir, exclude=(), overwrite=False):
        """Write tracked masks; returns (written, skipped because they exist)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = kept = 0
        for frame, label in sorted(self.labels.items()):
            if frame in exclude:
                continue
            dst = out_dir / f'{self.paths[frame].stem}.png'
            if dst.exists() and not overwrite:
                kept += 1
                continue
            cv2.imwrite(str(dst), label)
            written += 1
        return written, kept

    def close(self):
        self.state = None
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── internals ────────────────────────────────────────────────────────

    def _send(self, frame, obj_id):
        points = self.prompts[(frame, obj_id)]
        _, obj_ids, masks = self.predictor.add_new_points_or_box(
            inference_state=self.state, frame_idx=frame, obj_id=obj_id,
            points=np.array([p[:2] for p in points], dtype=np.float32),
            labels=np.array([p[2] for p in points], dtype=np.int32),
            clear_old_points=True)
        self.preview[frame] = self._combine(obj_ids, masks)

    def _rebuild(self):
        """Clear SAM 2's state and replay every click (keeps old labels for display)."""
        self.predictor.reset_state(self.state)
        self.tracked = False
        self.preview = {}
        for frame, obj_id in sorted(self.prompts):
            self._send(frame, obj_id)

    def _combine(self, obj_ids, masks):
        """Per-object SAM 2 logits -> one class-index mask, higher classes on top."""
        on = (masks[:, 0] > 0).cpu().numpy()
        label = np.zeros(on.shape[1:], dtype=np.uint8)
        order = sorted(range(len(obj_ids)), key=lambda i: self.obj_class[obj_ids[i]])
        for i in order:
            label[on[i]] = self.obj_class[obj_ids[i]]
        return cv2.resize(label, self.mask_size, interpolation=cv2.INTER_NEAREST)


class Labeler:
    """OpenCV window around a Session: clicks and keys in, overlay out."""

    def __init__(self, session, title, names, colors, display_width):
        self.s = session
        self.title = title
        self.names = names
        self.colors = colors
        lut = np.zeros((256, 3), dtype=np.uint8)
        lut[:len(colors)] = np.array(colors, dtype=np.uint8)
        self.lut = lut
        h, w = session.state['video_height'], session.state['video_width']
        self.scale = min(1.0, display_width / w)
        self.disp_size = (round(w * self.scale), round(h * self.scale))
        self.frame = 0
        self.cls = 1
        self.objects = []           # [(obj_id, class index)] in creation order
        self.obj = self._object_for(self.cls)
        self.exclude = set()
        self.show = True
        self.clicks = []
        self._cache = (None, None)
        cv2.setMouseCallback(WINDOW, self._on_mouse)

    def run(self):
        """Event loop; returns 'save', 'skip' or 'quit'."""
        redraw, q_armed = True, False
        while True:
            while self.clicks:
                x, y, positive = self.clicks.pop(0)
                self.s.add_point(self.frame, self.obj, self.cls, x, y, positive)
                redraw = True
            if redraw:
                cv2.imshow(WINDOW, self._render())
                redraw = False
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return 'quit'
            if key == 255:
                continue
            redraw = True
            ch = chr(key)
            if key == 27:
                return 'quit'
            if ch == 'q':
                if q_armed:
                    return 'skip'
                q_armed = True
                print('Press q again to skip this recording without saving.')
                continue
            q_armed = False
            if ch.isdigit() and 1 <= int(ch) < len(self.names):
                self.cls = int(ch)
                self.obj = self._object_for(self.cls)
            elif ch == 'n':
                self.obj = self._object_for(self.cls, new=True)
                print(f'New {self.names[self.cls]} object.')
            elif ch == 'z':
                if not self.s.undo(self.frame, self.obj):
                    print('Nothing to undo for this object on this frame.')
            elif ch == 'p':
                self._track()
            elif ch in 'ad[]':
                step = {'a': -1, 'd': 1, '[': -10, ']': 10}[ch]
                self.frame = min(max(self.frame + step, 0), len(self.s) - 1)
            elif ch == 'x':
                self.exclude ^= {self.frame}
            elif ch == 't':
                self.show = not self.show
            elif ch == 'h':
                self._print_controls()
            elif ch == 'w':
                if not self.s.prompts:
                    print('Nothing labelled yet - click on the image first.')
                    continue
                if self.s.dirty or not self.s.tracked:
                    self._track()
                return 'save'

    def _print_controls(self):
        classes = ', '.join(f'{i} {n}' for i, n in enumerate(self.names) if i)
        print(CONTROLS.format(classes=classes))

    def _object_for(self, cls, new=False):
        if not new:
            for obj_id, c in reversed(self.objects):
                if c == cls:
                    return obj_id
        obj_id = len(self.objects) + 1
        self.objects.append((obj_id, cls))
        return obj_id

    def _on_mouse(self, event, x, y, flags, param):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self.clicks.append((x / self.scale, y / self.scale,
                                event == cv2.EVENT_LBUTTONDOWN))

    def _track(self):
        if not self.s.prompts:
            print('Nothing to track - click on the image first.')
            return
        busy = self._render()
        self._text(busy, 'Tracking ... please wait', (20, busy.shape[0] // 2), 1.0)
        cv2.imshow(WINDOW, busy)
        cv2.waitKey(1)
        t0 = time.time()
        self.s.track()
        print(f'Tracked {len(self.s)} frames in {time.time() - t0:.0f} s. '
              'Step through with a / d and check every frame.')

    def _image(self, frame):
        if self._cache[0] != frame:
            img = cv2.imread(str(self.s.paths[frame]), cv2.IMREAD_COLOR)
            self._cache = (frame, cv2.resize(img, self.disp_size, interpolation=cv2.INTER_AREA))
        return self._cache[1]

    @staticmethod
    def _text(img, text, org, size=0.55):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1,
                    cv2.LINE_AA)

    def _render(self):
        s = self.s
        out = self._image(self.frame).copy()
        label = s.label(self.frame)
        if self.show and label is not None:
            lab = cv2.resize(label, self.disp_size, interpolation=cv2.INTER_NEAREST)
            painted = lab > 0
            blend = cv2.addWeighted(out, 0.45, self.lut[lab], 0.55, 0)
            out[painted] = blend[painted]

        for (frame, obj_id), points in s.prompts.items():
            if frame != self.frame:
                continue
            color = self.colors[s.obj_class[obj_id]]
            r = 7 if obj_id == self.obj else 5
            for x, y, positive in points:
                c = (round(x * self.scale), round(y * self.scale))
                if positive:
                    cv2.circle(out, c, r + 2, (0, 0, 0), -1)
                    cv2.circle(out, c, r, color, -1)
                else:
                    cv2.drawMarker(out, c, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 2 * r + 4, 4)
                    cv2.drawMarker(out, c, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 2 * r + 2, 2)

        if self.frame in self.exclude:
            cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (0, 0, 255), 8)
        if s.tracked and not s.dirty:
            status = 'tracked'
        elif s.labels:
            status = 'clicks changed - press p to track again'
        else:
            status = 'not tracked yet - click, then press p'
        nth = sum(1 for o, c in self.objects if c == self.cls and o <= self.obj)
        self._text(out, f'{self.title}   frame {self.frame + 1}/{len(s)}   '
                   f'{s.paths[self.frame].name}', (10, 22))
        self._text(out, f'class {self.cls}: {self.names[self.cls]} (object {nth})   '
                   f'{status}' + ('   EXCLUDED' if self.frame in self.exclude else ''),
                   (10, 46))
        self._text(out, '1-9 class  click add  right-click remove  n new object  z undo  '
                   'p track  a/d frame  x exclude  w save  h help', (10, out.shape[0] - 12),
                   0.45)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', default='dataset/images',
                    help='images root (one folder per recording)')
    ap.add_argument('--masks', default='dataset/masks', help='masks root to write into')
    ap.add_argument('--recordings', nargs='*', help='recording folder names (default: all)')
    ap.add_argument('--model', default='facebook/sam2.1-hiera-small',
                    help='SAM 2 model: facebook/sam2.1-hiera-{tiny,small,base-plus,large}')
    ap.add_argument('--device', default='auto', help='auto, cuda, mps or cpu')
    ap.add_argument('--chunk', type=int, default=150,
                    help='frames per session; longer recordings are done in parts')
    ap.add_argument('--mask-width', type=int, default=960,
                    help='saved mask width (training resizes to 512 wide anyway)')
    ap.add_argument('--display-width', type=int, default=1280, help='window width in pixels')
    ap.add_argument('--low-memory', action='store_true',
                    help='keep SAM 2 state in CPU RAM (slower, for small GPUs)')
    ap.add_argument('--overwrite', action='store_true', help='replace existing masks')
    ap.add_argument('--config', default=str(DEFAULT_CFG), help='train.yaml with class list')
    args = ap.parse_args()

    names, colors = load_classes(args.config)
    images_root, masks_root = Path(args.images), Path(args.masks)
    if not images_root.is_dir():
        sys.exit(f'{images_root} not found - run this from the "6 - Semantic Segmentation" folder')
    recordings = args.recordings or sorted(p.name for p in images_root.iterdir() if p.is_dir())
    if not recordings:
        sys.exit(f'No recording folders in {images_root}')

    import torch
    device = args.device
    if device == 'auto':
        device = ('cuda' if torch.cuda.is_available()
                  else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'cpu':
        print('[warn] no GPU found - SAM 2 on the CPU is very slow')
    # SAM 2 keeps its tracking memory in bfloat16, so it only runs under
    # bfloat16 autocast (also on GPUs without native bfloat16 - just slower)
    amp = (torch.autocast(device, dtype=torch.bfloat16) if device in ('cuda', 'cpu')
           else contextlib.nullcontext())
    predictor = build_predictor(args.model, device)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print(CONTROLS.format(classes=', '.join(f'{i} {n}' for i, n in enumerate(names) if i)))
    total = 0
    with torch.inference_mode(), amp:
        for rec in recordings:
            frames = sorted(p for p in (images_root / rec).glob('*')
                            if p.suffix.lower() in IMAGE_EXTS)
            if not frames:
                print(f'[warn] {rec}: no images, skipped')
                continue
            out_dir = masks_root / rec
            if not args.overwrite and all((out_dir / f'{p.stem}.png').exists() for p in frames):
                print(f'{rec}: every frame already has a mask - skipped (--overwrite to redo)')
                continue
            for start in range(0, len(frames), args.chunk):
                part = frames[start:start + args.chunk]
                title = rec if len(frames) <= args.chunk else \
                    f'{rec} [{start + 1}-{start + len(part)} of {len(frames)}]'
                print(f'\n{title}: loading {len(part)} frames ...')
                session = Session(predictor, part, args.mask_width, args.low_memory)
                try:
                    labeler = Labeler(session, title, names, colors, args.display_width)
                    result = labeler.run()
                    if result == 'save':
                        written, kept = session.save(out_dir, labeler.exclude, args.overwrite)
                        total += written
                        print(f'{title}: wrote {written} masks to {out_dir}'
                              + (f', kept {kept} existing' if kept else '')
                              + (f', excluded {len(labeler.exclude)}' if labeler.exclude else ''))
                    elif result == 'skip':
                        print(f'{title}: skipped, nothing saved')
                finally:
                    session.close()
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                if result == 'quit':
                    cv2.destroyAllWindows()
                    print(f'\nStopped. {total} masks written this run.')
                    return
    cv2.destroyAllWindows()
    print(f'\nDone. {total} masks written. Next: python tools/split_dataset.py --root dataset')


if __name__ == '__main__':
    main()
