#!/usr/bin/env python3
"""
extract_frames.py
=================
Pull frames out of a ros2 bag or a video file for labelling.

- keeps every Nth frame (--every)
- skips near-duplicate frames (--min-diff), so a car standing still does not
  produce 500 identical images
- rotates 180 deg by default, because the OAK-D is mounted upside down and
  every node rotates before processing; frames must be labelled the way the
  model will see them
- optionally keeps only frames inside the "bad" stretches found by
  lane_failure_report.py (--windows), which are the most useful to label

Output: <out>/<name>/<name>_<frame index>.jpg
One sub-folder per recording; split_dataset.py relies on this to keep
frames from the same drive in the same split.

Usage
-----
  python3 extract_frames.py --bag run_01/ --out dataset/images --every 10
  python3 extract_frames.py --video lane_debug.mp4 --out dataset/images --name lab_day2
  python3 extract_frames.py --bag run_01/ --out dataset/images \\
          --windows run_01_failures.json --pad 1.0 --every 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def iter_bag(path, topic):
    from bag_io import image_msg_to_bgr, iter_messages
    for t_ns, msg in iter_messages(path, topic):
        yield t_ns, image_msg_to_bgr(msg)


def iter_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f'Cannot open video {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield int(i / fps * 1e9), frame   # time from video start, in ns
        i += 1
    cap.release()


def load_windows(path, pad):
    """Return [(start_ns, end_ns), ...] from a lane_failure_report JSON."""
    with open(path, encoding='utf-8') as f:
        report = json.load(f)
    base = report.get('start_ns')
    if base is None:
        print('[warn] report has no start_ns (made from --text); assuming it '
              'starts at the same moment as this input.')
    return [(w['start_s'] - pad, w['end_s'] + pad) for w in report['windows']], base


def thumb(img):
    small = cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--bag', help='ros2 bag directory or file')
    src.add_argument('--video', help='video file (mp4, avi, ...)')
    ap.add_argument('--topic', default='/oak/rgb/image_raw',
                    help='image topic in the bag (raw or compressed)')
    ap.add_argument('--out', required=True, help='output images root')
    ap.add_argument('--name', help='recording name (default: bag/video file name)')
    ap.add_argument('--every', type=int, default=10, help='keep every Nth frame')
    ap.add_argument('--min-diff', type=float, default=4.0,
                    help='skip frames whose mean grey-level change vs the last '
                         'kept frame is below this (0 = keep all)')
    ap.add_argument('--no-rotate', action='store_true',
                    help='do NOT rotate 180 deg (only if already upright)')
    ap.add_argument('--windows', help='JSON from lane_failure_report.py')
    ap.add_argument('--pad', type=float, default=0.5,
                    help='seconds added before/after each window')
    ap.add_argument('--max-frames', type=int, default=0, help='stop after N (0 = no limit)')
    ap.add_argument('--quality', type=int, default=95, help='JPEG quality')
    args = ap.parse_args()

    source = args.bag or args.video
    name = args.name or Path(source.rstrip('/')).stem
    out_dir = Path(args.out) / name
    os.makedirs(out_dir, exist_ok=True)

    windows, base_ns = (load_windows(args.windows, args.pad) if args.windows
                        else (None, None))
    frames = iter_bag(args.bag, args.topic) if args.bag else iter_video(args.video)

    first_ns = None
    seen = kept = skipped_dup = 0
    last_thumb = None
    for idx, (t_ns, img) in enumerate(frames):
        if img is None:
            continue
        first_ns = t_ns if first_ns is None else first_ns
        if windows is not None:
            t_rel = (t_ns - (base_ns if base_ns is not None else first_ns)) / 1e9
            if not any(a <= t_rel <= b for a, b in windows):
                continue
        seen += 1
        if (seen - 1) % args.every:
            continue

        if not args.no_rotate:
            img = cv2.rotate(img, cv2.ROTATE_180)

        th = thumb(img)
        if (args.min_diff > 0 and last_thumb is not None
                and float(np.mean(np.abs(th - last_thumb))) < args.min_diff):
            skipped_dup += 1
            continue
        last_thumb = th

        cv2.imwrite(str(out_dir / f'{name}_{idx:06d}.jpg'), img,
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        kept += 1
        if args.max_frames and kept >= args.max_frames:
            break

    print(f'{name}: saved {kept} frames to {out_dir}  '
          f'(considered {seen}, skipped {skipped_dup} near-duplicates)')


if __name__ == '__main__':
    main()
