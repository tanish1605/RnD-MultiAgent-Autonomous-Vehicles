#!/usr/bin/env python3
"""
sign_distance_sim.py.

Monte Carlo comparison of four ways to measure a traffic sign's distance:

  box       depth from the height of the detection box (baseline)
  stereo    OAK-D stereo depth (simulated from its noise model)
  geometry  shape fit on the segmentation outline (sign_geometry.py)
  fused     geometry + stereo, weighted by their uncertainties

Each trial renders a sign through the real OAK-D calibration at a known
distance and angle, then disturbs it the way reality would:
  - the segmentation outline is off by up to about a pixel (blur + random
    threshold moves the edge in or out)
  - the sign is not perfectly upright (random roll and pitch)
  - the real sign differs from its nominal size (--size-error)
  - stereo depth has the error of a stereo pair, growing with distance
    squared, and sometimes gives no depth at all (--stereo-dropout)

This is a SIMULATION: it shows how the methods behave under these
assumptions, not how they perform on the car. Check the assumptions against
real measurements (tape measure) before quoting numbers.

Usage (from the object_detection package folder)
-----
  python3 tools/sign_distance_sim.py
  python3 tools/sign_distance_sim.py --trials 100 --disparity-sigma 0.5 --csv sim.csv
"""

import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from object_detection.sign_geometry import (  # noqa: E402
    box_size_distance, estimate_pose, fuse_distance, load_config, pose_vectors,
    render_mask, SignSpec, stereo_sigma)

DEFAULT_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'config', 'sign_geometry.yaml')
METHODS = ('box', 'stereo', 'geometry', 'fused')


def noisy_mask(spec, camera, z, yaw, x, rng, args):
    """Render the sign with realistic imperfections."""
    true_spec = SignSpec(spec.shape, spec.width * (1 + rng.normal(0, args.size_error)),
                         spec.height * (1 + rng.normal(0, args.size_error)))
    rvec, tvec = pose_vectors(z, yaw, x=x, y=rng.uniform(-0.1, 0.1),
                              roll_deg=rng.normal(0, args.roll))
    if args.pitch:
        pitch = math.radians(rng.normal(0, args.pitch))
        rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)],
                       [0, math.sin(pitch), math.cos(pitch)]])
        rvec = cv2.Rodrigues(rx @ cv2.Rodrigues(rvec)[0])[0]
    mask = render_mask(true_spec, camera, rvec, tvec).astype(np.float32)
    # move the edge in or out by up to about a pixel
    blurred = cv2.GaussianBlur(mask, (0, 0), args.edge_blur)
    return (blurred > rng.uniform(0.3, 0.7)).astype(np.uint8), rvec, tvec


def run(args):
    camera, _, stereo_cfg = load_config(args.config)
    focal = stereo_cfg.get('focal_px', 798.2)
    baseline = stereo_cfg.get('baseline_m', 0.0752)
    rng = np.random.default_rng(args.seed)
    rows = []
    for shape in args.shapes:
        spec = SignSpec(shape, args.size, args.size * 1.2 if shape == 'rectangle' else None,
                        size_rel_sigma=max(args.size_error, 0.01))
        for z in args.distances:
            for yaw in args.yaws:
                for _ in range(args.trials):
                    x = rng.uniform(-0.25, 0.25) * z
                    mask, _, _ = noisy_mask(spec, camera, z, yaw * rng.choice([-1, 1]), x,
                                            rng, args)
                    if not mask.any():
                        continue
                    bx, by, bw, bh = cv2.boundingRect(mask)
                    pose = estimate_pose(mask, spec, camera)

                    s_sigma = stereo_sigma(z, focal, baseline, args.disparity_sigma)
                    stereo = None
                    if rng.random() >= args.stereo_dropout:
                        stereo = (z + rng.normal(0, s_sigma), s_sigma)
                    geometry = (pose.z, pose.sigma_z) if pose else None
                    fused = fuse_distance(geometry, stereo)

                    est = {
                        'box': box_size_distance([bx, by, bx + bw, by + bh], spec, camera),
                        'stereo': stereo[0] if stereo else None,
                        'geometry': geometry[0] if geometry else None,
                        'fused': fused.z if fused else None,
                    }
                    for m in METHODS:
                        rows.append({
                            'shape': shape, 'z': z, 'yaw': yaw, 'method': m,
                            'estimate': est[m],
                            'error_pct': None if est[m] is None else 100 * (est[m] - z) / z,
                        })
    return rows


def summarise(rows, key):
    """Median |error| % and missing rate per (key value, method)."""
    table = {}
    for r in rows:
        cell = table.setdefault((r[key], r['method']), [[], 0])
        if r['error_pct'] is None:
            cell[1] += 1
        else:
            cell[0].append(abs(r['error_pct']))
    return table


def print_table(table, key, values):
    head = f'{key:>9s} | ' + ' | '.join(f'{m:>18s}' for m in METHODS)
    print(head)
    print('-' * len(head))
    for v in values:
        cells = []
        for m in METHODS:
            errs, missing = table.get((v, m), ([], 0))
            if not errs:
                cells.append(f'{"no estimate":>18s}')
                continue
            miss = 100 * missing / (missing + len(errs))
            cells.append(f'{np.median(errs):6.2f}% p90 {np.percentile(errs, 90):5.1f}%'
                         + ('*' if miss >= 1 else ' '))
        print(f'{v:>9} | ' + ' | '.join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=DEFAULT_CFG)
    ap.add_argument('--shapes', nargs='+',
                    default=['circle', 'triangle_up', 'square', 'octagon'])
    ap.add_argument('--distances', type=float, nargs='+', default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    ap.add_argument('--yaws', type=float, nargs='+', default=[0, 20, 40],
                    help='sign turned by this many degrees (random direction)')
    ap.add_argument('--trials', type=int, default=40)
    ap.add_argument('--size', type=float, default=0.10, help='sign width (m)')
    ap.add_argument('--size-error', type=float, default=0.02,
                    help='relative difference between real and nominal size')
    ap.add_argument('--roll', type=float, default=2.0, help='sign roll sigma (deg)')
    ap.add_argument('--pitch', type=float, default=3.0, help='sign/camera pitch sigma (deg)')
    ap.add_argument('--edge-blur', type=float, default=1.0,
                    help='outline noise scale (px); larger = worse segmentation')
    ap.add_argument('--disparity-sigma', type=float, default=0.25,
                    help='stereo disparity noise (px at the stereo resolution)')
    ap.add_argument('--stereo-dropout', type=float, default=0.05,
                    help='chance that stereo gives no depth for the sign')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--csv', help='write every trial to this CSV file')
    args = ap.parse_args()

    rows = run(args)
    print(f'\nMedian |distance error| (p90 = 90th percentile), {args.trials} trials per cell,'
          f' shapes {", ".join(args.shapes)}; * = some trials gave no estimate\n')
    print_table(summarise(rows, 'z'), 'distance', args.distances)
    print()
    print_table(summarise(rows, 'yaw'), 'yaw deg', args.yaws)
    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nWrote {len(rows)} rows to {args.csv}')


if __name__ == '__main__':
    main()
