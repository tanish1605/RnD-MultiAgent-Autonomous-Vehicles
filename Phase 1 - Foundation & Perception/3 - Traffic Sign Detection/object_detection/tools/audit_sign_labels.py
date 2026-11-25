#!/usr/bin/env python3
"""
audit_sign_labels.py.

Find likely label errors in a YOLO segmentation (polygon) sign dataset by
checking every outline against the usual shape of its class.

Prints:
  - every class with its most common file-name prefix and its usual shape
  - how many objects got each flag
  - how well the geometry flags find the file-name mismatches (an
    independent check), as recall and precision
  - a ranked list of labels to review, most suspicious first
and optionally writes everything to a CSV.

Only the label files are needed; images are not read. Augmented copies of
an image (Roboflow '.rf.<hash>' names) are checked once.

Usage (from the object_detection package folder)
-----
  python3 tools/audit_sign_labels.py path/to/export/train/labels
  python3 tools/audit_sign_labels.py train/labels --csv audit.csv --review 50
  python3 tools/audit_sign_labels.py train/labels --names data.yaml   # use class names
"""

import argparse
from collections import Counter, defaultdict
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from object_detection.label_audit import (  # noqa: E402
    audit, GEOMETRY_FLAGS, name_prefix, read_labels, score_against_names)

# review order: most convincing problems first
RANK = ['name_mismatch', 'shape_mismatch', 'orientation_mismatch', 'self_crossing',
        'poor_fit', 'box_label', 'tiny']


def load_names(path):
    import yaml
    with open(path, encoding='utf-8') as f:
        names = yaml.safe_load(f).get('names', [])
    return dict(enumerate(names)) if isinstance(names, list) else dict(names)


def rank_key(f):
    first = min((RANK.index(x) for x in f.flags), default=len(RANK))
    return (first, -f.margin)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('labels', help='folder with the YOLO segmentation .txt label files')
    ap.add_argument('--names', help='data.yaml of the export, to show class names')
    ap.add_argument('--image-size', type=int, nargs=2, default=[1000, 1000],
                    metavar=('W', 'H'), help='image size in pixels (only affects pixel numbers)')
    ap.add_argument('--min-margin', type=float, default=0.03,
                    help='IoU by which another shape must fit better to flag a shape mismatch')
    ap.add_argument('--review', type=int, default=30, help='how many labels to list for review')
    ap.add_argument('--csv', help='write every object and its flags to this CSV file')
    args = ap.parse_args()

    objects, empty, copy_mismatch = read_labels(args.labels, tuple(args.image_size))
    if not objects:
        sys.exit(f'No polygon labels found in {args.labels}')
    names = load_names(args.names) if args.names else {}
    findings, table = audit(objects, image_area=args.image_size[0] * args.image_size[1],
                            min_margin=args.min_margin)
    sources = {o.source for o in objects} | set(empty)

    print(f'\n{len(sources)} images ({len(empty)} without labels), {len(objects)} objects; '
          f'augmented copies that disagree: {len(copy_mismatch)}')

    # ── classes ─────────────────────────────────────────────────────────
    prefixes = defaultdict(Counter)
    counts = Counter()
    for o in objects:
        prefixes[o.class_id][name_prefix(o.source)] += 1
        counts[o.class_id] += 1
    print(f'\n{"class":>5} {"name":22s} {"objects":>7}  {"usual shape":14s} {"share":>5}  '
          'file-name prefixes')
    for cls in sorted(counts):
        shape, share = table.get(cls, (None, 0))
        pre = ', '.join(f'{p} {n}' for p, n in prefixes[cls].most_common(3))
        print(f'{cls:>5} {str(names.get(cls, "")):22s} {counts[cls]:>7}  '
              f'{str(shape or "unclear"):14s} {share:>5.0%}  {pre}')

    # ── flags ───────────────────────────────────────────────────────────
    flag_counts = Counter(x for f in findings for x in f.flags)
    print('\nflags: ' + ', '.join(f'{k} {flag_counts[k]}' for k in RANK if flag_counts[k]))
    strength = Counter(f.strength for f in findings if 'shape_mismatch' in f.flags)
    print('shape mismatches by strength: '
          + ', '.join(f'{k} {strength[k]}' for k in ('strong', 'medium') if strength[k]))

    s = score_against_names(findings)
    if s['name_mismatches']:
        print(f'\nfile-name check: {s["name_mismatches"]} labels whose class differs from '
              f'their file name. Geometry flagged {s["found"]} of them '
              f'(recall {s["recall"]:.0%}) among {s["geometry_flags"]} geometry flags; '
              f'the other {s["extra"]} are labels the names cannot check.')

    # ── review list ─────────────────────────────────────────────────────
    flagged = sorted((f for f in findings if f.flags), key=rank_key)
    print(f'\nReview first ({min(args.review, len(flagged))} of {len(flagged)} flagged labels):')
    print(f'  {"image":34s} {"class":>5} {"outline":13s} {"class usual":13s} flags')
    for f in flagged[:args.review]:
        shape = f.fit.shape if f.fit else '-'
        note = f' ({f.strength})' if 'shape_mismatch' in f.flags else ''
        extra = f'  {f.detail}' if f.detail else ''
        print(f'  {f.obj.source[:34]:34s} {f.obj.class_id:>5} {shape:13s} '
              f'{str(f.class_shape):13s} {", ".join(f.flags)}{note}{extra}')

    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['image', 'label_files', 'class', 'outline_shape', 'class_shape',
                        'iou_round', 'iou_triangle', 'iou_quad', 'margin', 'solidity',
                        'flags', 'detail'])
            for f in findings:
                iou = f.fit.iou if f.fit else {}
                w.writerow([f.obj.source, ' '.join(f.obj.files), f.obj.class_id,
                            f.fit.shape if f.fit else '', f.class_shape,
                            *(f'{iou.get(k, 0):.3f}' for k in ('round', 'triangle', 'quad')),
                            f'{f.margin:.3f}', f'{f.fit.solidity:.3f}' if f.fit else '',
                            ' '.join(f.flags), f.detail])
        print(f'\nWrote {len(findings)} rows to {args.csv}')
    print('\nGeometry flags are candidates, not verdicts: open each image and decide. '
          f'Flag types: {", ".join(GEOMETRY_FLAGS)} come from the outline alone.')


if __name__ == '__main__':
    main()
