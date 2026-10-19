#!/usr/bin/env python3
"""
lane_failure_report.py
======================
How often does the classic lane_detection node struggle, and when?

Reads the /lane/status strings published by lane_detection, e.g.

    OK | Y:DETECT(812pts) | D:OFFSET(3pts) | Y_rec:n D_rec:Y | err:-12.4px | llc_ok:True

and reports, per category, the share of frames affected plus the longest
"bad" stretches. Those stretches are where segmentation should help most
and are the best frames to label first: pass the --json output to
extract_frames.py --windows.

Frame categories
----------------
  lane_lost        llc_ok False            -> no pixel error, controller is blind
  no_yellow        status NO_YELLOW        -> no yellow history at all
  yellow_degraded  Y mode REJ / LOW_PTS    -> this frame's yellow fit not used
  dash_fallback    D mode OFFSET           -> centre GUESSED as yellow + l2c_offset
  dash_degraded    D mode REJ / LOW_PTS    -> this frame's dash fit not used
A frame is "bad" if any category applies. Categories overlap.

Usage
-----
  # from a bag recorded with:  ros2 bag record /lane/status /oak/rgb/image_raw
  python3 lane_failure_report.py --bag my_run/ --json my_run_failures.json

  # or from text saved with:   ros2 topic echo /lane/status > status.txt
  python3 lane_failure_report.py --text status.txt --fps 30
"""

import argparse
import json
import re
import sys
from collections import Counter

STATUS_RE = re.compile(
    r'^(?P<status>\S+) \| '
    r'Y:(?P<y_mode>\w+)\((?P<y_pts>\d+)pts\) \| '
    r'D:(?P<d_mode>\w+)\((?P<d_pts>\d+)pts\) \| '
    r'Y_rec:(?P<y_rec>[Yn]) D_rec:(?P<d_rec>[Yn]) \| '
    r'err:(?P<err>[-+]?[\d.]+)px \| '
    r'llc_ok:(?P<llc_ok>True|False)'
)

CATEGORIES = ('lane_lost', 'no_yellow', 'yellow_degraded',
              'dash_fallback', 'dash_degraded')


def parse_status(text):
    """Parse one status string -> dict, or None if it does not match."""
    m = STATUS_RE.match(text.strip())
    if not m:
        return None
    d = m.groupdict()
    d['llc_ok'] = d['llc_ok'] == 'True'
    d['err'] = float(d['err'])
    return d


def categorize(s):
    """Return the set of failure categories that apply to one parsed status."""
    cats = set()
    if not s['llc_ok']:
        cats.add('lane_lost')
    if s['status'] == 'NO_YELLOW':
        cats.add('no_yellow')
    if s['y_mode'] in ('REJ', 'LOW_PTS'):
        cats.add('yellow_degraded')
    if s['d_mode'] == 'OFFSET':
        cats.add('dash_fallback')
    if s['d_mode'] in ('REJ', 'LOW_PTS'):
        cats.add('dash_degraded')
    return cats


# ── input readers: both yield (time_sec, status_text) ──────────────────────

def read_bag(path, topic, info):
    """info['start_ns'] is set to the first message's bag timestamp."""
    from bag_io import iter_messages
    for t_ns, msg in iter_messages(path, topic):
        info.setdefault('start_ns', t_ns)
        yield (t_ns - info['start_ns']) / 1e9, msg.data


def read_text(path, fps):
    """Parse `ros2 topic echo` output: lines like  data: '...'  split by ---."""
    i = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('data:'):
                value = line[len('data:'):].strip().strip('\'"')
                yield i / fps, value
                i += 1


# ── analysis ───────────────────────────────────────────────────────────────

def find_windows(samples, merge_gap):
    """Group consecutive bad frames into windows; gaps <= merge_gap sec merge."""
    windows = []
    for t, cats in samples:
        if not cats:
            continue
        if windows and t - windows[-1]['end'] <= merge_gap:
            w = windows[-1]
            w['end'] = t
            w['frames'] += 1
            w['categories'].update(cats)
        else:
            windows.append({'start': t, 'end': t, 'frames': 1,
                            'categories': Counter(cats)})
    for w in windows:
        w['duration'] = w['end'] - w['start']
    return windows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--bag', help='ros2 bag directory or file')
    src.add_argument('--text', help='saved `ros2 topic echo /lane/status` output')
    ap.add_argument('--topic', default='/lane/status')
    ap.add_argument('--fps', type=float, default=30.0,
                    help='frame rate assumed for --text input (no timestamps)')
    ap.add_argument('--merge-gap', type=float, default=0.5,
                    help='merge bad stretches separated by <= this many seconds')
    ap.add_argument('--top', type=int, default=10, help='how many windows to list')
    ap.add_argument('--json', help='write windows + summary to this JSON file')
    args = ap.parse_args()

    info = {}
    reader = (read_bag(args.bag, args.topic, info) if args.bag
              else read_text(args.text, args.fps))

    samples, unparsed, errors = [], 0, []
    for t, text in reader:
        s = parse_status(text)
        if s is None:
            unparsed += 1
            continue
        samples.append((t, categorize(s)))
        if s['llc_ok']:
            errors.append(abs(s['err']))

    if not samples:
        sys.exit('No parsable /lane/status messages found.')

    total = len(samples)
    counts = Counter(c for _, cats in samples for c in cats)
    bad = sum(1 for _, cats in samples if cats)
    duration = samples[-1][0] - samples[0][0]

    print(f'\nFrames analysed : {total}  ({duration:.1f} s)'
          + (f'   [{unparsed} unparsed lines skipped]' if unparsed else ''))
    print(f'Healthy frames  : {total - bad:6d}  ({100 * (total - bad) / total:5.1f} %)')
    print(f'Bad frames      : {bad:6d}  ({100 * bad / total:5.1f} %)\n')
    print('Category            frames      %')
    for c in CATEGORIES:
        print(f'  {c:16s} {counts[c]:8d}  {100 * counts[c] / total:5.1f}')
    if errors:
        errors.sort()
        print(f'\n|pixel_error| when lane found: median {errors[len(errors) // 2]:.1f}px, '
              f'95th pct {errors[int(0.95 * (len(errors) - 1))]:.1f}px')

    windows = find_windows(samples, args.merge_gap)
    worst = sorted(windows, key=lambda w: w['duration'], reverse=True)[:args.top]
    print(f'\nLongest bad stretches ({len(windows)} total):')
    print('   start_s    end_s   dur_s  frames  main causes')
    for w in worst:
        causes = ', '.join(f'{c}:{n}' for c, n in w['categories'].most_common(3))
        print(f'  {w["start"]:8.2f} {w["end"]:8.2f} {w["duration"]:7.2f} '
              f'{w["frames"]:7d}  {causes}')

    if args.json:
        out = {
            'source': args.bag or args.text,
            # absolute bag time of start_s = 0 (None for --text input)
            'start_ns': info.get('start_ns'),
            'frames': total, 'bad_frames': bad, 'duration_s': duration,
            'category_counts': dict(counts),
            'windows': [{'start_s': w['start'], 'end_s': w['end'],
                         'frames': w['frames'],
                         'categories': dict(w['categories'])} for w in windows],
        }
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print(f'\nWrote {args.json}  (use with extract_frames.py --windows)')


if __name__ == '__main__':
    main()
