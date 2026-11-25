"""Unit tests for dataset_split.py and tools/prepare_sign_dataset.py (no ROS needed)."""

import os
import subprocess
import sys

import yaml

from object_detection.dataset_split import collect, relabel_lines, Sample, stratified_split

TOOL = os.path.join(os.path.dirname(__file__), '..', 'tools', 'prepare_sign_dataset.py')


def make_export(root, n_per_class=10, copies=3, classes=(0, 1, 2)):
    """Fake Roboflow export: every image exported `copies` times, labels identical."""
    img_dir, lbl_dir = root / 'train' / 'images', root / 'train' / 'labels'
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for cls in classes:
        for i in range(n_per_class):
            for k in range(copies):
                stem = f'sign{cls}_{i:05d}_jpg.rf.{cls}{i}{k}abc'
                (img_dir / f'{stem}.jpg').write_bytes(b'not really a jpeg')
                (lbl_dir / f'{stem}.txt').write_text(f'{cls} 0.4 0.4 0.6 0.4 0.5 0.6\n')
    (root / 'data.yaml').write_text(yaml.safe_dump({'names': ['a', 'b', 'c']}))


def test_collect_groups_copies(tmp_path):
    make_export(tmp_path)
    samples = collect(str(tmp_path))
    assert len(samples) == 30
    assert all(len(s.copies) == 3 for s in samples)
    assert {s.main_class for s in samples} == {0, 1, 2}


def test_split_keeps_copies_together_and_covers_every_class(tmp_path):
    make_export(tmp_path, n_per_class=20)
    samples = collect(str(tmp_path))
    split = stratified_split(samples, 0.15, 0.15, seed=1)
    assert set(split) == {s.source for s in samples}          # every image once
    for cls in (0, 1, 2):
        got = {split[s.source] for s in samples if s.main_class == cls}
        assert got == {'train', 'valid', 'test'}
    counts = {sp: sum(1 for v in split.values() if v == sp) for sp in ('train', 'valid', 'test')}
    assert counts == {'train': 42, 'valid': 9, 'test': 9}


def test_split_is_reproducible():
    samples = [Sample(f's{i}', classes=[i % 3]) for i in range(30)]
    assert stratified_split(samples, seed=7) == stratified_split(samples, seed=7)


def test_images_without_objects_are_split_too():
    samples = ([Sample(f'e{i}') for i in range(20)]
               + [Sample(f's{i}', classes=[0]) for i in range(20)])
    split = stratified_split(samples)
    assert {split[f'e{i}'] for i in range(20)} == {'train', 'valid', 'test'}


def test_relabel_lines():
    lines = ['3 0.1 0.2 0.3 0.4 0.5 0.6\n', '\n']
    assert relabel_lines(lines, 7) == ['7 0.1 0.2 0.3 0.4 0.5 0.6\n']


def test_prepare_tool_end_to_end(tmp_path):
    export, out = tmp_path / 'export', tmp_path / 'out'
    make_export(export)
    run = subprocess.run([sys.executable, TOOL, str(export), str(out),
                          '--relabel', 'sign1_00003_jpg=2', '--drop', 'sign0_00000_jpg'],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr

    def stems(split):
        return [os.path.splitext(n)[0] for n in os.listdir(out / split / 'images')]

    seen = {}
    for split in ('train', 'valid', 'test'):
        for stem in stems(split):
            src = stem.split('.rf.')[0]
            assert seen.setdefault(src, split) == split       # never in two splits
            assert (out / split / 'labels' / f'{stem}.txt').exists()
    assert 'sign0_00000_jpg' not in seen
    n_train = len(stems('train'))
    n_train_src = sum(1 for v in seen.values() if v == 'train')
    assert n_train == 3 * n_train_src                         # all copies in train
    for split in ('valid', 'test'):                           # one copy elsewhere
        assert len(stems(split)) == sum(1 for v in seen.values() if v == split)

    relabelled = [p for sp in ('train', 'valid', 'test')
                  for p in (out / sp / 'labels').glob('sign1_00003_jpg*')]
    assert relabelled and all(p.read_text().startswith('2 ') for p in relabelled)
    data = yaml.safe_load((out / 'data.yaml').read_text())
    assert data['names'] == ['a', 'b', 'c'] and data['val'] == 'valid/images'
    # the export itself is untouched
    assert (export / 'train' / 'labels').glob('sign1_00003_jpg*')
    original = next((export / 'train' / 'labels').glob('sign1_00003_jpg*'))
    assert original.read_text().startswith('1 ')


def test_prepare_tool_rejects_unknown_image(tmp_path):
    export = tmp_path / 'export'
    make_export(export)
    run = subprocess.run([sys.executable, TOOL, str(export), str(tmp_path / 'o'),
                          '--drop', 'no_such_image', '--dry-run'], capture_output=True, text=True)
    assert run.returncode != 0 and 'no_such_image' in run.stderr
