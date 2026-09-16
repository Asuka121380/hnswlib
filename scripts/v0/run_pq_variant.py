#!/usr/bin/env python3
"""Train, encode and validate one PQ variant; never overwrite an existing run.

Uses the existing byte-per-subquantizer V0 format. No query data or beta tuning.
Run inside a Slurm allocation, or use submit_pq_variant.sh. --dry-run is read-only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from train_pq_codebook import _write_json_atomic
from validate_edge_direction_sample import validate as validate_sample
from validate_v0_sidecar import validate as validate_sidecar
from v0pq_format import sha256_file

SCRIPT = Path(__file__).resolve().parent
REPO = SCRIPT.parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pq-m', type=int, required=True)
    p.add_argument('--nbits', type=int, required=True)
    for name in ('directions', 'sample-manifest', 'index-path', 'encoder', 'inspector', 'output-root'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--split-seed', type=int, default=29)
    p.add_argument('--iterations', type=int, default=25)
    p.add_argument('--nredo', type=int, default=1)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--validation-fraction', type=float, default=0.1)
    p.add_argument('--min-points-per-centroid', type=int, default=39)
    p.add_argument('--max-points-per-centroid', type=int, default=0,
                   help='0 disables Faiss internal subsampling by deriving a sufficient cap')
    p.add_argument('--block-size', type=int, default=4096)
    p.add_argument('--faiss-version', default='unknown', help='C++ encoder Faiss version')
    p.add_argument('--faiss-source-commit', default='unknown')
    p.add_argument('--dry-run', action='store_true')
    return p


def make_plan(a):
    if a.pq_m <= 0 or not 1 <= a.nbits <= 8:
        raise ValueError('pq-m must be positive; nbits must be in 1..8')
    if min(a.threads, a.iterations, a.nredo, a.block_size, a.min_points_per_centroid) <= 0:
        raise ValueError('training/encoding counts must be positive')
    if not 0 < a.validation_fraction < 1 or a.max_points_per_centroid < 0:
        raise ValueError('invalid split or centroid sample limit')
    if a.output_root.exists():
        raise ValueError(f'refusing existing output root: {a.output_root}')
    for name in ('directions', 'sample_manifest', 'index_path', 'encoder', 'inspector'):
        path = getattr(a, name).resolve()
        if not path.is_file():
            raise ValueError(f'missing {name}: {path}')
        setattr(a, name, path)
    a.output_root = a.output_root.resolve()
    sample = validate_sample(a.sample_manifest, directions_override=a.directions)
    manifest = json.loads(a.sample_manifest.read_text(encoding='utf-8'))
    if sample['dimension'] % a.pq_m:
        raise ValueError('dimension must be divisible by pq-m')
    # Do not silently train on directions from a different graph.
    index_sha = sha256_file(a.index_path)
    if manifest.get('index_sha256') != index_sha:
        raise ValueError('sample manifest index_sha256 missing or different from index')
    count = int(sample['sample_count'])
    train_count = count - max(1, int(count * a.validation_fraction))
    ksub = 1 << a.nbits
    if train_count < ksub * a.min_points_per_centroid:
        raise ValueError('insufficient training rows for min-points-per-centroid')
    cap = a.max_points_per_centroid or max(a.min_points_per_centroid, math.ceil(train_count / ksub))
    if cap < a.min_points_per_centroid:
        raise ValueError('max-points-per-centroid must be >= min-points-per-centroid')
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip()
    root = a.output_root
    cb = root / 'training/codebook.v0pq'
    sidecar = root / 'encoding/sidecar.v0meta'
    def py(name, *args):
        return [sys.executable, str(SCRIPT / name), *map(str, args)]
    commands = [
        ('train', py('train_pq_codebook.py', '--directions', a.directions,
            '--manifest', a.sample_manifest, '--M-pq', a.pq_m, '--nbits', a.nbits,
            '--seed', a.seed, '--split-seed', a.split_seed, '--iterations', a.iterations,
            '--nredo', a.nredo, '--threads', a.threads, '--validation-fraction', a.validation_fraction,
            '--min-points-per-centroid', a.min_points_per_centroid,
            '--max-points-per-centroid', cap, '--producer-git-commit', commit,
            '--output-codebook', cb, '--output-metrics', root / 'training/training_metrics.json',
            '--output-errors', root / 'training/validation_errors.f32')),
        ('codebook-validation', py('validate_v0pq.py', '--codebook', cb,
            '--expected-manifest', a.sample_manifest, '--expected-directions', a.directions)),
        ('reconstruction', py('compare_codebook_reconstruction.py', '--codebook', cb,
            '--directions', a.directions, '--manifest', a.sample_manifest, '--sample-rows', 1024,
            '--faiss-check', '--output-metrics', root / 'training/reconstruction_check.json')),
        ('encode', list(map(str, [a.encoder, '--index-path', a.index_path, '--codebook-path', cb,
            '--output-sidecar', sidecar, '--output-metrics', root / 'encoding/encoding_metrics.json',
            '--block-size', a.block_size, '--producer-git-commit', commit,
            '--faiss-version', a.faiss_version, '--faiss-source-commit', a.faiss_source_commit]))),
        ('inspect', list(map(str, [a.inspector, '--sidecar', sidecar, '--index-path', a.index_path,
            '--sample-records', 5]))),
    ]
    return {
        'schema_version': 1, 'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'producer_git_commit': commit, 'working_tree_dirty': bool(dirty),
        'index_sha256': index_sha, 'sample_manifest_sha256': sha256_file(a.sample_manifest),
        'directions_sha256': sample['directions_sha256'],
        'encoder_sha256': sha256_file(a.encoder), 'inspector_sha256': sha256_file(a.inspector),
        'code_storage': 'uint8_per_subquantizer', 'sidecar_code_bytes': a.pq_m,
        'faiss_packed_code_bytes': (a.pq_m * a.nbits + 7) // 8,
        'lut_bytes_float32': a.pq_m * ksub * 4,
        'resolved_max_points_per_centroid': cap,
        'training_rows': train_count, 'faiss_training_subsampling_expected': train_count > ksub * cap,
        'commands': commands,
    }


def main():
    p = parser()
    a = p.parse_args()
    try:
        plan = make_plan(a)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        p.error(str(e))
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    root = a.output_root
    root.mkdir(parents=True, exist_ok=False)
    for directory in ('training', 'encoding', 'logs'):
        (root / directory).mkdir()
    _write_json_atomic(root / 'resolved_config.json', plan)
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
    try:
        for stage, command in plan['commands']:
            print(f'pq_variant_stage={stage}', flush=True)
            with (root / 'logs' / f'{stage}.log').open('x', encoding='utf-8') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        report = validate_sidecar(root / 'encoding/sidecar.v0meta',
            expected_v0pq=root / 'training/codebook.v0pq', expected_index_sha256=plan['index_sha256'])
        report['sidecar_file_sha256'] = sha256_file(root / 'encoding/sidecar.v0meta')
        _write_json_atomic(root / 'encoding/validation_report.json', report)
        (root / 'COMPLETE').write_text('validated\n', encoding='utf-8')
    except Exception as e:
        _write_json_atomic(root / 'FAILED.json', {'error': str(e)})
        raise
    print(f'pq_variant_complete M={a.pq_m} b={a.nbits} output={root}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
