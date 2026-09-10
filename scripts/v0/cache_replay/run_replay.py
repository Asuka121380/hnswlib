#!/usr/bin/env python3
"""Build, record and run fixed-order native cache replay; Python is never timed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EXE = '.exe' if os.name == 'nt' else ''

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')

def machine():
    result = {'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count()}
    if hasattr(os, 'sched_getaffinity'):
        result['allowed_cpus'] = sorted(os.sched_getaffinity(0))
    if sys.platform.startswith('linux') and shutil.which('lscpu'):
        run = subprocess.run(['lscpu', '--json'], capture_output=True, text=True)
        if run.returncode == 0:
            result['lscpu'] = json.loads(run.stdout)
    return result

def command(args):
    subprocess.run([str(x) for x in args], check=True)

def compile_tool(repo, build, cxx):
    repo, build = Path(repo).resolve(), Path(build).resolve()
    build.mkdir(parents=True, exist_ok=True)
    headers = build / 'include' / 'hnswlib'
    headers.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for src in (repo / 'hnswlib').glob('*.h'):
        source_hashes[src.name] = digest(src)
        shutil.copyfile(src, headers / src.name)
    header = headers / 'edge_quant_v0.h'
    text = header.read_text(encoding='utf-8')
    old = '        const double query_inner_product =\n            lut_.innerProductUnchecked(edge);'
    if text.count(old) != 1:
        raise RuntimeError('PQ hook location changed; refusing to patch an unknown header')
    text = text.replace(old, '        ::cacheReplayPQ(edge.codeDataUnchecked(), exact_current_squared_distance);\n' + old)
    header.write_text('void cacheReplayPQ(const unsigned char*, double);\n' + text, encoding='utf-8')
    flags = ['-std=c++17', '-O3', '-march=native', '-fno-fast-math',
             '-DHNSWLIB_ENABLE_EDGE_QUANT_V0=1', '-DHNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING=1',
             '-DHNSWLIB_V0_STRICT_FP_CONTRACT_ENABLED=1', '-I', str(headers.parent)]
    for source, binary in [('replay.cpp', 'cache_replay'), ('fixture.cpp', 'fixture')]:
        command([cxx, *flags, HERE / source, '-o', build / (binary + EXE)])
    dump(build / 'build.json', {'repo': str(repo), 'compiler': str(cxx), 'flags': flags,
                              'headers_sha256': source_hashes,
                              'sources_sha256': {p: digest(HERE / p) for p in ['replay.cpp', 'fixture.cpp', 'counters.h']},
                              'compiler_version': subprocess.check_output([cxx, '--version'], text=True),
                              'machine': machine()})

def native(build, options):
    args = [Path(build).resolve() / ('cache_replay' + EXE)]
    for key, value in options.items():
        args += ['--' + key, str(value)]
    command(args)

def record(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {k: str(Path(getattr(args, k)).resolve()) for k in ['index', 'sidecar', 'queries']}
    options = dict(inputs, dimension=args.dimension, **{'query-start': args.query_start, 'query-count': args.query_count})
    hashes = {k: digest(v) for k, v in inputs.items()}
    capture = dict(ef=args.ef, k=args.k, beta=args.beta, retry=int(args.retry), prefetch=args.prefetch)
    native(args.build, dict(options, **capture, mode='record', trace=out / 'trace.bin', output=out / 'record.json', cpu=args.cpu))
    dump(out / 'manifest.json', {'schema': 1, 'inputs': inputs, 'sha256': hashes, 'options': options,
                               'capture': capture, 'trace_sha256': digest(out / 'trace.bin'),
                               'build': json.loads((Path(args.build) / 'build.json').read_text()),
                               'capture_binary_sha256': digest(Path(args.build) / ('cache_replay' + EXE))})
    print('Recorded:', out)

def replay(args):
    root = Path(args.trace_dir).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    for key, path in manifest['inputs'].items():
        if digest(path) != manifest['sha256'][key]:
            raise RuntimeError('Input changed: ' + path)
    if digest(root / 'trace.bin') != manifest['trace_sha256']:
        raise RuntimeError('Trace checksum mismatch')
    out = Path(args.out).resolve();out.mkdir(parents=True, exist_ok=False)
    record_info = json.loads((root / 'record.json').read_text())
    options = dict(manifest['options'])
    alternate = None
    if getattr(args, 'replay_sidecar', None):
        if not args.fold_bits:
            raise ValueError('--replay-sidecar requires explicit --fold-bits matching the new codebook')
        options['sidecar'] = str(Path(args.replay_sidecar).resolve())
        alternate = {'path': options['sidecar'], 'sha256': digest(options['sidecar'])}
    bits = args.fold_bits or [record_info['source_bits']]
    if any(b < 1 or b > 8 for b in bits):
        raise ValueError('bits must be between 1 and 8; native reader also validates codebook bounds')
    order = [(block, b, s) for block in range(args.blocks) for b in bits for s in args.spacing]
    random.Random(args.seed).shuffle(order)
    rows = []; reference = {}
    for block, b, spacing in order:
        path = out / f'b{block:02d}-bits{b}-spacing{spacing}.json'
        native(args.build, dict(options, mode='replay', trace=root / 'trace.bin', output=path,
                               spacing=spacing, counters=int(args.counters), **{'fold-bits': b, 'repeats': args.repeats, 'warmups': args.warmups, 'cpu': args.cpu}))
        result = json.loads(path.read_text())
        checksum = result['checksums'][0]
        if any(x != checksum for x in result['checksums']):
            raise RuntimeError('Checksum changed across repeats')
        if b in reference and reference[b] != checksum:
            raise RuntimeError('Layout changed numerical results; discard timings')
        reference[b] = checksum
        rows.append(dict(result, block=block, median_ns_per_query=statistics.median(result['elapsed_ns']) / result['queries']))
    summary = []
    for b in bits:
        for spacing in args.spacing:
            values = [r['median_ns_per_query'] for r in rows if r['lookup_bits'] == b and r['spacing'] == spacing]
            summary.append({'bits': b, 'spacing': spacing, 'median_ns_per_query': statistics.median(values),
                            'process_block_ns_per_query': values,
                            'folded_surrogate': next(r['folded_surrogate'] for r in rows if r['lookup_bits']==b)})
    dump(out / 'summary.json', {'metric': 'fixed-order replay ns/query; NOT end-to-end QPS',
                               'trace_manifest': manifest, 'run_build': json.loads((Path(args.build) / 'build.json').read_text()),
                               'run_binary_sha256': digest(Path(args.build) / ('cache_replay' + EXE)),
                               'alternate_sidecar': alternate, 'machine': machine(),
                               'cpu': args.cpu, 'counters': args.counters, 'seed': args.seed, 'warmups': args.warmups,
                               'order': order, 'summary': summary})
    for row in summary:
        print(row)

def smoke(args):
    root = Path(args.out).resolve();root.mkdir(parents=True, exist_ok=False)
    command([Path(args.build).resolve() / ('fixture' + EXE), root])
    rec = argparse.Namespace(build=args.build, out=root/'capture', index=root/'index.bin', sidecar=root/'sidecar.bin',
                             queries=root/'queries.fvecs', dimension=64, query_start=0, query_count=5,
                             ef=20, k=5, beta=1.4, retry=True, prefetch='legacy', cpu=-1)
    record(rec)
    # Also exercise the other search policy without retry.
    rec.out=root/'capture-no-retry';rec.retry=False;rec.prefetch='gate'
    record(rec)
    replay(argparse.Namespace(build=args.build, trace_dir=root/'capture', out=root/'timing', fold_bits=[8,6,4],
                              spacing=[1,4,16], blocks=2, repeats=2, warmups=1, seed=1234, cpu=-1, counters=False))
    # Broken trace must fail before entering the timed loop.
    trace = root/'capture'/'trace.bin'
    broken = root/'broken.bin';broken.write_bytes(trace.read_bytes()[:-1])
    options = dict(index=rec.index, sidecar=rec.sidecar, queries=rec.queries, dimension=64,
                   **{'query-count':5}, mode='replay', trace=broken, output=root/'should-not-exist.json')
    try:
        native(args.build, options)
    except subprocess.CalledProcessError:
        pass
    else:
        raise RuntimeError('Truncated trace accepted')
    print('PASS: real search capture, recording parity, 9 layout/bit cases, checksum equality, corrupt trace rejection')

def positive(s):
    v=int(s)
    if v<=0:raise argparse.ArgumentTypeError('must be positive')
    return v

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action', required=True)
    p=sub.add_parser('build');p.add_argument('--repo',required=True);p.add_argument('--build',required=True);p.add_argument('--cxx',default='g++')
    for name in ['record','replay','smoke']:
        p=sub.add_parser(name);p.add_argument('--build',required=True);p.add_argument('--out',required=True)
        if name=='smoke':continue
        p.add_argument('--cpu',type=int,default=-1)
        if name=='record':
            for key in ['index','sidecar','queries']:p.add_argument('--'+key,required=True)
            p.add_argument('--dimension',type=positive,default=960);p.add_argument('--query-start',type=int,default=0)
            p.add_argument('--query-count',type=positive,default=100);p.add_argument('--ef',type=positive,default=500)
            p.add_argument('--k',type=positive,default=10);p.add_argument('--beta',type=float,default=1.4)
            p.add_argument('--retry',action=argparse.BooleanOptionalAction,default=True)
            p.add_argument('--prefetch',choices=['legacy','gate'],default='legacy')
        else:
            p.add_argument('--trace-dir',required=True);p.add_argument('--fold-bits',type=positive,nargs='+')
            p.add_argument('--replay-sidecar',help='Separately trained sidecar for the identical index/edge ordering; requires --fold-bits')
            p.add_argument('--spacing',type=int,choices=[1,4,16],nargs='+',default=[1,4,16])
            p.add_argument('--blocks',type=positive,default=5);p.add_argument('--repeats',type=positive,default=5)
            p.add_argument('--warmups',type=int,default=2);p.add_argument('--seed',type=int,default=1234)
            p.add_argument('--counters',action='store_true',help='Linux diagnostic pass: cycles, instructions, L1D read misses')
    args=parser.parse_args()
    if getattr(args,'query_start',0)<0 or getattr(args,'warmups',0)<0:parser.error('counts must be nonnegative')
    if args.action=='build':compile_tool(args.repo,args.build,args.cxx)
    else:globals()[args.action](args)

if __name__=='__main__':
    main()
