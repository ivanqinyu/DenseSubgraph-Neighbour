#!/usr/bin/env python3
"""Run one published solver on a validated NPY graph (see README.md)."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

METHODS = {
    'QAD': 'qad', 'CO': 'co', 'DtC': 'dtc', 'MinimumPeeling': 'peeling',
    'UnconstrainedPeeling': 'peeling', 'Greedy++': 'iterative_peeling',
    'FISTA': 'fista', 'InternalUpper': 'internal',
    'InternalLower': 'internal', 'Demographics': 'demographics',
    'OPT': 'opt', 'Goldberg': 'goldberg',
}


def validate_graph(folder):
    edges = np.load(folder / 'edges.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(folder / 'labels.npy', mmap_mode='r', allow_pickle=False)
    if edges.dtype != np.dtype('<u4') or edges.ndim != 2 or edges.shape[1] != 2 or not edges.flags.c_contiguous:
        raise ValueError('edges.npy must be a C-contiguous little-endian uint32 array of shape (M, 2)')
    if labels.dtype != np.dtype('<u2') or labels.ndim != 1 or not labels.flags.c_contiguous or not len(labels):
        raise ValueError('labels.npy must be a nonempty little-endian uint16 vector of shape (N,)')
    n, m = len(labels), len(edges)
    if n >= 2**32:
        raise ValueError('N must be less than 2^32')
    groups = np.unique(labels)
    if len(groups) > 65535:
        raise ValueError('At most 65535 groups are supported (IDs 0..65534)')
    if not np.array_equal(groups, np.arange(len(groups))):
        raise ValueError('Labels must be compact group IDs 0,...,C-1; use prepare_data.py')
    prev = None
    for start in range(0, m, 1_000_000):
        block = edges[start:start + 1_000_000]
        if np.any(block[:, 0] >= block[:, 1]) or np.any(block[:, 1] >= n):
            raise ValueError('Edges must have 0 <= u < v < N (no self-loops); use prepare_data.py')
        keys = (block[:, 0].astype(np.uint64) << 32) | block[:, 1]
        if np.any(keys[1:] <= keys[:-1]) or (prev is not None and keys[0] <= prev):
            raise ValueError('Edges must be lexicographically sorted and unique; use prepare_data.py')
        prev = keys[-1]
    return edges, labels


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algorithm', choices=METHODS, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New or empty output directory')
    p.add_argument('--build-dir', type=Path, default=root / 'build')
    p.add_argument('--targets', default='20',
                   help='External coverage percentages from 10,20,30,40; DtC/OPT also allow 1..100')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--beta', type=float, default=0.6, help='InternalUpper maximum group share')
    p.add_argument('--tau', type=float, default=0.2, help='InternalLower minimum population fraction')
    p.add_argument('--rounds', type=int, default=5, help='Greedy++ rounds')
    p.add_argument('--iterations', type=int, default=50, help='FISTA iterations')
    p.add_argument('--mu', type=float, default=256, help='CO penalty weight')
    p.add_argument('--learning-rate', type=float, default=.02, help='CO Adam learning rate')
    p.add_argument('--steps', type=int, default=1000, help='CO Adam steps')
    p.add_argument('--time-limit', type=float, default=None,
                   help='DtC/OPT solver budget per target in seconds (default 30); not a process deadline')
    p.add_argument('--stop-factor', type=float, default=6)
    a = p.parse_args()
    try:
        targets = [int(x) for x in a.targets.split(',')]
        if not targets or len(set(targets)) != len(targets) or any(t < 1 or t > 100 for t in targets):
            raise ValueError('--targets must be distinct integer percentages from 1 to 100')
        restricted = ('QAD', 'CO', 'MinimumPeeling', 'Greedy++', 'FISTA')
        if a.algorithm in restricted and any(t not in (10, 20, 30, 40) for t in targets):
            raise ValueError(f'{a.algorithm} supports only --targets 10,20,30,40')
        if a.seed < 0 or a.rounds < 1 or a.iterations < 1 or not math.isfinite(a.mu) or a.mu < 0:
            raise ValueError('Invalid seed, round/iteration count, or mu')
        for name in ('beta', 'tau'):
            x = getattr(a, name)
            if not math.isfinite(x) or not 0 < x <= 1 or not math.isclose(x * 100, round(x * 100), abs_tol=1e-8):
                raise ValueError(f'--{name} must be in (0,1] with integer percentage precision')
        if a.time_limit is not None and a.algorithm not in ('DtC', 'OPT'):
            raise ValueError('--time-limit is supported only by DtC and OPT')
        seconds = 30 if a.time_limit is None else a.time_limit
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('--time-limit must be positive and finite')
        if not math.isfinite(a.stop_factor) or (a.stop_factor != 0 and a.stop_factor < 1):
            raise ValueError('--stop-factor must be 0 or >= 1')
        edges, labels = validate_graph(a.input)
        n, m = len(labels), len(edges)
        if a.algorithm == 'OPT' and n > 1024:
            raise ValueError('OPT supports at most 1024 vertices and may be exponentially slow')
        binary = (a.build_dir / 'bin' / METHODS[a.algorithm]).resolve()
        if not binary.is_file():
            raise ValueError(f'Missing executable {binary}; build with CMake first')
        if a.output.exists() and any(a.output.iterdir()):
            raise ValueError('Output directory must be empty to prevent overwriting prior runs')
        a.output.mkdir(parents=True, exist_ok=True)
        target_text = ','.join(map(str, targets))
        cmd = [str(binary), str(a.input.resolve()), str(n), str(m), str(a.output.resolve())]
        method = a.algorithm
        if method == 'QAD':
            cmd += ['QAD', str(a.seed), target_text]
        elif method == 'CO':
            cmd += [str(a.seed), str(a.mu), target_text, 'simple', str(a.learning_rate), str(a.steps)]
        elif method == 'DtC':
            cmd += ['dtc', str(a.seed), '--targets', target_text, '--time-limit', str(seconds),
                    '--stop-factor', str(a.stop_factor)]
        elif method in ('MinimumPeeling', 'UnconstrainedPeeling'):
            unconstrained = method == 'UnconstrainedPeeling'
            # Arguments: variant, starts, seed, mu, sweeps, targets (0 for unconstrained peeling).
            cmd += [('unconstrained' if unconstrained else 'constrained') + '_minimum_degree_peeling', '1',
                    str(a.seed), '10', '0', '0' if unconstrained else target_text]
        elif method in ('Greedy++', 'FISTA'):
            family = {'Greedy++': 'greedy_plus_plus', 'FISTA': 'fista'}[method]
            cmd += [family, str(a.rounds), str(a.iterations), '0', target_text]
        elif method in ('InternalUpper', 'InternalLower'):
            upper = method == 'InternalUpper'
            cmd += ['ddsp' if upper else 'dalks', str(a.seed), str(round(100 * (a.beta if upper else a.tau)))]
        with tempfile.TemporaryDirectory(prefix='dsp-input-') as temp:
            if method in ('OPT', 'Goldberg'):
                graph = Path(temp) / 'graph'
                if method == 'OPT':
                    with graph.open('w') as f:
                        f.write(f'{n} {m} {len(np.unique(labels))}\n')
                        f.write(' '.join(map(str, labels)) + '\n')
                        np.savetxt(f, edges, fmt='%d')
                    commands = [[str(binary), str(graph), str(seconds), str(t)] for t in targets]
                else:
                    with graph.open('wb') as f:
                        np.array([n, m], dtype='<u8').tofile(f)
                        edges.tofile(f)
                    commands = [[str(binary), str(graph)]]
                with (a.output / 'records.jsonl').open('w') as f:
                    for command in commands:
                        subprocess.run(command, check=True, stdout=f)
            else:
                subprocess.run(cmd, check=True)
        metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
        metadata.update(n=n, m=m)
        (a.output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'Output: {a.output.resolve()}')
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        p.exit(1, f'error: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
