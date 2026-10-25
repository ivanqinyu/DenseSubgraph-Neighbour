"""Run the approximation-guarantee experiment on the graphs from generate.py.

Runs DtC (complete search, no early stopping), QAD, CO, MinimumPeeling, Greedy++
and FISTA with the fixed demands stored in each graph's metadata, and audits
every returned core from the saved graph. The constrained optimum rho(S*) is
not computed. Instead, two certified bounds enclose each ratio:
  upper = U / rho(S),        U = Goldberg's unconstrained optimum >= rho(S*);
  lower = rho_best / rho(S), rho_best = densest verified feasible core on that
                             graph (all methods and the planted core) <= rho(S*).
Writes <output>/results.csv. Completed runs are reused on a rerun.
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = ['DtC', 'QAD', 'CO', 'P', 'G++', 'FISTA']
# Fixed-demand programs built from fixed_demands.cpp: method -> (macro, source directory).
BRIDGES = {'QAD': ('QAD', 'qad'), 'CO': ('CO', 'co'), 'P': ('PEELING', 'peeling'),
           'G++': ('GREEDY', 'iterative_peeling'), 'FISTA': ('FISTA', 'fista')}
ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')


def build_bridges(bin_dir, compiler):
    bin_dir.mkdir(parents=True, exist_ok=True)
    for macro, source in BRIDGES.values():
        target = bin_dir / source
        if target.exists():
            continue
        subprocess.run([compiler, '-O3', '-std=c++17', '-I', str(ROOT / 'src'), f'-DMETHOD_{macro}',
                        str(HERE / 'fixed_demands.cpp'), str(ROOT / 'src' / source / 'algorithm.cpp'),
                        str(ROOT / 'src/common/graph_io.cpp'), '-o', str(target)], check=True)


def measure(graph, core):
    """Size, induced edges and per-color external coverage, from the saved graph."""
    labels = np.load(graph / 'labels.npy')
    edges = np.load(graph / 'edges.npy')
    n = len(labels)
    assert len(np.unique(core)) == len(core) and (not len(core) or int(core.max()) < n)
    selected = np.zeros(n, dtype=bool)
    selected[core] = True
    u, v = edges[:, 0], edges[:, 1]
    su, sv = selected[u], selected[v]
    reached = np.zeros(n, dtype=bool)
    reached[v[su & ~sv]] = True
    reached[u[sv & ~su]] = True
    coverage = np.bincount(labels[reached], minlength=2).tolist()
    return len(core), int(np.count_nonzero(su & sv)), coverage


def verify(graph, meta, folder, record):
    core = (np.fromfile(folder / record['core_file'], dtype='<u4') if record.get('core_file')
            else np.array([], dtype='<u4'))
    size, internal, coverage = measure(graph, core)
    feasible = bool(size) and all(b >= q for b, q in zip(coverage, meta['demands']))
    assert feasible == bool(record['feasible']), (folder, coverage)
    if feasible:
        assert size == record['core_size'] and internal == record['internal_edges'], folder
        assert abs(internal / size - record['density']) < 1e-9 * max(1, internal / size), folder
    return dict(verified_feasible=feasible, verified_density=internal / size if feasible else None)


def run_method(job, args):
    graph, method = job
    meta = json.loads((graph / 'metadata.json').read_text())
    folder = args.output / 'runs' / graph.name / method
    cached = folder / 'verified.json'
    if cached.exists():
        return json.loads(cached.read_text())
    if folder.exists():  # interrupted attempt: start clean
        for f in folder.iterdir():
            f.unlink()
    folder.parent.mkdir(parents=True, exist_ok=True)
    n, m, demands = str(meta['n']), str(meta['m']), ','.join(map(str, meta['demands']))
    if method == 'DtC':
        cmd = [str(args.build_dir / 'bin/dtc'), str(graph), n, m, str(folder), 'densest_then_complete', '0',
               '--demands', demands, '--time-limit', str(args.time_limit), '--stop-factor', '0']
    else:
        cmd = [str(args.output / 'bin' / BRIDGES[method][1]), str(graph), n, m, str(folder), demands, '0']
    started = time.perf_counter()
    with (folder.parent / f'{method}.log').open('w') as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV,
                                  timeout=args.time_limit + 60)
            error = None if proc.returncode == 0 else 'process_error'
        except subprocess.TimeoutExpired:
            error = 'timeout'
    wall = time.perf_counter() - started
    if error:
        r = dict(feasible=False, density=None, status=error, verified_feasible=False, verified_density=None)
    else:
        r = json.loads((folder / 'records.jsonl').read_text().splitlines()[-1])
        r.setdefault('status', 'feasible' if r['feasible'] else 'no_feasible_core')
        r.update(verify(graph, meta, folder, r))
    r.update(dataset=graph.name, method=method, wall_seconds=wall)
    folder.mkdir(parents=True, exist_ok=True)  # a failed process may not have created it
    cached.write_text(json.dumps(r, allow_nan=False) + '\n')
    return r


def upper_bound(graph, args):
    folder = args.output / 'runs' / graph.name / 'Goldberg'
    cached = folder / 'verified.json'
    if cached.exists():
        return json.loads(cached.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    edges = np.load(graph / 'edges.npy')
    n = len(np.load(graph / 'labels.npy'))
    binary = folder / 'graph.bin'
    binary.write_bytes(np.array([n, len(edges)], dtype='<u8').tobytes() + edges.astype('<u4').tobytes())
    proc = subprocess.run([str(args.build_dir / 'bin/goldberg'), str(binary)], capture_output=True,
                          text=True, check=True, env=ENV, timeout=args.time_limit)
    binary.unlink()
    r = json.loads(proc.stdout)
    size, internal, _ = measure(graph, np.array(r['core'], dtype=np.int64))
    assert internal * r['denominator'] == r['numerator'] * size
    r = dict(dataset=graph.name, U=internal / size)
    cached.write_text(json.dumps(r) + '\n')
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=ROOT / 'inputs/dtc_guarantee')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/dtc_guarantee')
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--time-limit', type=float, default=3600, help='per-run limit in seconds')
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    for name in ('dtc', 'goldberg'):
        if not (args.build_dir / 'bin' / name).exists():
            sys.exit(f'{args.build_dir / "bin" / name} not found; build the package first (see README.md)')
    graphs = sorted(p.parent for p in args.input.glob('*/metadata.json'))
    if not graphs:
        sys.exit(f'No graphs under {args.input}; run generate.py first')
    build_bridges(args.output / 'bin', args.compiler)
    jobs = [(graph, method) for graph in graphs for method in METHODS]
    records = []
    with cf.ThreadPoolExecutor(args.workers) as pool:
        bounds = {r['dataset']: r for r in pool.map(lambda g: upper_bound(g, args), graphs)}
        for r in pool.map(lambda job: run_method(job, args), jobs):
            records.append(r)
            print(f"{r['dataset']:<16} {r['method']:<6} {r['status']:<12} density={r['density']}", flush=True)
    rows = []
    for graph in graphs:
        meta = json.loads((graph / 'metadata.json').read_text())
        size, internal, coverage = measure(graph, np.load(graph / 'planted_core.npy'))
        assert all(b >= q for b, q in zip(coverage, meta['demands']))
        planted = internal / size
        mine = [r for r in records if r['dataset'] == graph.name]
        best = max([planted] + [r['verified_density'] for r in mine if r['verified_feasible']])
        U = bounds[graph.name]['U']
        assert best <= U + 1e-12
        for r in mine:
            rho = r['verified_density']
            rows.append(dict(dataset=graph.name, c=meta['c'], q=meta['q'], seed=meta['seed'], n=meta['n'],
                             m=meta['m'], regime_ratio=meta['regime_ratio'], contrast=meta['contrast'],
                             U=U, rho_best=best, planted_density=planted, method=r['method'],
                             status=r['status'], feasible=rho is not None, density=rho,
                             ratio_lower=best / rho if rho else None, ratio_upper=U / rho if rho else None,
                             wall_seconds=r.get('wall_seconds')))
    with (args.output / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} rows -> {args.output / "results.csv"}')


if __name__ == '__main__':
    main()
