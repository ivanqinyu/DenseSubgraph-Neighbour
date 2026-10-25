"""Run the synthetic benchmark of Fig. 2 and write <output>/results.csv.

QAD, CO, P, G++ and FISTA run on all 350 graphs in data/random/ at alpha = 0.1-0.4.
DtC, OPT and Goldberg run on the 70 graphs with n = 50. DtC stops early once its core
is certified within a factor of 6; DtC and OPT get --time-limit seconds per target.
Both are deterministic, so one 3,600 s budget gives the same results as the paper's
30 s pass followed by a 3,600 s rerun of the targets that timed out.

Each core is re-measured from the graph before it is recorded. Finished runs are
reused, so the script can be stopped and restarted. One line per graph, method and
target: feasible, density, seconds (solver time), status, and for OPT whether
optimality was proven.
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, Graph, solve  # noqa: E402

TARGETS = (10, 20, 30, 40)
ALL_SIZES = {'QAD': 'QAD', 'CO': 'CO', 'P': 'MinimumPeeling', 'G++': 'Greedy++', 'FISTA': 'FISTA'}
SMALL_ONLY = {'DtC': 'DtC', 'OPT': 'OPT', 'Goldberg': 'Goldberg'}  # n = 50


def run(graph_dir, method, args):
    meta = json.loads((graph_dir / 'metadata.json').read_text())
    algorithm = {**ALL_SIZES, **SMALL_ONLY}[method]
    options = dict(seed=0, targets=','.join(map(str, TARGETS)))
    if method in ('DtC', 'OPT'):
        options['time_limit'] = args.time_limit
    output = args.output / 'runs' / graph_dir.name / method
    records = solve(algorithm, graph_dir, output, **options)
    graph = Graph(graph_dir)
    base = dict(dataset=graph_dir.name, family=meta['family'], n=meta['n'],
                degree=meta['parameters'].get('expected_degree', ''), method=method)
    rows = []
    for record in records:
        target = record.get('alpha_percent') or round(record.get('alpha', 0) * 100)
        seconds = record.get('seconds', record.get('elapsed_seconds'))
        core = graph.core(output, record) if (record.get('core_file') or 'core' in record) else []
        measured = graph.measure(core)
        if method == 'Goldberg':  # unconstrained optimum, the same for every target
            rows += [dict(base, target=t, status='optimal', feasible='', density=measured['density'],
                          seconds=seconds) for t in TARGETS]
            continue
        feasible = len(core) > 0 and graph.feasible(measured['coverage'], target)
        assert feasible == bool(record['feasible']), (graph_dir.name, method, target)
        rows.append(dict(base, target=target, status=record['status'], feasible=feasible,
                         density=measured['density'] if feasible else '', seconds=seconds,
                         optimal=record.get('completed', '') if method == 'OPT' else ''))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/random')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/synthetic')
    parser.add_argument('--methods', default='QAD,CO,P,G++,FISTA,DtC,OPT,Goldberg')
    parser.add_argument('--time-limit', type=float, default=3600, help='DtC and OPT seconds per target')
    parser.add_argument('--workers', type=int, default=1, help='parallel single-thread solver runs')
    args = parser.parse_args()
    graphs = sorted(p.parent for p in args.data.glob('*/metadata.json'))
    methods = args.methods.split(',')
    jobs = [(g, m) for m in methods for g in graphs
            if m in ALL_SIZES or json.loads((g / 'metadata.json').read_text())['n'] == 50]
    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        for done, result in enumerate(pool.map(lambda job: run(*job, args), jobs), 1):
            rows += result
            if done % 100 == 0 or done == len(jobs):
                print(f'{done}/{len(jobs)} runs', flush=True)
    fields = ['dataset', 'family', 'n', 'degree', 'method', 'target', 'status', 'feasible', 'density', 'seconds',
              'optimal']
    with (args.output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), 'rows ->', args.output / 'results.csv')


if __name__ == '__main__':
    main()
