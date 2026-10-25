"""Run the real-graph comparison (Tables 2 and 3) and write <output>/results.csv.

On every real dataset with edges (in data/ or, after prepare_data.py, in inputs/):
QAD and CO with seeds 0-9; P, G++, FISTA, Goldberg and the internal-diversity methods
(InternalUpper beta = 0.6, InternalLower tau = 0.2, Demographics) once. Coverage
targets are alpha = 0.2 (Table 3) and 0.3 (Table 2). Runs are serial and single
thread; a run that exceeds --timeout seconds is recorded as a timeout (OOT).

Each core is re-measured from the graph and checked against the external demands, so
the internal methods and Goldberg also get an external feasibility. One line per
dataset, method, seed and target. Finished runs are reused on a restart.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, Graph, solve  # noqa: E402

TARGETS = (20, 30)
SEEDED = {'QAD': 'QAD', 'CO': 'CO'}
ONCE = {'P': 'MinimumPeeling', 'G++': 'Greedy++', 'FISTA': 'FISTA', 'Goldberg': 'Goldberg',
        'Upper': 'InternalUpper', 'Lower': 'InternalLower', 'Demographic': 'Demographics'}
INTERNAL = ('Upper', 'Lower', 'Demographic')  # solve internal constraints; no targets


def datasets(folders, chosen):
    found = {}
    for folder in folders:
        for meta in sorted(folder.glob('*/metadata.json')):
            graph = meta.parent
            if (graph / 'edges.npy').exists():
                found.setdefault(graph.name, graph)  # data/ first, then inputs/
    return [found[name] for name in (chosen or sorted(found))]


def run(graph_dir, graph, method, seed, args):
    algorithm = {**SEEDED, **ONCE}[method]
    options = {} if method in INTERNAL or method == 'Goldberg' else dict(targets=','.join(map(str, TARGETS)))
    if method in SEEDED:
        options['seed'] = seed
    output = args.output / 'runs' / graph_dir.name / method / f'seed_{seed}'
    records = solve(algorithm, graph_dir, output, timeout=args.timeout, **options)
    base = dict(dataset=graph_dir.name, method=method, seed=seed)
    if records is None:
        return [dict(base, target=t, status='timeout', feasible=False, density='', seconds='') for t in TARGETS]
    rows = []
    for record in records:
        core = graph.core(output, record, mask=method in INTERNAL)
        measured = graph.measure(core)
        seconds = record.get('seconds', record.get('elapsed_seconds'))
        targets = TARGETS if method in INTERNAL or method == 'Goldberg' else [round(record['alpha'] * 100)]
        for target in targets:
            feasible = len(core) > 0 and graph.feasible(measured['coverage'], target)
            rows.append(dict(base, target=target, status=record.get('status', 'completed'), feasible=feasible,
                             density=measured['density'] if len(core) else '', seconds=seconds))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/real')
    parser.add_argument('--datasets', default='', help='comma-separated names (default: all available)')
    parser.add_argument('--methods', default='QAD,CO,P,G++,FISTA,Goldberg,Upper,Lower,Demographic')
    parser.add_argument('--seeds', type=int, default=10, help='seeds for QAD and CO')
    parser.add_argument('--timeout', type=float, default=3600, help='seconds per run')
    args = parser.parse_args()
    rows = []
    for graph_dir in datasets([ROOT / 'data', ROOT / 'inputs'], [d for d in args.datasets.split(',') if d]):
        graph = Graph(graph_dir)  # loaded once per dataset
        for method in args.methods.split(','):
            for seed in range(args.seeds if method in SEEDED else 1):
                rows += run(graph_dir, graph, method, seed, args)
                print(graph_dir.name, method, f'seed {seed}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset', 'method', 'seed', 'target', 'status', 'feasible',
                                                    'density', 'seconds'])
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), 'rows ->', args.output / 'results.csv')


if __name__ == '__main__':
    main()
