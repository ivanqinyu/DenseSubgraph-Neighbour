"""Run the LastFM case study (Fig. 4) and write <output>/coverage.csv.

Compares unconstrained minimum-degree peeling with QAD on lastfm-asia, whose groups
are 18 countries, under two demands: 10 users per country (fixed) and 30% of each
country (proportional). QAD runs with seeds 0-9 under each demand; fixed demands go
through experiments/dtc_guarantee/fixed_demands.cpp, built from the same QAD source.
Every core is re-measured from the graph. coverage.csv has, per country, the demand
and the number of distinct outside neighbors reached by peeling and by QAD (mean,
min and max over seeds); summary.json has the densities.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, Graph, demands, read_records, solve  # noqa: E402

GRAPH = ROOT / 'data/lastfm-asia'
SEEDS = range(10)


def build_fixed_demand_qad(output, compiler):
    binary = output / 'qad-fixed-demands'
    if not binary.exists():
        output.mkdir(parents=True, exist_ok=True)
        src = ROOT / 'src'
        subprocess.run([compiler, '-O3', '-std=c++17', '-I', str(src), '-DMETHOD_QAD',
                        str(ROOT / 'experiments/dtc_guarantee/fixed_demands.cpp'), str(src / 'qad/algorithm.cpp'),
                        str(src / 'common/graph_io.cpp'), '-o', str(binary)], check=True)
    return binary


def run_fixed(binary, graph, folder, quota, seed):
    if not (folder / 'records.jsonl').exists():
        folder.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS='1')
        subprocess.run([str(binary), str(GRAPH), str(len(graph.labels)), str(len(graph.edges)), str(folder),
                        ','.join(map(str, quota)), str(seed)], check=True, capture_output=True, env=env)
    return read_records(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/lastfm')
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    graph = Graph(GRAPH)
    peeling_dir = args.output / 'peeling'
    peeling = graph.measure(graph.core(peeling_dir, solve('UnconstrainedPeeling', GRAPH, peeling_dir)[0]))
    quotas = {'fixed': np.full(len(graph.counts), 10), 'proportional': demands(graph.counts, 30)}
    binary = build_fixed_demand_qad(args.output / 'bin', args.compiler)
    coverage, density = {}, {}
    for mode, quota in quotas.items():
        runs = []
        for seed in SEEDS:
            folder = args.output / mode / f'seed_{seed}'
            if mode == 'fixed':
                record = run_fixed(binary, graph, folder, quota, seed)[0]
            else:
                record = solve('QAD', GRAPH, folder, targets=30, seed=seed)[0]
            measured = graph.measure(graph.core(folder, record))
            assert np.all(measured['coverage'] >= quota), (mode, seed)  # every demand met
            runs.append(measured)
        coverage[mode] = np.array([r['coverage'] for r in runs])
        density[mode] = float(np.mean([r['density'] for r in runs]))
    labels = json.loads((GRAPH / 'metadata.json').read_text())['original_label_map']
    rows = []
    for group in range(len(graph.counts)):
        row = dict(group=group, country=labels[str(group)], population=int(graph.counts[group]),
                   peeling=int(peeling['coverage'][group]))
        for mode in quotas:
            row.update({f'{mode}_demand': int(quotas[mode][group]),
                        f'{mode}_qad_mean': float(coverage[mode][:, group].mean()),
                        f'{mode}_qad_min': int(coverage[mode][:, group].min()),
                        f'{mode}_qad_max': int(coverage[mode][:, group].max())})
        rows.append(row)
    with (args.output / 'coverage.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(peeling_density=peeling['density'], qad_density_fixed=density['fixed'],
                   qad_density_proportional=density['proportional'], seeds=len(SEEDS))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
