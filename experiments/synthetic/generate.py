"""Generate the synthetic benchmark of Fig. 2 (350 graphs) into data/random/.

n = 50, 100, 200, 400 and 800 with seeds 0-9. ER and SBM keep the expected average
degree d = 4, 8 or 16 at every size (fixed probabilities would leave the small graphs
nearly empty):

- ER: p = d / (n - 1); four balanced groups assigned uniformly at random.
- SBM: four nearly equal blocks as groups, p_in / p_out = 0.1 / 0.015, with p_out set
  by the target degree.
- LFR (networkx): tau1 = 2.5, tau2 = 1.5, mixing parameter 0.25, the size-dependent
  parameters in LFR_PARAMETERS; groups are the planted communities, numbered by
  their smallest vertex. A failed attempt is retried with the next seed.

The graphs are byte-identical to those used in the paper (checked with networkx 3.2.1).
"""
import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIZES = (50, 100, 200, 400, 800)
SEEDS = range(10)
DEGREES = (4, 8, 16)
RATIO = .1 / .015  # p_in / p_out
# n: (average_degree, min_community, max_community, max_degree), as passed to networkx.
LFR_PARAMETERS = {50: (4, 5, 25, 25), 100: (6, 10, 50, 20), 200: (8, 20, 100, 40), 400: (16, 40, 200, 80),
                  800: (16, 80, 400, 160)}
# Graph seeds of the paper's LFR graphs, for seeds 0-9 at each n.
LFR_SEEDS = {
    50: (121423, 86116, 113112, 128344, 130160, 73996, 70320, 101512, 69073, 107041),
    100: (124834, 96713, 135589, 96391, 114880, 99952, 88606, 98482, 94479, 94328),
    200: (155973, 149703, 143502, 103028, 142065, 85358, 75811, 149981, 61669, 84591),
    400: (68223, 75293, 101752, 77170, 147483, 70507, 108057, 101743, 159675, 88764),
    800: (128733, 65703, 85364, 93743, 125858, 78247, 147910, 71307, 110905, 86489),
}
LABEL_POLICY = 'Balanced random labels for ER; SBM labels are the planted blocks; LFR planted communities'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_edges(rng, n, probability):
    """Each pair (u < v) independently with probability(u, v)."""
    u, v = np.triu_indices(n, 1)
    keep = rng.random(len(u)) < probability(u, v)
    return np.column_stack((u[keep], v[keep])).astype('<u4')


def balanced(n, groups):
    return np.repeat(np.arange(groups), [n // groups + (i < n % groups) for i in range(groups)])


def generate(family, n, seed, degree):
    key = [20261025, ('er', 'sbm').index(family), n, seed, 1, degree]
    graph_seed = int(np.random.SeedSequence(key).generate_state(1)[0])
    rng = np.random.default_rng(graph_seed)
    if family == 'er':
        p = degree / (n - 1)
        edges = sample_edges(rng, n, lambda u, v: p)
        labels = rng.permutation(balanced(n, 4))
        parameters = dict(expected_degree=degree, p=p)
    else:
        labels = rng.permutation(balanced(n, 4))
        size = np.bincount(labels)
        # A vertex in block b expects (size_b - 1) p_in + (n - size_b) p_out neighbors;
        # averaging over vertices gives p_out from the target degree.
        weight = float(np.sum(size * ((size - 1) * RATIO + (n - size))) / n)
        p_out = degree / weight
        p_in = RATIO * p_out
        edges = sample_edges(rng, n, lambda u, v: np.where(labels[u] == labels[v], p_in, p_out))
        parameters = dict(expected_degree=degree, p_in=p_in, p_out=p_out)
    return edges, labels.astype('<u2'), graph_seed, parameters


def generate_lfr(n, seed):
    graph_seed = LFR_SEEDS[n][seed]
    degree, min_community, max_community, max_degree = LFR_PARAMETERS[n]
    for attempt in range(50):
        try:
            g = nx.LFR_benchmark_graph(n, 2.5, 1.5, .25, average_degree=degree, min_community=min_community,
                                       max_community=max_community, max_degree=max_degree,
                                       seed=graph_seed * 1000 + attempt, max_iters=1000 if n == 50 else 500)
            break
        except nx.ExceededMaxIterations:
            continue
    g.remove_edges_from(nx.selfloop_edges(g))
    edges = np.array(sorted((min(u, v), max(u, v)) for u, v in g.edges()), dtype='<u4').reshape(-1, 2)
    labels = np.zeros(n, dtype='<u2')
    for group, community in enumerate(sorted({frozenset(g.nodes[v]['community']) for v in g}, key=min)):
        labels[list(community)] = group
    parameters = dict(tau1=2.5, tau2=1.5, mu=.25, average_degree=degree, min_community=min_community,
                      max_community=max_community, max_degree=max_degree, networkx_seed=graph_seed * 1000 + attempt)
    return edges, labels, graph_seed, parameters


def write(folder, family, edges, labels, graph_seed, parameters):
    folder.mkdir()
    np.save(folder / 'edges.npy', edges, allow_pickle=False)
    np.save(folder / 'labels.npy', labels, allow_pickle=False)
    meta = dict(dataset=folder.name, format='nfds-compact-v1', family=family, graph_seed=graph_seed, n=len(labels),
                m=len(edges), groups={str(c): int(k) for c, k in enumerate(np.bincount(labels))},
                parameters=parameters, label_policy=LABEL_POLICY,
                sha256={f: digest(folder / f) for f in ('edges.npy', 'labels.npy')})
    (folder / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/random')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for n in SIZES:
        for seed in SEEDS:
            for family in ('er', 'sbm'):
                for degree in DEGREES:
                    folder = args.output / f'random-{family}-n{n}-d{degree}-r{seed}'
                    if not folder.exists():
                        write(folder, family, *generate(family, n, seed, degree))
            folder = args.output / f'random-lfr-n{n}-r{seed}'
            if not folder.exists():
                write(folder, 'lfr', *generate_lfr(n, seed))
    print(len(list(args.output.iterdir())), 'graphs in', args.output)


if __name__ == '__main__':
    main()
