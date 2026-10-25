"""Gated-witness graphs at the boundary of the DtC approximation guarantee.

One instance GW(c, q, seed) has demands (red, blue) = (1, q), total demand
d = q + 1, and global density rho(G) = m/n just above 3d, so the theorem's
condition rho(G) >= (kappa+1) d with kappa = 2 holds with minimal slack.

  K          k red vertices, G(k, 0.9), sized so that rho(K) ~ c * 3d.
  gates      q red vertices; gate i is adjacent to witness i and to two
             periphery vertices (degree 3).
  witnesses  q blue vertices, witness i adjacent only to gate i. These are all
             blue vertices, so every feasible core keeps all gates.
  red leaf   one red vertex adjacent to one periphery vertex.
  periphery  red 4-out random graph (minimum degree >= 4); each K vertex also
             has one periphery neighbor. Its size sets rho(G) ~ 3d.

The planted core K + gates is feasible with density ~ rho(K). Minimum-degree
order removes witnesses and leaf (degree 1), then gates (degree <= 2), before
any periphery vertex (degree >= 4); after a gate leaves, its witness can never
be covered again. Hence every feasible state of that order lies before the
first gate removal and has density ~ rho(G) = rho(K)/c. Vertex IDs are
randomly permuted so ID-based tie-breaking carries no structure.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

VERSION = 'gated-witness-v1'
P_CORE = .9
OUT_DEGREE = 4


def generate(c, q, seed):
    rng = np.random.default_rng(np.random.SeedSequence([20261024, int(round(100 * c)), q, seed]))
    d = q + 1
    target = 3 * d
    k = math.ceil(2 * c * target / P_CORE) + 1
    u, v = np.triu_indices(k, 1)
    keep = rng.random(len(u)) < P_CORE
    core_edges = np.column_stack((u[keep], v[keep]))
    e_core = len(core_edges)
    # Fixed parts: gates/witnesses (3q edges), leaf (1), K-periphery (k).
    fixed_vertices = k + 2 * q + 1
    fixed_edges = e_core + 3 * q + 1 + k
    periphery = math.ceil((fixed_edges - target * fixed_vertices) / (target - OUT_DEGREE))
    assert periphery > 2 * OUT_DEGREE, 'c too small for a periphery'
    n = fixed_vertices + periphery
    P = k + np.arange(periphery)
    gates = k + periphery + np.arange(q)
    witnesses = gates[-1] + 1 + np.arange(q)
    leaf = n - 1
    pieces = [core_edges]
    picks = rng.integers(0, periphery - 1, size=(periphery, OUT_DEGREE))
    picks += picks >= np.arange(periphery)[:, None]  # skip self
    pieces.append(np.column_stack((np.repeat(P, OUT_DEGREE), k + picks.ravel())))
    pieces.append(np.column_stack((np.arange(k), rng.choice(P, size=k))))
    for g, w in zip(gates, witnesses):
        pieces.append(np.array([[g, w]] + [[g, p] for p in rng.choice(P, size=2, replace=False)]))
    pieces.append(np.array([[rng.choice(P), leaf]]))
    edges = np.unique(np.sort(np.concatenate(pieces), axis=1), axis=0)
    # Duplicate random picks can leave a periphery vertex below degree 4 or
    # rho(G) below 3d; repair with extra periphery-periphery edges only.
    present = set(map(tuple, edges.tolist()))
    def add_periphery_edge(a=None):
        while True:
            x = int(rng.choice(P)) if a is None else int(a)
            y = int(rng.choice(P))
            e = (min(x, y), max(x, y))
            if x != y and e not in present:
                present.add(e)
                return
    degree = np.bincount(edges.ravel(), minlength=n)
    for p in P[degree[P] < OUT_DEGREE]:
        for _ in range(OUT_DEGREE - degree[p]):
            add_periphery_edge(p)
    while len(present) < target * n:
        add_periphery_edge()
    edges = np.array(sorted(present), dtype=np.int64)
    labels = np.zeros(n, dtype=np.uint16)
    labels[witnesses] = 1
    planted = np.concatenate((np.arange(k), gates))
    # Construction checks, on the unpermuted graph.
    degree = np.bincount(edges.ravel(), minlength=n)
    assert (degree[witnesses] == 1).all() and degree[leaf] == 1 and (degree[gates] == 3).all()
    assert degree[P].min() >= OUT_DEGREE and degree[:k].min() > 2 * OUT_DEGREE
    m = len(edges)
    assert m >= target * n
    permutation = rng.permutation(n)
    edges = np.sort(permutation[edges], axis=1)
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))].astype(np.uint32)
    relabeled = np.empty(n, dtype=np.uint16)
    relabeled[permutation] = labels
    planted = np.sort(permutation[planted]).astype(np.uint32)
    meta = dict(family='gated_witness', generator_version=VERSION, c=c, q=q, seed=seed,
                demands=[1, q], total_demand=d, n=n, m=m, rho_G=m / n,
                regime_ratio=m / n / d, core_size=k, core_edge_probability=P_CORE,
                core_edges=e_core, rho_core=e_core / k, contrast=e_core / k / (m / n),
                periphery=periphery, periphery_out_degree=OUT_DEGREE,
                planted_core_size=len(planted), planted_density=e_core / len(planted))
    return edges, relabeled, planted, meta


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--contrasts', type=float, nargs='+', default=[2, 4, 8, 16, 32, 64])
    parser.add_argument('--q', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--output', type=Path, default=Path('inputs/dtc_guarantee'))
    args = parser.parse_args()
    for c in args.contrasts:
        for seed in args.seeds:
            folder = args.output / f'gw-c{c:g}-q{args.q}-s{seed}'
            if folder.exists():
                print('exists', folder)
                continue
            edges, labels, planted, meta = generate(c, args.q, seed)
            folder.mkdir(parents=True)
            np.save(folder / 'edges.npy', edges, allow_pickle=False)
            np.save(folder / 'labels.npy', labels, allow_pickle=False)
            np.save(folder / 'planted_core.npy', planted, allow_pickle=False)
            meta.update(dataset=folder.name, format='nfds-compact-v1',
                        groups={str(g): int((labels == g).sum()) for g in (0, 1)},
                        sha256={f: digest(folder / f) for f in ('edges.npy', 'labels.npy', 'planted_core.npy')},
                        generator_sha256=digest(Path(__file__)))
            (folder / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
            print(f"{folder.name}: n={meta['n']} m={meta['m']} rho(G)/d={meta['regime_ratio']:.3f} "
                  f"contrast={meta['contrast']:.2f}", flush=True)


if __name__ == '__main__':
    main()
