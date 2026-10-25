# Approximation-guarantee experiment

Reproduces the paper's figure on the approximation guarantee of DtC: the ratio
between the optimum and each method's density on adversarial graphs that sit
exactly at the guarantee's condition, rho(G) = 3 d_Sigma.

## Instances

`generate.py` builds gated-witness graphs with demands (red, blue) = (1, 4), so
d_Sigma = 5 and rho(G) = 15:

- a dense red region, G(k, 0.9), whose density is c times rho(G);
- four blue vertices, each adjacent only to its own red gate vertex of degree 3;
- a sparse red periphery (minimum degree 4) that sets rho(G), plus one red leaf.

Every feasible core keeps all gates, and the dense region together with the
gates is feasible with density close to that of the dense region. Peeling-based
methods delete the gates early, after which no blue vertex can be covered, so
their feasible candidates keep most of the periphery. The density contrast c
takes the values 2, 4, 8, 16, 32 and 64, with five graphs (seeds 0-4) each. The
generator is deterministic; each graph's `metadata.json` records its checksums.

## Methods and ratio

- DtC runs the complete search of the paper (`--stop-factor 0`, 3600-second limit).
- QAD, CO, MinimumPeeling (P), Greedy++ (G++, five rounds) and FISTA (50
  iterations) run through `fixed_demands.cpp`, which passes absolute demands to
  the unchanged implementations in `src/`; its header explains how.
- Goldberg's unconstrained optimum U is at least the constrained optimum, so
  U / rho(S) bounds each method's approximation ratio from above. `run.py` also
  reports a lower bound from the densest verified feasible core on each graph.

`run.py` audits every returned core from the saved graph before using it.

## Run

From the package root, after building with CMake as described in the top-level
README:

```bash
python experiments/dtc_guarantee/generate.py           # 30 graphs, about 150 MB, about 1 minute
python experiments/dtc_guarantee/run.py --workers 4    # a few minutes
python -m pip install matplotlib
python experiments/dtc_guarantee/plot.py
```

Graphs go to `inputs/dtc_guarantee/`; results, logs and cores to
`outputs/dtc_guarantee/`, with the per-run table in `results.csv` and the figure
in `dtc_guarantee.pdf`. Each marker is the largest U / rho(S) over the five
graphs at that contrast.

## Expected results

All 180 runs return feasible cores, and DtC completes its search on every graph.

| Contrast | DtC, QAD, CO | P, G++, FISTA |
|---:|---:|---:|
| 2 | 1.06 | 1.95 |
| 4 | 1.03 | 4.01 |
| 8 | 1.01 | 8.02 |
| 16 | 1.01 | 16.0 |
| 32 | 1.00 | 32.0 |
| 64 | 1.00 | 64.0 |

DtC, QAD and CO return the same core, the dense region plus the gates, on every
graph; P, G++ and FISTA also agree with one another.
