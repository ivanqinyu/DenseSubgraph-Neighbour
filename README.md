# Densest subgraphs with diverse external neighborhoods

Code for the paper. Given a graph whose vertices belong to groups, find a core `S` of
maximum density `|E(S)| / |S|` such that, for every group `c`, at least `d_c`
distinct vertices of `c` outside `S` are adjacent to `S`.

## Build

Linux (or WSL), Python 3.9+, a C++17 compiler and CMake 3.16+.

```bash
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
```

## Run a solver

```bash
python run.py --algorithm QAD --input data/high-school-2013-class --output outputs/hs13c-qad --targets 30
python run.py --algorithm CO  --input data/high-school-2013-class --output outputs/hs13c-co  --targets 30
```

`--targets 30` asks for `ceil(0.3 |V_c|)` outside neighbors in every group; several
targets can be given (`--targets 10,20,30,40`). The output directory gets `records.jsonl`
(one record per target: density, core size, feasibility, coverage per group) and the
core's vertex IDs.

| `--algorithm` | Paper | Options (default) |
|---|---|---|
| `QAD` | QAD | `--seed` |
| `CO` | CO | `--seed`, `--mu` (256), `--learning-rate` (0.02), `--steps` (1000) |
| `DtC` | DtC | `--stop-factor` (6), `--time-limit` (30 s per target) |
| `MinimumPeeling`, `Greedy++`, `FISTA` | P, G++, FISTA | `--rounds` (5), `--iterations` (50) |
| `InternalUpper`, `InternalLower`, `Demographics` | internal-diversity baselines | `--beta` (0.6), `--tau` (0.2) |
| `OPT`, `Goldberg` | exact references | `--time-limit` for OPT (at most 1,024 vertices) |
| `UnconstrainedPeeling` | peeling in Fig. 4 | |

By default DtC stops once its core's density is at least 1/6 of an upper bound on the
optimum, or after the time limit; `--stop-factor 0` runs the complete search of the
paper. Its running time grows quickly with the total demand. The internal-diversity methods solve
a different problem; their `feasible` field refers to their own constraints. Goldberg
ignores the groups and gives an upper bound.

## Reproduce the paper

Each experiment writes to `outputs/<experiment>/`.

| Paper | Commands | Notes |
|---|---|---|
| Fig. 2 | `python experiments/synthetic/generate.py`<br>`python experiments/synthetic/run.py --workers 8`<br>`python experiments/synthetic/plot_feasibility.py`<br>`python experiments/synthetic/plot_density_time.py` | Generates the 350 benchmark graphs. DtC and OPT get 3,600 s per target; about an hour with 8 workers. |
| Fig. 3 | `python experiments/dtc_guarantee/generate.py`<br>`python experiments/dtc_guarantee/run.py --workers 4`<br>`python experiments/dtc_guarantee/plot.py` | Approximation guarantee of DtC; see `experiments/dtc_guarantee/README.md`. |
| Fig. 4 | `python experiments/lastfm/run.py`<br>`python experiments/lastfm/plot.py` | LastFM case study; under a minute. |
| Tables 2, 3 | `python experiments/real/run.py`<br>`python experiments/real/tables.py` | Uses every real dataset with edges in `data/` or `inputs/` (see `data/README.md`); serial, about a day for all datasets, mostly CO on the largest graphs. |

## Data

`data/` holds the real datasets; see `data/README.md` for sources, the format and the
datasets whose edges are not shipped. The synthetic graphs are generated (Fig. 2).

## License

The code is released under the MIT License (`LICENSE`). Datasets keep the terms of
their providers (see `data/README.md`). `src/fista/algorithm.cpp` retains the BSD notice
of the NetworkX formula it uses.
