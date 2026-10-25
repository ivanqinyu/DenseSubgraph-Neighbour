# Data

Each dataset directory has a `metadata.json` and, where the files are small enough
to ship,

- `edges.npy`: uint32 `(m, 2)`, sorted and unique, with `u < v` (no self-loops);
- `labels.npy`: uint16 `(n,)`, compact group IDs `0, ..., C-1`; vertex IDs are row positions.

For the real datasets, `metadata.json` records `n`, `m`, `groups` (group sizes),
`original_label_map` (each group's label in the source), `source`, `notes` and the
`sha256` of both arrays. Graphs are unweighted and undirected.

## Real datasets

| Paper | Directory | Source | Shipped |
|---|---|---|---|
| HS13-G, HS13-C | `high-school-2013-gender`, `high-school-2013-class` | [SocioPatterns](https://sociopatterns.org/datasets/high-school-contact-and-friendship-networks/) | all |
| Email-Eu | `email-Eu-core` | [SNAP](https://snap.stanford.edu/data/email-Eu-core.html) | all |
| LastFM | `lastfm-asia` | [SNAP](https://snap.stanford.edu/data/feather-lastfm-social.html) | all |
| Deezer | `deezer-europe` | [SNAP](https://snap.stanford.edu/data/feather-deezer-social.html) | all |
| GitHub | `github-web-ml` | [SNAP](https://snap.stanford.edu/data/github-social.html) | all |
| Pokec | `pokec-location` | FairGNN instance of [SNAP Pokec](https://snap.stanford.edu/data/soc-Pokec.html) | all |
| Arxiv | `arxiv-subject` | [OGB](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) | all |
| Genius | `genius-status` | [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale) | all |
| Twitch | `twitch-language` | [SNAP](https://snap.stanford.edu/data/twitch_gamers.html) | labels |
| Reddit | `reddit-community` | [GraphSAGE](https://snap.stanford.edu/graphsage/) | labels |
| MAG | `mag-venue` | [OGB](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) | labels |
| Proteins | `proteins-species` | [OGB](https://ogb.stanford.edu/docs/nodeprop/#ogbn-proteins) | labels |
| Pokec-G | `pokec-gender` | [SNAP](https://snap.stanford.edu/data/soc-Pokec.html) | labels |
| Amazon | `ogbn-products` | [OGB](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) | labels |
| Bluesky | `bluesky-language` | [Zenodo](https://zenodo.org/records/14669616) | metadata |

Datasets shipped without edges (too large for the repository) can be downloaded from
the linked sources. The `notes` in each `metadata.json` describe the preprocessing:
for example, Twitch drops users whose language is OTHER and Reddit maps vertex IDs
and labels through GraphSAGE's `id_map`. `prepare_data.py` converts text or NPY edge
and label files into this format (`python prepare_data.py --help`), and a converted
graph can be checked against the checksums in `metadata.json`. For Amazon, Proteins
and Pokec-G:

```bash
python prepare_data.py --edges products/raw/edge.csv.gz --labels products/raw/node-label.csv.gz --delimiter , --output inputs/ogbn-products
python prepare_data.py --edges proteins/raw/edge.csv.gz --labels proteins/raw/species.csv.gz --delimiter , --output inputs/proteins-species
python prepare_data.py --edges soc-pokec-relationships.txt.gz --pokec-profiles soc-pokec-profiles.txt.gz --output inputs/pokec-gender
```

Bluesky's per-user data are not redistributed; rebuild the graph from Zenodo (about
20 GB of downloads; needs zlib headers):

```bash
python -m pip install -r requirements-bluesky.txt
python prepare_bluesky.py --download --work-dir downloads/bluesky --output inputs/bluesky-language
```

## Synthetic graphs

The benchmark of Fig. 2 (ER, SBM and LFR, n = 50 to 800, ten seeds each: 350 graphs)
is not shipped; `python experiments/synthetic/generate.py` writes it to `random/`, byte
for byte as used in the paper. Its `metadata.json` has `family`, `graph_seed`,
`parameters` and `label_policy` instead of `original_label_map`, `source` and `notes`.

## Licensing

Datasets keep the terms of their providers. In particular, the high-school datasets
derive from SocioPatterns and are distributed under its Creative Commons
Attribution-NonCommercial-ShareAlike terms; cite Mastrandrea et al., PLoS ONE 10(9),
e0136497 (2015).
