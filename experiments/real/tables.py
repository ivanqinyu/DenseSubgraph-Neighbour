"""Format Tables 2 and 3 from run.py's results.csv (Markdown, written next to it).

Table 2 (alpha = 0.3): density and runtime. QAD and CO: mean and sample standard
deviation over their runs; P, G++ and FISTA: one run. N/A means no externally feasible
core, OOT a timeout; bold marks the highest density among the feasible heuristics.
Goldberg is the unconstrained optimum, with * when its core is externally feasible
(then it is also the constrained optimum).

Table 3 (alpha = 0.2): density and external feasibility of the internal-diversity
methods, QAD and CO. A check mark means every external demand is met (for QAD and CO,
in every run; otherwise the number of feasible runs is shown).
"""
import argparse
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = {'high-school-2013-gender': 'HS13-G', 'high-school-2013-class': 'HS13-C', 'email-Eu-core': 'Email-Eu',
         'lastfm-asia': 'LastFM', 'deezer-europe': 'Deezer', 'github-web-ml': 'GitHub', 'pokec-location': 'Pokec',
         'proteins-species': 'Proteins', 'twitch-language': 'Twitch', 'arxiv-subject': 'Arxiv',
         'reddit-community': 'Reddit', 'genius-status': 'Genius', 'mag-venue': 'MAG', 'pokec-gender': 'Pokec-G',
         'bluesky-language': 'Bluesky', 'ogbn-products': 'Amazon'}
HEURISTICS = ('QAD', 'CO', 'P', 'G++', 'FISTA')


def number(x, digits=2):
    return '< 0.01' if digits == 2 and x < .005 else f'{x:.{digits}f}'


def cell(runs, field='density', digits=2):
    """Mean (and standard deviation for several runs) over the feasible runs."""
    if not runs:
        return ''
    if any(r['status'] == 'timeout' for r in runs):
        return 'OOT'
    values = [float(r[field]) for r in runs if field == 'seconds' or r['feasible'] == 'True']
    if not values:
        return 'N/A'
    text = number(statistics.mean(values), digits)
    if len(runs) > 1:
        text += f' (± {statistics.stdev(values) if len(values) > 1 else 0:.3f})'
    return text


def table2(rows, datasets):
    lines = ['| Dataset | Goldberg | QAD | CO | P | G++ | FISTA | QAD (s) | CO (s) | P (s) | G++ (s) | FISTA (s) |',
             '|---' * 12 + '|']
    for d in datasets:
        runs = {m: [r for r in rows if r['dataset'] == d and r['method'] == m and r['target'] == '30']
                for m in HEURISTICS + ('Goldberg',)}
        gold = runs['Goldberg'][0] if runs['Goldberg'] else None
        means = {m: statistics.mean(float(r['density']) for r in runs[m] if r['feasible'] == 'True')
                 for m in HEURISTICS if any(r['feasible'] == 'True' for r in runs[m])}
        best = max(means.values(), default=None)
        density = [f'**{cell(runs[m])}**' if m in means and means[m] == best else cell(runs[m]) for m in HEURISTICS]
        seconds = [cell(runs[m], 'seconds') if cell(runs[m]) not in ('OOT', 'N/A') else cell(runs[m])
                   for m in HEURISTICS]
        goldberg = f"{float(gold['density']):.2f}{'*' if gold['feasible'] == 'True' else ''}" if gold else ''
        lines.append('| ' + ' | '.join([NAMES.get(d, d), goldberg] + density + seconds) + ' |')
    return lines


def table3(rows, datasets):
    methods = ('Upper', 'Lower', 'Demographic', 'QAD', 'CO')
    lines = ['| Dataset | ' + ' | '.join(methods) + ' |', '|---' * 6 + '|']
    feasible_graphs = dict.fromkeys(methods, 0)
    for d in datasets:
        cells = []
        for m in methods:
            runs = [r for r in rows if r['dataset'] == d and r['method'] == m and r['target'] == '20']
            if not runs or any(r['status'] == 'timeout' for r in runs):
                cells.append('OOT' if runs else '')
                continue
            ok = sum(r['feasible'] == 'True' for r in runs)
            feasible_graphs[m] += ok == len(runs)
            mark = '✓' if ok == len(runs) else '✗' if ok == 0 or len(runs) == 1 else f'✓ in {ok}/{len(runs)}'
            density = statistics.mean(float(r['density']) for r in runs if r['density'])
            cells.append(f'{density:.3f} {mark}')
        lines.append('| ' + ' | '.join([NAMES.get(d, d)] + cells) + ' |')
    totals = [f'{feasible_graphs[m]}/{len(datasets)}' for m in methods]
    lines.append('| Externally feasible | ' + ' | '.join(totals) + ' |')
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=ROOT / 'outputs/real/results.csv')
    args = parser.parse_args()
    with args.results.open() as stream:
        rows = list(csv.DictReader(stream))
    present = {r['dataset'] for r in rows}
    datasets = [d for d in NAMES if d in present] + sorted(present - set(NAMES))
    for name, lines in (('table2.md', table2(rows, datasets)), ('table3.md', table3(rows, datasets))):
        (args.results.parent / name).write_text('\n'.join(lines) + '\n')
        print('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
