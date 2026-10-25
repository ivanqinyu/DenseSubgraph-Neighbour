"""Plot Fig. 2a: external feasibility versus coverage target (needs matplotlib).

Each point is the share of graphs whose core meets every demand. Every seed (r0-r9)
is a full replicate with the same number of graphs, so the point is also the mean of
the per-seed feasibilities; the error bar is their standard deviation (clipped to
0-100%). A method feasible on every graph has no error bar.
"""
import argparse
import csv
import os
import re
import statistics
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dsp-neighbour-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
TARGETS = (10, 20, 30, 40)
# G++ is sky blue rather than green: green and the gray of P look alike under
# deuteranopia. Marker shapes and line styles also tell the series apart.
STYLES = {
    'QAD': dict(color='#0072B2', marker='o', linestyle=(0, (2, 2)), markersize=8, zorder=6),
    'CO': dict(color='#D55E00', marker='^', linestyle=(0, (1, 2)), markersize=8, zorder=5),
    'P': dict(color='#808080', marker='v', linestyle='-', markersize=7, zorder=3),
    'G++': dict(color='#56B4E9', marker='P', linestyle='-', markersize=7, zorder=4),
    'FISTA': dict(color='#332288', marker='*', linestyle='-', markersize=8, markerfacecolor='#332288', zorder=3),
}


def points(rows, method):
    out = []
    for target in TARGETS:
        runs = [r for r in rows if r['method'] == method and int(r['target']) == target]
        by_seed = {}
        for r in runs:
            by_seed.setdefault(re.search(r'-r(\d+)$', r['dataset']).group(1), []).append(r['feasible'] == 'True')
        share = 100 * sum(r['feasible'] == 'True' for r in runs) / len(runs)
        spread = statistics.stdev(100 * sum(v) / len(v) for v in by_seed.values())
        out.append(dict(method=method, target=target, feasible=sum(r['feasible'] == 'True' for r in runs),
                        total=len(runs), percent=share, low=max(share - spread, 0), high=min(share + spread, 100)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=ROOT / 'outputs/synthetic/results.csv')
    parser.add_argument('--methods', default='QAD,CO,P,G++,FISTA')
    args = parser.parse_args()
    with args.results.open() as stream:
        rows = list(csv.DictReader(stream))
    methods = args.methods.split(',')
    data = {m: points(rows, m) for m in methods}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 12, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10, 'legend.fontsize': 10, 'axes.linewidth': .9, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(243.78 / 72, 243.78 / 72))
    for method in methods:
        p = data[method]
        x = [q['target'] / 100 for q in p]
        y = [q['percent'] for q in p]
        style = {'markerfacecolor': 'none', **STYLES[method]}  # hollow markers except FISTA
        ax.errorbar(x, y, yerr=[[q['percent'] - q['low'] for q in p], [q['high'] - q['percent'] for q in p]],
                    fmt='none', ecolor=style['color'], elinewidth=1, capsize=2, capthick=1, zorder=style['zorder'] - .5)
        ax.plot(x, y, label=method, linewidth=1.65, markeredgewidth=1.5, **style)
    low = min(q['low'] for p in data.values() for q in p)
    ax.set_xlabel(r'Coverage target $\alpha$', labelpad=6)
    ax.set_ylabel('Feasibility (%)', labelpad=6)
    ax.set_xticks([t / 100 for t in TARGETS], [f'{t / 100:.1f}' for t in TARGETS])
    ax.set_xlim(.08, .42)
    ax.set_ylim(min(48, low - 2), 102)
    ax.set_yticks([t for t in (25, 50, 75, 100) if t >= ax.get_ylim()[0]])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis='y', color='#e9e9e9', linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3, width=.8)
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(.04, .04), handlelength=2, handletextpad=.6,
              labelspacing=.4, borderaxespad=0)
    fig.subplots_adjust(left=.24, right=.97, bottom=.19, top=.96)
    for suffix in ('pdf', 'png'):
        fig.savefig(args.results.parent / f'fig2a_feasibility.{suffix}', dpi=500, facecolor='white')
    for p in data.values():
        print(p[0]['method'].ljust(6), '  '.join(f"{q['feasible']}/{q['total']}" for q in p))


if __name__ == '__main__':
    main()
