"""Plot the approximation-guarantee figure from run.py's results.csv (needs matplotlib)."""
import argparse
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dtc-guarantee-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

ROOT = Path(__file__).resolve().parents[2]
BOUND = 6
# P, G++ and FISTA return the same core on every instance (asserted below), so
# they share one series. The last entry is a horizontal offset factor that keeps
# overlapping markers apart.
SERIES = [
    ('DtC', 'DtC', dict(color='#CC79A7', marker='s', markerfacecolor='#CC79A7', zorder=6), 1.0),
    ('QAD', 'QAD', dict(color='#0072B2', marker='o', markerfacecolor='none', zorder=5), 1 / 1.18),
    ('CO', 'CO', dict(color='#D55E00', marker='^', markerfacecolor='none', zorder=5), 1.18),
    ('P, G++, FISTA', 'P', dict(color='#808080', marker='v', markerfacecolor='none', zorder=4), 1.0),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/dtc_guarantee')
    args = parser.parse_args()
    with (args.output / 'results.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    by_graph = defaultdict(dict)
    for r in rows:
        by_graph[r['dataset']][r['method']] = r
    for name, methods in by_graph.items():
        assert len(methods) == 6 and all(r['feasible'] == 'True' for r in methods.values()), name
        assert methods['DtC']['status'] == 'completed', name  # complete search, no timeout
        d = methods['DtC']
        assert int(d['m']) >= 3 * (int(d['q']) + 1) * int(d['n']), name  # rho(G) >= 3 d_Sigma
        assert methods['P']['density'] == methods['G++']['density'] == methods['FISTA']['density'], name
    contrasts = sorted({float(r['c']) for r in rows})
    # Printed at 3.3 x 1.6 in without scaling, so font sizes are final point sizes.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7.5,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                         'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(3.3, 1.6))
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    handles = []
    for label, method, style, dodge in SERIES:
        # Worst case over the graphs at each contrast: the guarantee is a worst-case bound.
        y = [max(float(r['ratio_upper']) for r in rows if float(r['c']) == c and r['method'] == method)
             for c in contrasts]
        style = dict(style, markersize=4.2)
        line = '-' if label in ('DtC', 'P, G++, FISTA') else 'none'
        ax.plot([c * dodge for c in contrasts], y, linewidth=1.1 if label == 'DtC' else .9,
                markeredgewidth=1, linestyle=line, **style)
        handles.append(Line2D([], [], linewidth=1.1, markeredgewidth=1, linestyle=line, **style))
    ax.axhline(BOUND, color='#222222', linewidth=.8, linestyle=(0, (4, 2)), zorder=3)
    ax.text(contrasts[-1] * 1.25, BOUND * 1.12, 'DtC guarantee (6)', fontsize=6.5,
            color='#222222', va='bottom', ha='right')
    ax.set_xlim(contrasts[0] / 1.3, contrasts[-1] * 1.3)
    ax.set_ylim(.8, 100)
    ax.xaxis.set_major_locator(FixedLocator(contrasts))
    ax.set_xticklabels([f'{c:g}' for c in contrasts])
    ticks = [1, 4, 16, 64]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_yticklabels([str(t) for t in ticks])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(NullLocator())
    ax.set_xlabel('Density contrast', labelpad=2)
    ax.set_ylabel(r'$\rho(S_{\mathrm{Goldberg}})\,/\,\rho(S)$', labelpad=3)
    ax.grid(axis='y', color='#e9e9e9', linewidth=.5)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2.5, width=.6, pad=2)
    ax.legend(handles=handles, labels=[s[0] for s in SERIES], frameon=False, loc='upper left',
              bbox_to_anchor=(.01, 1.04), ncol=2, handlelength=1.6, handletextpad=.4,
              columnspacing=1, labelspacing=.3, borderaxespad=0)
    fig.subplots_adjust(left=.14, right=.985, bottom=.21, top=.97)
    for suffix in ('pdf', 'png'):
        fig.savefig(args.output / f'dtc_guarantee.{suffix}', dpi=500, facecolor='white')
    plt.close(fig)
    print(args.output / 'dtc_guarantee.pdf')


if __name__ == '__main__':
    main()
