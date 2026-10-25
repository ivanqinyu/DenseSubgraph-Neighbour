"""Plot Fig. 4 from run.py's coverage.csv and summary.json (needs matplotlib).

Two panels, fixed (10 users per country) and proportional (30%) demands. For each
country: the outside neighbors reached by unconstrained peeling (gray circle), by QAD
(mean over seeds, blue diamond) and the demand (black bar), on a symmetric log scale.
Countries that peeling leaves below the demand have orange labels.
"""
import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dsp-neighbour-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
BLUE, GRAY, ORANGE, BLACK = '#007f89', '#737373', '#ce701b', '#3c4247'
PANELS = [('fixed', '(a) Demand = 10 users per country'), ('proportional', '(b) Demand = 30% per country')]


def peeling_label_offset(panel, value):
    """Placement of the gray labels; the special cases avoid overlaps in the paper's data."""
    if panel == 'proportional' and value == 505:
        return (7, 0), 'left', 'center'
    if panel == 'fixed' and value in (12, 15):
        return (0, 1 if value == 12 else 4), 'center', 'bottom'
    if panel == 'fixed' and value == 29:
        return (0, -3), 'center', 'top'
    return (0, -4), 'center', 'top'


def qad_label_offset(panel, position):
    if panel == 'fixed':
        return 11 if position == 6 else 5
    return {6: 11, 14: 15, 15: 10}.get(position, 5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/lastfm')
    args = parser.parse_args()
    with (args.output / 'coverage.csv').open() as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: int(r['country']))
    summary = json.loads((args.output / 'summary.json').read_text())
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                         'font.size': 11, 'axes.labelsize': 12, 'xtick.labelsize': 8.5, 'ytick.labelsize': 10,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharey=True)
    fig.subplots_adjust(left=.08, right=.99, bottom=.301, top=.85, wspace=.15)
    x = np.arange(len(rows))
    peeling = np.array([int(r['peeling']) for r in rows])
    for ax, (panel, title) in zip(axes, PANELS):
        qad = np.array([float(r[f'{panel}_qad_mean']) for r in rows])
        demand = np.array([int(r[f'{panel}_demand']) for r in rows])
        ax.set_yscale('symlog', linthresh=1, linscale=.65, base=10)
        ax.set_ylim(-.95, 2700)
        ax.set_xlim(-.7, len(x) - .3)
        ax.set_yticks([0, 1, 10, 100, 1000], labels=['0', r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$'])
        ax.vlines(x, peeling, qad, color='#c7cbcc', lw=.9, zorder=2)
        ax.scatter(x, peeling, s=12, facecolors='white', edgecolors=GRAY, linewidths=.8, zorder=4)
        ax.scatter(x, qad, s=12, marker='D', color=BLUE, zorder=3)
        ax.scatter(x, demand, marker='_', s=30, color=BLACK, linewidths=1, zorder=5)
        for i in x:
            offset, ha, va = peeling_label_offset(panel, peeling[i])
            ax.annotate(str(peeling[i]), (i, peeling[i]), xytext=offset, textcoords='offset points', ha=ha, va=va,
                        fontsize=8.5, color=GRAY)
            ax.annotate(str(int(np.floor(qad[i] + .5))), (i, qad[i]), xytext=(0, qad_label_offset(panel, i)),
                        textcoords='offset points', ha='center', va='bottom', fontsize=8.5, color=BLUE)
        ax.set_xticks(x, labels=[r['country'] for r in rows])
        for tick, unmet in zip(ax.get_xticklabels(), peeling < demand):
            if unmet:
                tick.set_color(ORANGE)
        ax.set_xlabel('Country ID', labelpad=3)
        ax.text(.5, -.30, title + '\n' + rf"$\rho_{{QAD}}={summary[f'qad_density_{panel}']:.3f}$",
                transform=ax.transAxes, ha='center', va='top', fontsize=12.5, linespacing=1.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_linewidth(.7)
        ax.tick_params(width=.7, length=2.5, pad=2, labelleft=True)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e9e9e9', lw=.6)
    axes[0].set_ylabel('Neighbor Coverage', labelpad=3)
    handles = [Line2D([], [], marker='o', ms=4, markerfacecolor='white', markeredgecolor=GRAY, ls=''),
               Line2D([], [], marker='D', ms=4, color=BLUE, ls=''),
               Line2D([], [], marker='_', ms=7, color=BLACK, ls='')]
    fig.legend(handles, [rf"Peeling ($\rho={summary['peeling_density']:.3f}$)", 'Neighborhood-constrained (QAD)',
                         'Demand'], loc='upper center', bbox_to_anchor=(.5, .995), ncol=3, fontsize=12, frameon=False,
               handletextpad=.4, handlelength=1, columnspacing=1.3)
    for suffix in ('pdf', 'png'):
        fig.savefig(args.output / f'fig4_lastfm.{suffix}', dpi=400)
    print('saved', args.output / 'fig4_lastfm.pdf')


if __name__ == '__main__':
    main()
