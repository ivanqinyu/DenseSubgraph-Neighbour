"""Plot Fig. 2b: mean density against mean time at n = 50 for QAD, CO, DtC and OPT.

A graph-target pair is kept when QAD and CO are feasible, OPT proved optimality and
DtC finished with a feasible core. Means are over the kept pairs, each with equal
weight; the dashed line is the mean of Goldberg's unconstrained optimum over the same
pairs. Also writes the kept pairs and the per-method statistics as CSV files.
"""
import argparse
import csv
import math
import os
import statistics
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dsp-neighbour-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

ROOT = Path(__file__).resolve().parents[2]
# Colors and markers of QAD and CO match Fig. 2a; the last field places the label.
STYLES = {'QAD': ('#0072B2', 'o', 'left'), 'CO': ('#D55E00', '^', 'left'), 'DtC': ('#CC79A7', 's', 'right'),
          'OPT': ('#444444', '*', 'left')}


def keep(runs):
    """Reason to leave a graph-target pair out, or '' to keep it."""
    reasons = [m for m in ('QAD', 'CO') if runs[m]['feasible'] != 'True']
    if runs['OPT']['optimal'] != 'True':
        reasons.append('OPT not proven optimal')
    if runs['DtC']['feasible'] != 'True' or runs['DtC']['status'] != 'completed':
        reasons.append(f"DtC {runs['DtC']['status']}")
    return '; '.join(reasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=ROOT / 'outputs/synthetic/results.csv')
    args = parser.parse_args()
    with args.results.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['n'] == '50']
    pairs = {}
    for r in rows:
        pairs.setdefault((r['dataset'], r['target']), {})[r['method']] = r
    kept, report = [], []
    for (dataset, target), runs in sorted(pairs.items()):
        reason = keep(runs)
        report.append(dict(dataset=dataset, target=target, kept=not reason, reason=reason))
        if not reason:
            kept.append(runs)
    stats = []
    for method in ('QAD', 'CO', 'DtC', 'OPT', 'Goldberg'):
        density = [float(runs[method]['density']) for runs in kept]
        seconds = [float(runs[method]['seconds']) for runs in kept]
        ratio = [float(runs[method]['density']) / float(runs['OPT']['density']) for runs in kept]
        stats.append(dict(method=method, pairs=len(kept), density_mean=statistics.mean(density),
                          density_ratio_to_opt=statistics.mean(ratio), seconds_mean=statistics.mean(seconds),
                          seconds_median=statistics.median(seconds)))
    for name, table in (('fig2b_pairs.csv', report), ('fig2b_statistics.csv', stats)):
        with (args.results.parent / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    mean = {s['method']: s for s in stats}

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8.5, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'pdf.fonttype': 42, 'axes.linewidth': .6,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(1.75, 1.65))
    fig.subplots_adjust(left=.29, right=.97, bottom=.23, top=.98)
    for method, (color, marker, side) in STYLES.items():
        x, y = mean[method]['density_mean'], mean[method]['seconds_mean']
        ax.scatter(x, y, color=color, marker=marker, s=40 if marker == '*' else 19, linewidth=.35,
                   edgecolor='white', zorder=4)
        ax.annotate(method, (x, y), xytext=(-5 if side == 'left' else 5, 0), textcoords='offset points',
                    ha='right' if side == 'left' else 'left', va='center', fontsize=8)
    upper = mean['Goldberg']['density_mean']
    ax.axvline(upper, color='#777777', linestyle='--', linewidth=.75)
    ax.text(upper, .98, 'Goldberg ', transform=ax.get_xaxis_transform(), rotation=90, ha='right', va='top',
            fontsize=7)
    # Limits padded around the points: room for labels on the left, one decade above and below.
    densities = [mean[m]['density_mean'] for m in STYLES]
    times = [mean[m]['seconds_mean'] for m in STYLES]
    ax.set_xlim(min(densities) - .12, upper + .1)
    ax.set_xticks([t / 2 for t in range(math.ceil(2 * min(densities) - .2), math.floor(2 * upper) + 1)])
    ax.set_yscale('log')
    low, high = math.floor(math.log10(min(times))) - 1, math.ceil(math.log10(max(times))) + 1
    ax.set_ylim(10 ** low, 10 ** high)
    ax.set_yticks([10.0 ** e for e in range(low + 1, high, 2)])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(which='major', length=2.5, width=.6, pad=2)
    ax.set_xlabel('Mean density', labelpad=3)
    ax.set_ylabel('Mean time (s)', labelpad=1)
    ax.grid(axis='y', alpha=.22, linewidth=.5)
    ax.set_axisbelow(True)
    for suffix in ('pdf', 'png'):
        fig.savefig(args.results.parent / f'fig2b_density_time.{suffix}', dpi=350, facecolor='white')
    print(f'{len(kept)} of {len(pairs)} pairs kept')
    for s in stats:
        print(f"{s['method']:<9} density {s['density_mean']:.3f} ({s['density_ratio_to_opt']:.3f} of OPT), "
              f"time mean {s['seconds_mean']:.3g} s, median {s['seconds_median']:.3g} s")


if __name__ == '__main__':
    main()
