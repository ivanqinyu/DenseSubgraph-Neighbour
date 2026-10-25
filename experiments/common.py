"""Helpers shared by the experiment scripts: run a solver, then check its core."""
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def solve(algorithm, graph, output, timeout=None, **options):
    """Run one solver through run.py and return its records, or None after `timeout` seconds.

    Options become command-line flags (time_limit=30 -> --time-limit 30). A finished
    run (run.json written) is reused; an interrupted one is started again; a timeout is
    remembered in timeout.json.
    """
    output = Path(output)
    if (output / 'run.json').exists():
        return read_records(output)
    if (output / 'timeout.json').exists():
        return None
    shutil.rmtree(output, ignore_errors=True)
    command = [sys.executable, str(ROOT / 'run.py'), '--algorithm', algorithm, '--input', str(graph),
               '--output', str(output)]
    for key, value in options.items():
        command += ['--' + key.replace('_', '-'), str(value)]
    # A new session, so that a timeout also stops the solver that run.py started.
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    try:
        _, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        output.mkdir(parents=True, exist_ok=True)
        (output / 'timeout.json').write_text(json.dumps(dict(seconds=timeout)) + '\n')
        return None
    if process.returncode:
        raise RuntimeError(f'{algorithm} on {graph} failed:\n{stderr[-2000:]}')
    return read_records(output)


def read_records(output):
    return [json.loads(line) for line in (Path(output) / 'records.jsonl').read_text().splitlines() if line.strip()]


def demands(counts, target):
    """External demand ceil(target% of each group size)."""
    return (np.asarray(counts) * target + 99) // 100


class Graph:
    """A dataset loaded once, for checking the cores that solvers return."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.labels = np.load(self.folder / 'labels.npy')
        self.edges = np.load(self.folder / 'edges.npy')
        self.counts = np.bincount(self.labels)

    def core(self, output, record, mask=False):
        """Vertex IDs of a record's core. OPT and Goldberg list it in the record; the internal
        methods write a uint8 membership mask (mask=True); the others a uint32 ID list."""
        if 'core' in record:
            return np.asarray(record['core'], dtype=np.int64)
        path = Path(output) / record['core_file']
        if mask:
            return np.flatnonzero(np.fromfile(path, dtype=np.uint8))
        return np.fromfile(path, dtype='<u4').astype(np.int64)

    def measure(self, core):
        """Size, density and, per group, the distinct outside vertices adjacent to the core."""
        inside = np.zeros(len(self.labels), dtype=bool)
        inside[core] = True
        u, v = self.edges[:, 0], self.edges[:, 1]
        a, b = inside[u], inside[v]
        reached = np.zeros(len(self.labels), dtype=bool)
        reached[v[a & ~b]] = True
        reached[u[b & ~a]] = True
        internal = int(np.count_nonzero(a & b))
        return dict(size=len(core), density=internal / len(core) if len(core) else 0.0,
                    coverage=np.bincount(self.labels[reached], minlength=len(self.counts)))

    def feasible(self, coverage, target):
        return bool(np.all(coverage >= demands(self.counts, target)))
