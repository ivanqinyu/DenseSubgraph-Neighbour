#!/usr/bin/env python3
"""Convert local NPY or text edges and labels into the solver's compact format."""
import argparse
import bz2
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def read_array(path, delimiter, skiprows, columns, dtype):
    if path.suffix == '.npy':
        return np.load(path, allow_pickle=False)
    opener = gzip.open if path.suffix == '.gz' else bz2.open if path.suffix == '.bz2' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return np.loadtxt(handle, dtype=dtype, delimiter=delimiter, skiprows=skiprows,
                          usecols=columns, ndmin=2, comments='#')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edges', type=Path, required=True, help='NPY (m,2), or text edge list')
    p.add_argument('--labels', type=Path, help='NPY labels or text; one row per vertex')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pokec-profiles', type=Path,
                   help='SNAP soc-pokec-profiles.txt.gz; rebuild valid-gender induced graph')
    p.add_argument('--delimiter', default=None, help='text delimiter; default whitespace')
    p.add_argument('--labels-delimiter', default=None, help='defaults to --delimiter')
    p.add_argument('--edge-columns', default='0,1', help='zero-based text columns')
    p.add_argument('--label-column', type=int, default=0)
    p.add_argument('--label-id-column', type=int, help='text node-ID column; otherwise labels are in vertex order')
    p.add_argument('--edge-skiprows', type=int, default=0)
    p.add_argument('--label-skiprows', type=int, default=0)
    p.add_argument('--id-offset', type=int, default=0, help='subtract from edge/node IDs; use 1 for one-based IDs')
    p.add_argument('--source', default='', help='public source URL or description')
    a = p.parse_args()
    try:
        cols = tuple(int(c) for c in a.edge_columns.split(','))
        if len(cols) != 2:
            raise ValueError('--edge-columns must contain exactly two columns')
        edges = read_array(a.edges, a.delimiter, a.edge_skiprows, cols, np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in 'iu':
            raise ValueError('edges must be an integer array with shape (m,2)')
        edges = edges.astype(np.int64) - a.id_offset
        if a.pokec_profiles:
            if a.labels or a.id_offset:
                raise ValueError('--pokec-profiles cannot be combined with --labels or --id-offset')
            opener = gzip.open if a.pokec_profiles.suffix == '.gz' else open
            records = []
            with opener(a.pokec_profiles, 'rt', encoding='utf-8') as f:
                for line in f:
                    fields = line.rstrip('\n').split('\t')
                    if len(fields) >= 4 and fields[3] in ('0', '1'):
                        records.append((int(fields[0]), int(fields[3])))
            records.sort()
            profiles = np.asarray(records, dtype=np.int64)
            if profiles.ndim != 2 or len(np.unique(profiles[:, 0])) != len(profiles):
                raise ValueError('no valid profiles or duplicate profile IDs')
            ids, raw = profiles[:, 0], profiles[:, 1]
            mapped = np.searchsorted(ids, edges)
            inside = mapped < len(ids)
            matches = inside & (ids[np.minimum(mapped, len(ids)-1)] == edges)
            edges = mapped[matches.all(axis=1)]
        else:
            if a.labels is None:
                raise ValueError('--labels or --pokec-profiles is required')
            delimiter = a.labels_delimiter if a.labels_delimiter is not None else a.delimiter
            labelcols = (a.label_column,) if a.label_id_column is None else (a.label_id_column, a.label_column)
            raw = read_array(a.labels, delimiter, a.label_skiprows, labelcols, str)
            if a.label_id_column is not None:
                if a.labels.suffix == '.npy':
                    raise ValueError('--label-id-column applies only to text labels')
                ids = raw[:, 0].astype(np.int64) - a.id_offset
                if not np.array_equal(np.sort(ids), np.arange(len(ids))):
                    raise ValueError('label IDs must cover each vertex 0..n-1 exactly once after offset')
                raw = raw[np.argsort(ids), 1]
            elif raw.ndim == 2 and raw.shape[1] == 1:
                raw = raw[:, 0]
        if raw.ndim != 1 or len(raw) == 0:
            raise ValueError('labels must be a nonempty vector (one categorical label per vertex)')
        if raw.dtype.kind == 'f' and not np.isfinite(raw).all():
            raise ValueError('labels contain missing/nonfinite values')
        # Preserve numeric category ordering across text and NPY inputs.
        # Mixed/nonnumeric labels remain categorical strings.
        if raw.dtype.kind in 'US':
            try:
                numeric = np.asarray([int(value) for value in raw], dtype=np.int64)
            except (ValueError, OverflowError):
                pass
            else:
                raw = numeric
        names, labels = np.unique(raw, return_inverse=True)
        if len(names) > 65535 or len(labels) >= 2**32:
            raise ValueError('compact format supports at most 65535 groups and fewer than 2^32 vertices')
        if edges.size and (edges.min() < 0 or edges.max() >= len(labels)):
            raise ValueError('edge endpoint outside label vector; do not omit isolated vertex labels')
        before = len(edges)
        edges.sort(axis=1)
        edges = edges[edges[:, 0] != edges[:, 1]].astype('<u4')
        # A packed key avoids the large temporary structured arrays from unique(axis=0).
        keys = (edges[:, 0].astype(np.uint64) << np.uint64(32)) | edges[:, 1]
        keys = np.unique(keys)
        edges = np.column_stack((keys >> np.uint64(32), keys & np.uint64(0xffffffff))).astype('<u4')
        a.output.mkdir(parents=True, exist_ok=True)
        np.save(a.output / 'edges.npy', edges)
        np.save(a.output / 'labels.npy', labels.astype('<u2'))
        counts = np.bincount(labels)
        metadata = {'dataset': a.output.name, 'format': 'nfds-compact-v1', 'n': len(labels),
                    'm': len(edges), 'groups': {str(i): int(c) for i, c in enumerate(counts)},
                    'original_label_map': {str(i): str(v) for i, v in enumerate(names)},
                    'source': a.source,
                    'notes': [f'Removed {before-len(edges)} self-loop or duplicate edge records.'],
                    'sha256': {f: digest(a.output/f) for f in ('edges.npy', 'labels.npy')}}
        (a.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(json.dumps({'output': str(a.output), 'n': len(labels), 'm': len(edges), 'groups': len(names)}))
    except (ValueError, OSError, IndexError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
