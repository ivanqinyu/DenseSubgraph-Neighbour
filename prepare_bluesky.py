#!/usr/bin/env python3
"""Prepare the mutual-following, dominant-language Bluesky graph from Zenodo v3."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = 'https://zenodo.org/records/14669616'
FILES = ('followers.csv.gz', 'user_posts.tar.gz')


def digest(path, algorithm='sha256'):
    value = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def download(entry, folder):
    """Resume a partial transfer and verify the upstream checksum before publishing."""
    target = folder / entry['key']
    partial = target.with_name(target.name + '.part')
    expected = int(entry['size'])
    algorithm, checksum = entry['checksum'].split(':', 1)
    if not target.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise ValueError(f'Partial download too large: {partial}')
        if offset < expected:
            headers = {'User-Agent': 'dsp-neighbour-data-preparation/1.0'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(entry['links']['self'], headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    offset = 0
                if response.status == 206 and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected download range')
                with partial.open('ab' if offset else 'wb') as stream:
                    for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                        stream.write(block)
        if partial.stat().st_size != expected or digest(partial, algorithm) != checksum:
            raise ValueError(f'Download size/checksum mismatch: {partial}')
        partial.replace(target)
    elif target.stat().st_size != expected or digest(target, algorithm) != checksum:
        raise ValueError(f'Existing file size/checksum mismatch: {target}')
    return target


def validate(folder):
    metadata = json.loads((folder / 'metadata.json').read_text())
    edges = np.load(folder / 'edges.npy', mmap_mode='r')
    labels = np.load(folder / 'labels.npy', mmap_mode='r')
    ids = np.load(folder / 'user_ids.npy', mmap_mode='r')
    if edges.dtype != np.dtype('<u4') or edges.shape != (metadata['m'], 2):
        raise ValueError('Invalid edge array')
    if labels.dtype != np.dtype('<u2') or labels.shape != (metadata['n'],):
        raise ValueError('Invalid label array')
    if ids.dtype != np.dtype('<u4') or ids.shape != labels.shape or not np.all(ids[1:] > ids[:-1]):
        raise ValueError('Invalid original user ID map')
    counts = np.bincount(labels)
    if len(counts) != len(metadata['groups']):
        raise ValueError('Invalid group count')
    for group in metadata['groups']:
        if int(counts[group['id']]) != group['population']:
            raise ValueError('Group population mismatch')
    previous = -1
    for start in range(0, len(edges), 1_000_000):
        block = edges[start:start + 1_000_000]
        keys = (block[:, 0].astype(np.uint64) << 32) | block[:, 1]
        if not (np.all(block[:, 0] < block[:, 1]) and int(block.max()) < len(labels)
                and int(keys[0]) > previous and np.all(keys[1:] > keys[:-1])):
            raise ValueError('Edges must be normalized, sorted, unique and in range')
        previous = int(keys[-1])
    return metadata


def write_metadata(folder, inputs, summary, sources):
    """Write metadata.json in the schema shared by all datasets; build details go to preparation.json."""
    build = validate(folder)
    groups = sorted(build['groups'], key=lambda group: group['id'])
    metadata = {
        'dataset': folder.name, 'format': 'nfds-compact-v1', 'n': build['n'], 'm': build['m'],
        'groups': {str(group['id']): group['population'] for group in groups},
        'original_label_map': {str(group['id']): group['language'] for group in groups},
        'source': f'Failla and Rossetti, Bluesky Social Dataset v3 ({SOURCE})',
        'notes': ['Labels are the dominant language of authored, language-tagged posts, not country or nationality.',
                  'Edges join users who follow each other.',
                  'Source archives: ' + ', '.join(f"{name} md5:{info['md5']}" for name, info in inputs.items()) + '.',
                  'user_ids.npy maps each vertex to its anonymous upstream user ID; build details are in preparation.json.'],
        'sha256': {name: digest(folder / name) for name in ('edges.npy', 'labels.npy', 'user_ids.npy')},
    }
    preparation = dict(build, source_files=inputs, language_policy=summary['policy'],
                       language_counts=summary['counts'], implementation=summary['implementation'],
                       preparation_source_sha256={str(path.relative_to(ROOT)): digest(path) for path in sources})
    (folder / 'preparation.json').write_text(json.dumps(preparation, indent=2) + '\n')
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True, help='Intermediate files and optional downloads')
    parser.add_argument('--output', type=Path, required=True, help='New prepared graph directory')
    parser.add_argument('--download', action='store_true', help='Fetch public manifest and archives from Zenodo')
    parser.add_argument('--followers', type=Path, help='Existing followers.csv.gz (offline mode)')
    parser.add_argument('--posts', type=Path, help='Existing user_posts.tar.gz (offline mode)')
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 65535:
        parser.error('--top-k must be between 1 and 65535')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('--output must be new or empty')
    if not args.download and not (args.followers and args.posts):
        parser.error('Provide --download or both --followers and --posts')
    if args.download and (args.followers or args.posts):
        parser.error('Use either --download or local archive paths')
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.download:
        with urllib.request.urlopen('https://zenodo.org/api/records/14669616', timeout=60) as response:
            manifest = json.load(response)
        (work / 'zenodo_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        entries = {entry['key']: entry for entry in manifest['files']}
        followers, posts = (download(entries[name], work) for name in FILES)
    else:
        followers, posts = args.followers.resolve(), args.posts.resolve()
    # Hashing also binds resumed intermediates to their exact input archives.
    inputs = {name: {'size': path.stat().st_size, 'md5': digest(path, 'md5')}
              for name, path in zip(FILES, (followers, posts))}
    identity = work / 'input_identity.json'
    if identity.exists() and json.loads(identity.read_text()) != inputs:
        raise ValueError('Input archives changed; use a fresh --work-dir')
    identity.write_text(json.dumps(inputs, indent=2) + '\n')
    helpers = ROOT / 'tools' / 'bluesky'
    binary = work / 'build_graph'
    subprocess.run([args.compiler, '-O3', '-std=c++17', str(helpers / 'build_graph.cpp'), '-lz', '-o', str(binary)], check=True)
    processed = work / 'languages'
    summary_path = processed / 'language_summary.json'
    if not summary_path.exists():
        subprocess.run([sys.executable, str(helpers / 'infer_languages.py'), '--archive', str(posts),
                        '--output-dir', str(processed), '--status', str(work / 'language_status.json'),
                        '--skip-invalid-json'], check=True)
    summary = json.loads(summary_path.read_text())
    if summary['partial'] or summary['archive']['md5'] != inputs[FILES[1]]['md5']:
        raise ValueError('Incomplete language table or source checksum mismatch')
    subprocess.run([str(binary), '--followers', str(followers), '--cache-dir', str(work / 'reciprocal'),
                    '--languages', str(processed / 'user_languages.tsv'), '--output', str(args.output),
                    '--top-k', str(args.top_k)], check=True)
    metadata = write_metadata(args.output, inputs, summary,
                              (Path(__file__).resolve(), helpers / 'infer_languages.py', helpers / 'build_graph.cpp'))
    print(json.dumps({'n': metadata['n'], 'm': metadata['m'], 'groups': len(metadata['groups']), 'validated': True}))


if __name__ == '__main__':
    main()
