#!/usr/bin/env python3
"""Stream per-user JSONL files and infer primary languages without saving posts.

Full outputs are published atomically only after the complete archive is read.
--sample-users produces explicitly named sample outputs, never full outputs.
Language codes are preserved as supplied (lowercased and whitespace stripped).
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import platform
import tarfile
import time
import zlib

try:
    import orjson
    loads = orjson.loads
except ImportError:
    loads = json.loads

try:
    import isal
    from isal.igzip import IGzipFile
except ImportError:
    IGzipFile = None


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.md5()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def tell(self):
        return self.stream.tell()


def atomic_json(path, obj):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def infer(archive, output_dir, status_path, sample_users=None, decompressor='auto', skip_invalid_json=False):
    if decompressor not in ('auto', 'isal', 'stdlib'):
        raise ValueError('Unknown decompressor')
    use_isal = IGzipFile is not None and decompressor != 'stdlib'
    if decompressor == 'isal' and IGzipFile is None:
        raise ValueError('ISA-L requested but python-isal is not installed')
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = '.sample' if sample_users is not None else ''
    output = output_dir / f'user_languages{suffix}.tsv'
    summary_path = output_dir / f'language_summary{suffix}.json'
    temporary = output.with_name(output.name + '.tmp')
    invalid_output = output_dir / f'invalid_json_records{suffix}.jsonl'
    invalid_temporary = invalid_output.with_name(invalid_output.name + '.tmp')
    stats = Counter(dict.fromkeys([
        'users_processed', 'eligible_users', 'users_without_tagged_authored_posts',
        'users_with_tied_primary_language', 'posts_total', 'reposts_excluded',
        'authored_posts', 'authored_posts_missing_or_null_langs',
        'tagged_authored_posts', 'untagged_authored_posts',
        'multilingual_authored_posts', 'null_language_entries',
        'ignored_language_entries', 'blank_lines', 'parsed_posts',
        'json_parser_fallback_records', 'invalid_json_records_skipped',
    ], 0))
    languages = Counter()
    language_votes = Counter()
    seen = set()
    start = time.monotonic()
    last_status = 0.0
    file_stat = archive.stat()
    provenance = {'path': str(archive.resolve()), 'size_bytes': file_stat.st_size,
                  'mtime_ns': file_stat.st_mtime_ns}
    implementation = {
        'python': platform.python_version(),
        'json_parser': 'orjson' if loads is not json.loads else 'stdlib.json',
        'json_parser_version': orjson.__version__ if loads is not json.loads else platform.python_version(),
        'decompressor': 'isal.igzip' if use_isal else 'tarfile/zlib',
        'decompressor_version': isal.__version__ if use_isal else zlib.ZLIB_RUNTIME_VERSION,
    }

    def status(state, force=False, error=None):
        nonlocal last_status
        now = time.monotonic()
        if force or now - last_status >= 5:
            value = {'state': state, 'partial': sample_users is not None,
                     'sample_users': sample_users, 'elapsed_seconds': now - start,
                     'counts': dict(stats), 'archive': provenance,
                     'implementation': implementation}
            if error:
                value['error'] = error
            atomic_json(status_path, value)
            last_status = now

    status('running', True)
    try:
        with archive.open('rb') as raw, temporary.open('w') as target, ExitStack() as stack:
            invalid_log = stack.enter_context(invalid_temporary.open('w'))
            reader = HashingReader(raw)
            decoded = stack.enter_context(IGzipFile(fileobj=reader, mode='rb')) if use_isal else None
            target.write('user_id\tlanguage\twinning_votes\ttagged_posts\tauthored_posts\n')
            with tarfile.open(fileobj=decoded if use_isal else reader,
                              mode='r|' if use_isal else 'r|gz') as tar:
                for member in tar:
                    tar.members.clear()  # tarfile otherwise retains millions of headers.
                    if member.isdir():
                        continue
                    name = PurePosixPath(member.name)
                    if not member.isfile() or not re.fullmatch(r'\d+\.jsonl', name.name):
                        raise ValueError(f'Unexpected archive member: {member.name}')
                    uid = int(name.stem)
                    if uid in seen:
                        raise ValueError(f'Duplicate user member: {uid}')
                    seen.add(uid)
                    votes = Counter()
                    authored = tagged = 0
                    source = tar.extractfile(member)
                    if source is None:
                        raise ValueError(f'Cannot read user member: {uid}')
                    with source:
                        for line_number, line in enumerate(source, 1):
                            if not line.strip():
                                stats['blank_lines'] += 1
                                continue
                            stats['posts_total'] += 1
                            try:
                                record = loads(line)
                            except (ValueError, UnicodeDecodeError):
                                try:
                                    record = json.loads(line)
                                    stats['json_parser_fallback_records'] += 1
                                except (ValueError, UnicodeDecodeError) as error:
                                    if not skip_invalid_json:
                                        raise ValueError(f'Invalid JSON: user {uid}, line {line_number}') from None
                                    stats['invalid_json_records_skipped'] += 1
                                    invalid_log.write(json.dumps({
                                        'user_id': uid, 'line': line_number, 'bytes': len(line),
                                        'sha256': hashlib.sha256(line).hexdigest(),
                                        'error': str(error),
                                    }) + '\n')
                                    status('running')
                                    continue
                            if not isinstance(record, dict):
                                raise ValueError(f'Non-object record: user {uid}, line {line_number}')
                            record_uid = record.get('user_id')
                            if type(record_uid) is not int or record_uid != uid:
                                raise ValueError(f'user_id mismatch/type: user {uid}, line {line_number}')
                            stats['parsed_posts'] += 1
                            if record.get('repost_from') is not None or record.get('reposted_author') is not None:
                                stats['reposts_excluded'] += 1
                                status('running')
                                continue
                            authored += 1
                            stats['authored_posts'] += 1
                            tags = record.get('langs')
                            if tags is None:
                                stats['authored_posts_missing_or_null_langs'] += 1
                                tags = []
                            if not isinstance(tags, list):
                                raise ValueError(f'langs must be list/null: user {uid}, line {line_number}')
                            valid = set()
                            for tag in tags:
                                if tag is None:
                                    stats['null_language_entries'] += 1
                                    continue
                                if not isinstance(tag, str):
                                    raise ValueError(f'Non-string language: user {uid}, line {line_number}')
                                normalized = tag.strip().lower()
                                if normalized in ('', 'und', 'mul', 'zxx'):
                                    stats['ignored_language_entries'] += 1
                                    continue
                                valid.add(normalized)
                            if valid:
                                tagged += 1
                                stats['tagged_authored_posts'] += 1
                                votes.update(valid)
                                language_votes.update(valid)
                                if len(valid) > 1:
                                    stats['multilingual_authored_posts'] += 1
                            else:
                                stats['untagged_authored_posts'] += 1
                            status('running')
                    stats['users_processed'] += 1
                    if votes:
                        best = max(votes.values())
                        winners = sorted(tag for tag, count in votes.items() if count == best)
                        language = winners[0]
                        if len(winners) > 1:
                            stats['users_with_tied_primary_language'] += 1
                        languages[language] += 1
                        stats['eligible_users'] += 1
                        target.write(f'{uid}\t{language}\t{best}\t{tagged}\t{authored}\n')
                    else:
                        stats['users_without_tagged_authored_posts'] += 1
                    status('running')
                    if sample_users is not None and stats['users_processed'] >= sample_users:
                        break
            if sample_users is None:
                if decoded is not None:
                    # Consume gzip trailer and verify CRC before publishing outputs.
                    while decoded.read(8 * 1024 * 1024):
                        status('running')
                # Include any raw bytes after tar's end-of-archive marker in MD5.
                while reader.read(8 * 1024 * 1024):
                    status('running')
                provenance['md5'] = reader.digest.hexdigest()
            else:
                provenance['md5'] = None
                provenance['md5_note'] = 'Partial sample: full raw archive was not hashed.'
        end_stat = archive.stat()
        changed = (end_stat.st_size, end_stat.st_mtime_ns) != (file_stat.st_size, file_stat.st_mtime_ns)
        provenance['changed_during_read'] = changed
        if changed and sample_users is None:
            raise ValueError('Archive changed while being read; outputs are not published.')
        summary = {'partial': sample_users is not None, 'sample_users': sample_users,
                   'archive': provenance, 'counts': dict(stats),
                   'implementation': implementation,
                   'primary_language_users': dict(sorted(languages.items())),
                   'language_post_votes': dict(sorted(language_votes.items())),
                   'policy': {'authored_only': True, 'minimum_tagged_authored_posts': 1,
                              'votes': 'One vote per distinct normalized tag per authored post',
                              'normalization': 'Strip whitespace and lowercase; preserve dataset codes',
                              'ignored_tags': ['', 'und', 'mul', 'zxx'],
                              'invalid_json': 'Exclude records rejected by both parsers; count and log hashes' if skip_invalid_json else 'Fail',
                              'json_parser_fallback': 'Try standard-library JSON if the fast parser rejects a record',
                              'tie_break': 'Lexicographically smallest normalized tag',
                              'language_selection': 'All languages; no top-k filtering'},
                   'elapsed_seconds': time.monotonic() - start}
        temporary.replace(output)
        invalid_temporary.replace(invalid_output)
        atomic_json(summary_path, summary)
        status('completed', True)
        return summary
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        status('failed', True, str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('bluesky/processed'))
    parser.add_argument('--status', type=Path, default=Path('bluesky/logs/languages_status.json'))
    parser.add_argument('--sample-users', type=int, help='Process only N users; outputs use .sample names')
    parser.add_argument('--decompressor', choices=['auto', 'isal', 'stdlib'], default='auto',
                        help='Use optional ISA-L when installed, otherwise standard-library streaming gzip')
    parser.add_argument('--skip-invalid-json', action='store_true',
                        help='Exclude and audit records rejected by both JSON parsers')
    args = parser.parse_args()
    if args.sample_users is not None and args.sample_users < 1:
        parser.error('--sample-users must be positive')
    infer(args.archive, args.output_dir, args.status, args.sample_users, args.decompressor, args.skip_invalid_json)


if __name__ == '__main__':
    main()
