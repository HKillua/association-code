"""Copy small, hash-bound profile artifacts for review; never run a model.

This standard-library-only collector leaves originals untouched. COLLECTED
describes evidence transport, not acceptance of the experiment or its claims.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


LIMIT = 256 * 1024


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def collect(profile_dir, acceptance, output, attachments=()):
    profile_dir, acceptance, output = map(Path, (profile_dir, acceptance, output))
    output.mkdir(parents=True, exist_ok=False)
    files = output / 'files'
    files.mkdir()
    report = {'status': 'COLLECTING', 'scope': 'artifact transport only; no training or GPU use',
              'files': [], 'errors': [], 'limits': [], 'attachments_supplied': len(attachments)}
    documents = {}

    def save(source, name, expected=None):
        item = {'source': str(source.resolve()), 'name': name}
        report['files'].append(item)
        if not source.is_file():
            item['status'] = 'MISSING'
            report['errors'].append(f'Missing file: {source}')
            return None
        item.update(bytes=source.stat().st_size, sha256=digest(source))
        if expected is not None and item['sha256'] != expected:
            item['status'] = 'HASH_MISMATCH'
            report['errors'].append(f'Hash mismatch: {source}')
            return None
        if item['bytes'] > LIMIT:
            item['status'] = 'METADATA_ONLY_SIZE_LIMIT'
            report['limits'].append(f'Over {LIMIT} bytes, original retained: {source}')
            return None
        with source.open('rb') as stream:
            payload = stream.read(LIMIT + 1)
        # Detect a change between the hash pass and copy; never reserialize JSON.
        if len(payload) != item['bytes'] or len(payload) > LIMIT or hashlib.sha256(payload).hexdigest() != item['sha256']:
            item['status'] = 'SOURCE_CHANGED'
            report['errors'].append(f'Source changed during collection: {source}')
            return None
        destination = files / name
        with destination.open('xb') as stream:
            stream.write(payload)
        item.update(status='COPIED', copy=str(destination.relative_to(output)),
                    copy_sha256=digest(destination))
        if item['copy_sha256'] != item['sha256']:
            item['status'] = 'COPY_HASH_MISMATCH'
            report['errors'].append(f'Copied bytes disagree with source hash: {destination}')
            return None
        return payload

    try:
        acceptance_bytes = save(acceptance, 'acceptance.json')
        if acceptance_bytes is None:
            raise ValueError('Acceptance summary is missing or exceeds the copy limit')
        accepted = json.loads(acceptance_bytes)
        for name in ('profile', 'run'):
            expected = accepted[name + '_sha256']
            if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
                raise ValueError(f'Invalid expected {name} SHA256')
            payload = save(profile_dir / (name + '.json'), name + '.json', expected)
            if payload is not None:
                documents[name] = json.loads(payload)
        if len(documents) == 2:
            if documents['run']['profile_sha256'] != accepted['profile_sha256']:
                raise ValueError('run.json does not bind the accepted profile hash')
            for key in ('status', 'peak_allocated_bytes', 'peak_reserved_bytes', 'phase_seconds',
                        'step_phase_seconds', 'step_positive_pair_counts'):
                if documents['profile'][key] != accepted[key]:
                    raise ValueError(f'Original profile disagrees with acceptance: {key}')
            report['profile_run_hash_link_verified'] = True
        for index, source in enumerate(attachments, 1):
            source = Path(source)
            save(source, f'attachment-{index:02d}-{source.name}')
    except (OSError, ValueError, KeyError, TypeError) as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    report['status'] = 'FAILED' if report['errors'] else ('PARTIAL' if report['limits'] else 'COLLECTED')
    (output / 'collection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-dir', required=True)
    parser.add_argument('--acceptance', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--attachment', action='append', default=[], help='Existing wrapper source; copied, never executed')
    args = parser.parse_args(argv)
    try:
        result = collect(args.profile_dir, args.acceptance, args.output, args.attachment)
    except OSError as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return {'COLLECTED': 0, 'PARTIAL': 2, 'FAILED': 1}[result['status']]


if __name__ == '__main__':
    sys.exit(main())
