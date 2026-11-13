"""Create a fresh Level1 scene split from ZIP labels, or convert old metadata.

Neither mode reads measurement/reference CSVs. Existing outputs are not replaced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from association.data.archive import MTADArchive, build_scene_manifest
from association.data.scene_manifest import build_fresh_scene_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-manifest', help='Legacy shared-A manifest conversion')
    source.add_argument('--archive', nargs='+', help='Level1 ZIP or ordered continuous ZIP parts')
    parser.add_argument('--source-description', default='prior shared-A manifest')
    parser.add_argument('--train-scenes', type=int, default=2100)
    parser.add_argument('--validation-scenes', type=int, default=300)
    parser.add_argument('--test-scenes', type=int, default=600)
    parser.add_argument('--development-scenes', nargs='+', default=['0'],
                        help='Already previewed Level1 scenes, forced into training (default: 0)')
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists: ' + str(output))
    caps = dict(T_core=args.train_scenes, T_head=0, V_select=args.validation_scenes, T_test=args.test_scenes)
    label_reads = 0
    if args.archive:
        with MTADArchive(args.archive) as archive:
            manifest = build_fresh_scene_manifest(archive, args.seed, caps,
                                                 development_scenes=args.development_scenes)
            label_reads = len(archive.members_read)
    else:
        previous = json.loads(Path(args.source_manifest).read_text(encoding='utf-8'))
        manifest = build_scene_manifest(previous, args.seed, caps, source_description=args.source_description)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), bytes=output.stat().st_size,
        counts={split: len(scenes) for split, scenes in manifest['splits'].items()},
        historical_protection_counts=manifest['historical_protection_counts'],
        unused_scenes=sum(role == 'unused' for role in manifest['scene_splits'].values()),
        new_label_reads=label_reads,
        new_measurement_reads=0, new_reference_reads=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
