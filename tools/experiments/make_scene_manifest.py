"""Create the 2100/300/600 scene protocol from historical manifest metadata only.

No archive, labels, measurements, references or training are opened. Existing
outputs are never replaced. This does not change training component checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from association.data.archive import build_scene_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--source-description', required=True)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    source = json.loads(Path(args.source_manifest).read_text(encoding='utf-8'))
    manifest = build_scene_manifest(source, args.seed, source_description=args.source_description)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), bytes=output.stat().st_size,
        counts={split: len(scenes) for split, scenes in manifest['splits'].items()},
        historical_protection_counts=manifest['historical_protection_counts'],
        unused_scenes=sum(role == 'unused' for role in manifest['scene_splits'].values()),
        new_measurement_reads=0, new_reference_reads=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
