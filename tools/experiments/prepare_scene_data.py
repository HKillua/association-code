"""Prepare train/V alignment and population metadata once; never read T_test."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from association.data.alignment import prepare_scene_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'manifest', 'existing-alignment', 'alignment-output', 'summary-output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    outputs = [Path(args.alignment_output), Path(args.summary_output)]
    if len(set(outputs)) != 2 or any(path.exists() for path in outputs):
        raise FileExistsError('Use two distinct, new output files')
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    existing = json.loads(Path(args.existing_alignment).read_text(encoding='utf-8'))
    try:
        alignment, summary = prepare_scene_data(args.archive, manifest, existing,
            progress=lambda event: print(json.dumps(event), flush=True))
    except Exception as error:
        outputs[1].parent.mkdir(parents=True, exist_ok=True)
        with outputs[1].open('x', encoding='utf-8') as stream:
            json.dump(dict(status='FAILED', error_type=type(error).__name__, error=str(error),
                command=[sys.executable, __file__, *(argv if argv is not None else sys.argv[1:])]), stream, indent=2)
            stream.write('\n')
        raise
    summary['command'] = [sys.executable, __file__, *(argv if argv is not None else sys.argv[1:])]
    for path, payload in zip(outputs, (alignment, summary)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(status=summary['status'], prepared_scenes=summary['prepared_scenes'],
        historical_verified=summary['historical_alignment_verified'], elapsed_seconds=summary['elapsed_seconds'])), flush=True)
    return 0 if summary['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
