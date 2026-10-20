"""Frozen 048: original 046 / selected 045 groups x mean / final 046 fusion.

Only V native measurements are read. Existing 046 targets/masks are reused for
post-prediction evaluation. No training, threshold selection, reference CSV or
test access. Suggested duration is advisory only; failures are not retried.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np

from association.data.motion_evidence import prepare_motion_scenes
from association.evaluation.evidence_fusion_combo import (
    evaluate_combo, read_json, read_saved_predictions, read_scores)
from association.evaluation.association import evaluate_scenes
from association.training.evidence_fusion import load_checkpoint
from association.utils.runtime import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'manifest', 'association-scores', 'association-summary',
                 'fusion-predictions', 'fusion-checkpoint', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--advisory-seconds', type=float, default=1800.)
    args = parser.parse_args(argv)
    started = time.monotonic()
    output = Path(args.output)
    owns_output = False
    state = dict(status='RUNNING', optimizer_updates=0, test_opened=False,
        reference_csv_read=False, training_data_read=False, threshold_selected=False,
        device=args.device, cpu_threads=args.cpu_threads, advisory_reached=False)
    def checkpoint():
        elapsed = time.monotonic()-started
        if elapsed >= args.advisory_seconds and not state['advisory_reached']:
            state['advisory_reached'] = True
            print('ADVISORY: duration exceeded; continuing the same fixed evaluation, no retry.', flush=True)
            write_json(output/'run.json', dict(state, elapsed_seconds=elapsed))
    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt(f'cancelled by signal {signum}')
    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        if args.cpu_threads < 1 or not 0 < args.advisory_seconds < float('inf'):
            raise ValueError('positive threads and finite advisory duration required')
        torch.set_num_threads(args.cpu_threads)
        if args.device == 'cuda:0' and not torch.cuda.is_available():
            raise ValueError('requested CUDA unavailable; no device fallback')
        output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        write_json(output/'execution.json', dict(command=sys.argv if argv is None else argv,
            inputs={key: str(Path(getattr(args, key)).resolve()) for key in (
                'archive', 'manifest', 'association_scores', 'association_summary',
                'fusion_predictions', 'fusion_checkpoint')},
            advisory_seconds=args.advisory_seconds, device=args.device, cpu_threads=args.cpu_threads))
        write_json(output/'run.json', state)
        summary = read_json(args.association_summary)
        if (summary['status'] != 'COMPLETED' or summary['scope'] != 'native-ols-evidence-association-v1'
                or summary['variant'] != 'full_evidence' or summary['selected']['update'] != 1000):
            raise ValueError('expected completed 045 selected update 1000')
        threshold = summary['selected']['selection']['threshold']
        model, saved_checkpoint = load_checkpoint(args.fusion_checkpoint, device=args.device)
        if saved_checkpoint['step'] != 1000 or saved_checkpoint['configuration']['training']['updates'] != 1000:
            raise ValueError('expected fixed 046 final update 1000')
        saved = read_saved_predictions(args.fusion_predictions)
        scores = read_scores(args.association_scores)
        manifest = read_json(args.manifest)
        scenes = prepare_motion_scenes(args.archive, manifest, 'V_select', check=checkpoint,
                                       include_evaluation_metadata=True)
        report = evaluate_combo(model, scenes, saved, scores, threshold,
                                output=output, device=args.device, check=checkpoint)
        # Only after predictions: detect a mislabelled score file using its
        # frozen threshold, never refit or use these labels for groups/features.
        records = []
        for scene in scenes:
            row = scores[scene['scene_id']]
            identities = saved[scene['scene_id']]['identities']
            records.append(dict(row, scene_id=scene['scene_id'], component_id=scene['component_id'],
                labels=identities[row['pairs'][:, 0]] == identities[row['pairs'][:, 1]]))
        association_check = evaluate_scenes(records, threshold)
        expected_association = summary['selected_result']['metrics']
        association_passed = all(association_check[key] == expected_association[key]
            for key in ('scene_count', 'pair_count', 'tp', 'fp', 'fn', 'tn')) and np.isclose(
                association_check['macro_f1'], expected_association['macro_f1'], rtol=0., atol=1e-12)
        write_json(output/'association-replay.json', dict(passed=bool(association_passed),
            frozen_threshold=threshold, metrics=association_check))
        if not association_passed:
            raise ValueError('saved 045 scores do not reproduce the supplied selected summary')
        expected = dict(scene_count=64, component_count=17, fragment_count=1635, query_count=104640,
                        target_valid_by_attribute=[81207]*4)
        if any(report[key] != value for key, value in expected.items()):
            raise ValueError('048 must retain the complete fixed V population')
        system = read_json(output/'system-evaluation.json')
        if (system['target_identity_count'] != 377 or any(
                metric['available_target_count'] != 18364
                for arm in system['overall'].values() for metric in arm['all'].values())):
            raise ValueError('048 unique-target view must retain 377 targets / 18364 valid points per attribute')
        write_json(output/'evaluation.json', report)
        checkpoint()
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started,
                     threshold045=threshold, replay_passed=report['replay_passed'], **expected)
        write_json(output/'run.json', state)
        print(f'COMPLETED: {output.resolve()}', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if owns_output:
            state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
            write_json(output/'run.json', state)
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
