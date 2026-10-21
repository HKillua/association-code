"""Evaluate a completed native-time checkpoint's selected snapshot on full V_select.

An explicit device and a separately authorized new wall-time budget are required.
This entry point does not resume training, fit a normalizer, calibrate a margin,
select a new threshold, or open T_test. The output directory must be new.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _source_settings(saved, manifest, alignment):
    """Compare saved semantics directly, without a source/hash/version gate."""
    from association.training.tmtaff import STAGES, _configuration
    from association.data.archive import validate_manifest

    if saved.get('format_version') != 1 or saved.get('input_adapter') != 'native_time_v1':
        raise ValueError('an explicit native_time_v1 complete checkpoint is required')
    config = _configuration(saved['config'])
    selected = saved.get('selected')
    if not isinstance(selected, dict) or 'threshold' not in selected:
        raise ValueError('checkpoint must contain a selected snapshot and its original threshold')
    states = selected.get('states')
    if not isinstance(states, (list, tuple)) or len(states) != 3 or not all(isinstance(state, dict) for state in states):
        raise ValueError('selected snapshot must contain all association, fusion and objective states')
    threshold = selected['threshold']
    if threshold is not None and (isinstance(threshold, bool)
            or not isinstance(threshold, (int, float)) or not math.isfinite(threshold)):
        raise ValueError('saved threshold must be finite or None (all rejected)')
    expected_updates = [(stage, update) for stage in STAGES
                        for update in range(1, config['updates'][stage] + 1)]
    actual_updates = [(row.get('stage'), row.get('update')) for row in saved.get('history', [])]
    cursor = saved.get('cursor', {})
    # The last durable checkpoint may precede the loop's final stage increment.
    final_cursor = (cursor.get('stage'), cursor.get('update')) in (
        (len(STAGES), 0), (len(STAGES)-1, config['updates'][STAGES[-1]]))
    if (actual_updates != expected_updates or cursor.get('completed_updates') != len(expected_updates)
            or not final_cursor):
        raise ValueError('all configured training updates must already be complete; evaluation never resumes them')
    normalizer = saved['normalizer']
    train_splits = normalizer.get('training_splits')
    if train_splits not in (['T_core'], ['T_core', 'T_head']):
        raise ValueError('saved normalizer must declare development training splits')
    adapter = normalizer.get('adapter_config', {})
    data = {key: adapter[key] for key in ('length', 'max_observations', 'time_unit_seconds')}
    if (normalizer.get('input_adapter') != 'native_time_v1'
            or data['max_observations'] != saved.get('max_observations')
            or data['length'] != saved['model_config']['length']
            or data['length'] != saved['fusion_config']['length']):
        raise ValueError('checkpoint model, normalizer and native adapter settings disagree')
    contract = saved['data_contract']
    if not isinstance(contract, list) or len(contract) != 2 or not all(contract):
        raise ValueError('checkpoint must retain its full training and validation data contract')
    longitude_policies = {row['reference_alignment'].get('longitude_policy', 'linear')
                          for population in contract for row in population}
    if len(longitude_policies) != 1 or not longitude_policies.issubset({'linear', 'shortest_arc'}):
        raise ValueError('checkpoint reference longitude policies must agree across all scenes')
    longitude_policy = next(iter(longitude_policies))
    if longitude_policy != 'linear':
        data['reference_longitude_policy'] = longitude_policy
    validate_manifest(manifest)
    for rows, splits in zip(contract, (train_splits, ['V_select'])):
        ids = [sid for split in splits for sid in manifest['splits'][split]]
        if [row['scene_id'] for row in rows] != ids:
            raise ValueError('manifest scene population/order differs from the checkpoint')
        for row in rows:
            sid = row['scene_id']
            if (row['split'] not in splits or sid not in manifest['splits'][row['split']]
                    or row['component_id'] != manifest['components'][sid]
                    or row.get('input_adapter') != 'native_time_v1'
                    or row.get('max_observations') != data['max_observations']
                    or row.get('time_unit_seconds') != data['time_unit_seconds']):
                raise ValueError('manifest roles/components or saved adapter semantics differ')
            if alignment.get(sid) != {key: row['reference_alignment'][key]
                                     for key in ('time_offset_seconds', 'max_gap_seconds')}:
                raise ValueError(f'reference alignment differs from the checkpoint: scene {sid}')
    if normalizer.get('training_scene_ids') != [row['scene_id'] for row in contract[0]]:
        raise ValueError('normalizer training population differs from the saved data contract')
    return config, data, train_splits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, nargs='+', required=True,
                        help='one ZIP or all continuous ZIP parts in byte order')
    for name in ('checkpoint', 'manifest', 'alignment', 'output', 'device'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-seconds', type=float, required=True,
                        help='new total budget including loading, preparation, evaluation and result I/O')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    started = time.monotonic()
    output, owns_output = Path(args.output), False
    previous_signal = signal.getsignal(signal.SIGTERM)
    run = dict(status='RUNNING', mode='selected_snapshot_evaluation_only',
               device=args.device, budget_seconds=args.max_seconds,
               source_checkpoint=str(Path(args.checkpoint).resolve()),
               archive=[str(path.resolve()) for path in args.archive], manifest=str(Path(args.manifest).resolve()),
               alignment=str(Path(args.alignment).resolve()), stage='loading',
               optimizer_updates=0, normalizer_refit=False, margin_recalibrated=False,
               threshold_reselected=False, test_opened=False,
               budget_scope='this invocation, imports/checkpoint loading/preparation/evaluation/result I/O',
               resume_from=None)

    def check():
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError('evaluation-only total budget exceeded')

    def write_run(status, **extra):
        # Failure evidence remains writable even if importing torch itself fails.
        run.update(status=status, elapsed_seconds=time.monotonic()-started, **extra)
        temporary = output / 'run.json.tmp'
        temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(output / 'run.json')

    def terminated(signum, frame):
        raise InterruptedError('SIGTERM: evaluation-only invocation interrupted')

    signal.signal(signal.SIGTERM, terminated)
    try:
        if not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.cpu_threads < 1:
            raise ValueError('--max-seconds must be finite and positive; --cpu-threads must be positive')
        output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        write_run('RUNNING')
        import torch
        from association.utils import runtime
        from association.training.tmtaff import _scenes, load_complete_snapshot
        from association.evaluation.tmtaff import evaluate_complete
        from association.data.dataset import load_timed_dataset
        check()
        torch.set_num_threads(args.cpu_threads)
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        manifest, alignment = runtime.read_json(args.manifest), runtime.read_json(args.alignment)
        config, data, train_splits = _source_settings(saved, manifest, alignment)
        run.update(data=data, training_splits=train_splits, source_cursor=saved['cursor'],
                   source_selection={key: value for key, value in saved['selected'].items() if key != 'states'},
                   validation_batches=config['validation_batches'], singleton_policy=config['singleton_policy'])
        run['stage'] = 'preparation'
        write_run('RUNNING')
        check()
        dataset = load_timed_dataset(args.archive, manifest,
            train_splits=train_splits, data_config=data,
            alignment_by_scene=alignment, check=check)
        check()
        if _scenes(dataset['train'], dataset['validation'], data['length']) != saved['data_contract']:
            raise ValueError('prepared scene contract differs from the checkpoint; evaluation aborted')
        check()
        preparation = dict(report=dataset['report'], elapsed_seconds=time.monotonic()-started,
            training_pool_purpose='original data-contract comparison only; no refit, calibration or optimization')
        runtime.write_json(output / 'preparation.json', preparation)
        validation = dataset['validation']
        run['validation_scene_ids'] = [scene['scene_id'] for scene in validation]
        run['validation_target_count'] = sum(len(scene['fusion_targets']) for scene in validation)
        run['validation_node_count'] = sum(len(scene['node_ids']) for scene in validation)
        del dataset  # Release the prepared training pool before model allocation.
        check()
        run['stage'] = 'model_loading'
        write_run('RUNNING')
        device = runtime.device_from_string(args.device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        elif device.type == 'cpu':
            device = torch.device('cpu')
        loaded = load_complete_snapshot(args.checkpoint, device=device, selected=True)
        if (loaded['normalizer'] != saved['normalizer']
                or loaded['selection'] != run['source_selection']):
            raise ValueError('loaded selected snapshot metadata differs from the inspected checkpoint')
        del saved
        run.update(stage='evaluation', device=str(device))
        write_run('RUNNING')
        check()

        def progress(item):
            check()
            run['evaluation_progress'] = item
            runtime.write_json(output / 'evaluation-progress.json', item)
            check()

        evaluation = evaluate_complete(loaded['model'], loaded['fusion'], loaded['normalizer'],
            validation, device=device, batch_tracks=config['validation_batches']['tracks'],
            pair_batch_size=config['validation_batches']['pairs'], select_on_validation=False,
            threshold=loaded['selection']['threshold'], singleton_policy=config['singleton_policy'],
            check=check, progress=progress)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        check()
        if evaluation['selection'] is not None or evaluation['threshold'] != loaded['selection']['threshold']:
            raise ValueError('evaluation changed the saved selection/threshold')
        run['stage'] = 'result_io'
        runtime.write_json(output / 'summary.json', dict(
            mode=run['mode'], selection=loaded['selection'], normalizer=loaded['normalizer'],
            selection_scope='original checkpoint V_select association macro-F1 selection; no reselection',
            evaluation=evaluation, test_opened=False))
        check()
        run['stage'] = 'completed'
        write_run('COMPLETED')
        check()  # A late result-write overrun is INTERRUPTED, never a false success.
        print(f'COMPLETED selected-snapshot V_select evaluation: {output.resolve()}', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, (TimeoutError, InterruptedError, KeyboardInterrupt))
        if owns_output:
            # A complete-looking summary is not authoritative after an overrun.
            summary = output / 'summary.json'
            if summary.exists():
                summary.replace(output / 'summary.incomplete.json')
            write_run('INTERRUPTED' if interrupted else 'FAILED',
                      failure_type=type(error).__name__, failure=str(error))
        print(f'{"INTERRUPTED" if interrupted else "FAILED"} {type(error).__name__}: {error}',
              file=sys.stderr, flush=True)
        return 124 if isinstance(error, TimeoutError) else 130 if interrupted else 1
    finally:
        signal.signal(signal.SIGTERM, previous_signal)


if __name__ == '__main__':
    raise SystemExit(main())
