"""Run the approved unit-baseline A once, inside launch_detached's one service.

Specification: archive/manifest/alignment/fixed056/090C inputs must exist;
output must be new and disjoint from them. This checkout's two unit-draft
configs fix fresh TCM2424 -> fresh TFM1000 -> fixed056 evaluation, CPU4 and
cuda:0 (the outer launcher binds one GPU UUID). No shell, resume, retries,
sampling changes, automatic GPU selection, or training beyond those counts.
Each child inherits our process group, writes separate stdout/stderr, and has
its actual exit code recorded. Exit zero alone is insufficient: completed
run/evaluation, exact counts/config/unit/checkpoint and complete V artifacts
must pass before starting the next child. A failure or TERM stops the chain
and preserves all files. launch_detached owns durable hosting/group cleanup.

Acceptance -> focused evidence: ordered success; nonzero/missing/incomplete/
wrong-unit endpoint stops successors; outputs cannot be reused; a real short
child shares our PGID and TERM is forwarded with its true exit retained.
Tests use synthetic artifacts or substituted children, never research data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from association.utils import runtime


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _predictions(directory, scene_ids):
    index = runtime.read_json(directory/'index.json')
    _require(index.get('test_opened') is False and
             [row['scene_id'] for row in index['scenes']] == scene_ids,
             'prediction index does not cover the complete ordered V population')
    for row in index['scenes']:
        for key in ('npz', 'metadata'):
            path = (directory/row[key]).resolve()
            _require(path.parent == directory.resolve() and path.is_file() and path.stat().st_size > 0,
                     'missing or invalid scene prediction artifact')


def _validate_stage(name, output, configs, scene_ids):
    """Read completed artifacts on CPU; never construct a model or open ZIPs."""
    import torch
    from association.data.physical_units import normalizer_speed_policy, SPEED_UNIT_POLICY
    from association.models.tmtaff import ModelConfig
    from association.models.fusion import PairFusionConfig
    from tools.experiments.completion_value_fit import _same_tree

    directory = output/name
    run, summary = [runtime.read_json(directory/file) for file in ('run.json', 'summary.json')]
    _require(run.get('status') == 'COMPLETED' and run.get('test_opened') is False
             and summary.get('test_opened') is False, name+' did not complete')
    if name == 'fixed056':
        _require(run.get('scenes') == len(scene_ids) == 300 and
                 run.get('optimizer_updates') == 0 and run.get('tam_scoring_forwards') == 0 and
                 run.get('threshold_reselected') is False, 'fixed056 changed its evaluation scope')
        _require([row['scene_id'] for row in summary['evaluation']['per_scene']] == scene_ids,
                 'fixed056 summary has incomplete V population')
        _predictions(directory/'predictions', scene_ids)
        return dict(completed_scenes=len(scene_ids), optimizer_updates=0)

    expected = configs[name]
    updates, stage, stage_index = (2424, 'TCM', 0) if name == 'tcm' else (1000, 'TFM_FROZEN', 2)
    preparation = runtime.read_json(directory/'preparation.json')
    payload = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=True)
    history = runtime.read_json(directory/'history.json')
    coverage = runtime.read_json(directory/'coverage.json')
    cursor = dict(stage=stage_index, update=updates, completed_updates=updates)
    _require(run.get('phase') == name and run.get('final_evaluation_status') == 'COMPLETED'
             and run.get('selection_performed') is False and run['cursor']['completed_updates'] == updates
             and payload.get('phase') == name and payload.get('format_version') == 1
             and payload.get('input_adapter') == 'native_time_v1'
             and payload['cursor'] == cursor and run.get('checkpoint_cursor') == cursor,
             name+' is not the fixed completed training/evaluation endpoint')
    _require(preparation['configuration'] == expected and run['config'] == expected['training']
             and payload['config'] == expected['training']
             and payload['model_config'] == asdict(ModelConfig(**expected['model']))
             and payload['fusion_config'] == asdict(PairFusionConfig(**expected['fusion'])),
             name+' configuration differs from the frozen A configuration')
    _require(normalizer_speed_policy(payload['normalizer']) == SPEED_UNIT_POLICY and
             _same_tree(payload['normalizer'], summary['normalizer']), name+' unit contract differs')
    _require(len(payload['states']) >= 2, name+' lacks model/fusion state')
    for state in payload['states'][:2]:
        marker = state.get('speed_unit_contract')
        _require(isinstance(marker, torch.Tensor) and marker.shape == torch.Size([])
                 and marker.dtype == torch.int64 and marker.item() == 1, name+' lacks physical unit marker')
    _require(_same_tree(history, runtime.jsonable(payload['history']))
             and _same_tree(coverage, runtime.jsonable(payload['coverage']))
             and _same_tree(coverage, summary['coverage']) and
             [row.get('update') for row in history if row.get('stage') == stage] == list(range(1, updates+1)),
             name+' successful-update history/coverage differs from its checkpoint')
    if name == 'tcm':
        item = coverage['TCM']
        _require(payload.get('selected') is None and item.get('completed_epochs') == 1
                 and item.get('track_presentations') == item.get('unique_track_count') == 155134
                 and item.get('transition_presentations') == 35555236,
                 'TCM is not the approved full ordered epoch')
        evaluation = runtime.read_json(directory/'completion-evaluation.json')
        _require(evaluation.get('scene_count') == len(scene_ids) == 300 and
                 [row['scene_id'] for row in evaluation['per_scene']] == scene_ids,
                 'TCM final evaluation has incomplete V population')
    else:
        from tools.experiments.tmtaff_fixed056_evaluate import bind_endpoint, source_exposure
        bound_run, _, bound_payload = bind_endpoint(directory)
        source_exposure(bound_run, bound_payload)
        _require(Path(payload['initialization']['source_run']).resolve() == (output/'tcm').resolve()
                 and payload.get('pairing_audit') is True and (directory/'pairing-initial.pt').is_file()
                 and coverage['TFM_FROZEN']['pair_presentations'] == 64000,
                 'TFM source, audit or pair exposure differs')
        endpoints = summary['fusion_evaluations']
        _require([row['update'] for row in endpoints] == [0, 1000]
                 and all(row['status'] == 'COMPLETED' for row in endpoints),
                 'TFM requires both complete fixed oracle endpoints')
        for update in (0, 1000):
            _require((directory/'snapshots'/f'update-{update:06d}.pt').is_file(), 'missing TFM snapshot')
            endpoint = directory/'fusion-evaluations'/f'update-{update:06d}'
            evaluation = runtime.read_json(endpoint/'evaluation.json')
            _require([row['scene_id'] for row in evaluation['per_scene']] == scene_ids,
                     'TFM endpoint evaluation has incomplete V population')
            _predictions(endpoint/'predictions', scene_ids)
    return dict(completed_updates=updates, completed_validation_scenes=len(scene_ids))


def _execute_stage(row, logs, write_state):
    """Keep child in the outer payload PGID; no detached grandchildren."""
    process = None
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise InterruptedError('SIGTERM: stop fixed chain without successors')
    signal.signal(signal.SIGTERM, terminated)
    try:
        with (logs/(row['name']+'.stdout.log')).open('xb') as stdout, \
                (logs/(row['name']+'.stderr.log')).open('xb') as stderr:
            process = subprocess.Popen(row['argv'], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, close_fds=True)  # inherits PGID; never setsid
            row.update(pid=process.pid, pgid=os.getpgrp())
            write_state()
            row['returncode'] = process.wait()
    finally:
        try:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                row['returncode'] = process.returncode
            row.update(ended_at=_utc(), elapsed_seconds=time.monotonic()-started)
            write_state()
        finally:
            signal.signal(signal.SIGTERM, previous)


def run(args):
    from association.training.tmtaff import _configuration
    from association.data.physical_units import SPEED_UNIT_POLICY
    _require(args.device == 'cuda:0' and args.cpu_threads == 4, 'A fixes cuda:0 and CPU4')
    _require(len(args.archive) == 3, 'A requires all three ordered Level1 archive parts')
    inputs = [*args.archive, args.manifest, args.alignment, args.source_system, args.reference090c_run]
    inputs = [path.resolve(strict=True) for path in inputs]
    output = args.output.resolve()
    for source in inputs:
        _require(output != source and source not in output.parents and output not in source.parents,
                 'output overlaps an immutable input')
    if output.exists():
        raise FileExistsError('output exists; no reuse, resume or retry')
    configs = {name: runtime.read_json(ROOT/'configs'/f'tmtaff-{name}-unit-draft.json') for name in ('tcm', 'tfm')}
    for name, config in configs.items():
        config['training'] = _configuration(config['training'], phase=name)
        expected = dict(TCM=2424 if name == 'tcm' else 0, TAM_FROZEN=0,
                        TFM_FROZEN=1000 if name == 'tfm' else 0, JOINT=0)
        _require(config['training']['updates'] == expected and config['training']['seed'] == 20
                 and config['training']['time_policy'] == 'advisory'
                 and config['data'].get('speed_unit_policy') == SPEED_UNIT_POLICY,
                 'frozen configuration no longer matches approved A')
    manifest = runtime.read_json(args.manifest)
    scene_ids = [str(sid) for sid in manifest['splits']['V_select']]
    _require(len(manifest['splits']['T_core']) == 2100 and len(scene_ids) == 300,
             'A requires the frozen T2100/V300 population')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    logs = output/'logs'; logs.mkdir()
    state = dict(status='RUNNING', started_at=_utc(), stages=[], test_opened=False,
        scope='approved unit-baseline A: TCM2424/TFM1000/fixed056 once', retries=0,
        device=args.device, cpu_threads=4, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        configurations=configs, argv=sys.argv, output=str(output))
    started = time.monotonic()
    def write_state():
        state['elapsed_seconds'] = time.monotonic()-started
        runtime.write_json(output/'run.json', state)
    common = ['--archive', *map(str, args.archive), '--manifest', str(args.manifest),
              '--alignment', str(args.alignment), '--device', args.device, '--cpu-threads', '4']
    fit = [sys.executable, str(ROOT/'tools/experiments/tmtaff_complete_fit.py')]
    commands = {}
    for name in ('tcm', 'tfm'):
        commands[name] = [*fit, '--phase', name, '--config', str(ROOT/'configs'/f'tmtaff-{name}-unit-draft.json'),
                         '--output', str(output/name), '--time-policy', 'advisory', *common]
    commands['tfm'] += ['--initialize-tcm-from', str(output/'tcm'), '--pairing-audit', '--save-predictions']
    commands['fixed056'] = [sys.executable, str(ROOT/'tools/experiments/tmtaff_fixed056_evaluate.py'),
        '--tfm-run', str(output/'tfm'), '--source-system', str(args.source_system),
        '--reference090c-run', str(args.reference090c_run), '--output', str(output/'fixed056'), *common]
    previous = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        raise InterruptedError('SIGTERM: stop fixed chain without successors')
    signal.signal(signal.SIGTERM, terminated)
    try:
        write_state()
        for name, argv in commands.items():
            row = dict(name=name, argv=argv, started_at=_utc(), status='RUNNING',
                       returncode=None, validation_status='NOT_STARTED')
            state['stages'].append(row); write_state()
            _execute_stage(row, logs, write_state)
            _require(row['returncode'] == 0, name+' child exited '+str(row['returncode']))
            row['validation_status'] = 'RUNNING'; write_state()
            row['verified'] = _validate_stage(name, output, configs, scene_ids)
            row.update(status='COMPLETED', validation_status='COMPLETED'); write_state()
        state['status'] = 'COMPLETED'
    except BaseException as error:
        state.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        if state['stages'] and state['stages'][-1]['status'] != 'COMPLETED':
            row = state['stages'][-1]
            row.update(status='FAILED', error_type=type(error).__name__, error=str(error))
            if row['validation_status'] == 'RUNNING':
                row['validation_status'] = 'FAILED'
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        state['ended_at'] = _utc(); write_state()
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, nargs='+', required=True)
    for name in ('manifest', 'alignment', 'source-system', 'reference090c-run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('cuda:0',), required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    # Children run in the frozen checkout, so resolve caller-relative inputs now.
    for name in ('manifest', 'alignment', 'source_system', 'reference090c_run', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    args.archive = [path.resolve() for path in args.archive]
    try:
        run(args)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'FAILED {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
