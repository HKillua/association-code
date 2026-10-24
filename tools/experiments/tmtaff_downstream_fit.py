"""092's conditional Level1 TAM/TFM entry; reuses the existing phase trainer.

Exit 75 means WAITING_DEPENDENCIES and consumes no attempt. An incomplete source
is inspected through run.json only: no checkpoint, output, or GPU is opened.
This entry never retries, resumes, changes a batch, or composes a full system.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
WAITING = 75


class WaitingDependencies(Exception):
    pass


def _json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _experiment_contract(phase):
    """Fixed public experiment profile; tiny tests replace this Python helper."""
    return dict(source=_json(ROOT / 'configs/tmtaff-tcm-level1.json'),
                target=_json(ROOT / f'configs/tmtaff-{phase}-level1.json'),
                train_scenes=2100, validation_scenes=300,
                train_fragments=155134, validation_fragments=24599)


def _canonical(config, phase):
    from association.training.tmtaff import _configuration
    return dict(config, training=_configuration(config['training'], phase=phase))


def check_source(source, manifest_path, alignment_path, config_path, phase, *, cpu_threads=None):
    """Bind a *finished* 088 endpoint to the frozen 092 phase and Level1 data."""
    source = Path(source)
    run_path = source / 'run.json'
    if not run_path.is_file():
        raise WaitingDependencies('088 has no run.json yet')
    run = _json(run_path)
    status = run.get('status')
    if status in ('FAILED', 'INTERRUPTED'):
        raise ValueError(f'088 is {status}; no automatic source substitution or retry')
    if status != 'COMPLETED':
        raise WaitingDependencies(f'088 status={status}; wait for training and final evaluation')
    _require(run.get('final_evaluation_status') == 'COMPLETED',
             '088 declared COMPLETED without completed final evaluation')

    # Torch/checkpoint access only follows the terminal-state boundary above.
    import torch
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    from association.data.archive import validate_manifest
    from association.models.fusion import PairFusionConfig
    from association.models.tmtaff import ModelConfig
    profile = _experiment_contract(phase)
    expected_source = _canonical(profile['source'], 'tcm')
    target = _canonical(_json(config_path), phase)
    _require(target == _canonical(profile['target'], phase), '092 target configuration differs')
    manifest, alignment = _json(manifest_path), _json(alignment_path)
    validate_manifest(manifest)
    _require(manifest.get('dataset_level') == 'Level1' and manifest.get('split_unit') == 'scene',
             '092 requires the fresh Level1 scene manifest')
    _require(not manifest['splits']['T_head'], '092 does not use T_head')
    for split, count in (('T_core', profile['train_scenes']), ('V_select', profile['validation_scenes'])):
        _require(len(manifest['splits'][split]) == count, f'{split} scene population differs')
    saved = torch.load(source / 'checkpoint.pt', map_location='cpu', weights_only=True)
    _require(all(bool(torch.isfinite(value).all()) for key, value in saved['states'][0].items()
                 if key in ('positions', 'heading_mean_std')
                 or key.startswith(('input_embedding.', 'tcm_encoder.', 'tcm_decoder.'))),
             '088 transferred completion contains nonfinite parameters')
    updates = expected_source['training']['updates']['TCM']
    _require(run.get('phase') == 'tcm' and run.get('test_opened') is False
             and run['cursor']['completed_updates'] == updates
             and saved.get('format_version') == 1 and saved.get('phase') == 'tcm'
             and saved.get('training_stages') == ['TCM'] and saved.get('selected') is None
             and saved['cursor'] == dict(stage=0, update=updates, completed_updates=updates),
             'source is not the completed fixed TCM-only endpoint')
    _require(saved['config'] == expected_source['training']
             and run['config'] == expected_source['training']
             and asdict(ModelConfig(**saved['model_config'])) == asdict(ModelConfig(**expected_source['model']))
             and asdict(PairFusionConfig(**saved['fusion_config'])) == asdict(PairFusionConfig(**expected_source['fusion'])),
             '088 source configuration differs')
    prep, summary = _json(source / 'preparation.json'), _json(source / 'summary.json')
    _require(_canonical(prep['configuration'], 'tcm') == expected_source,
             '088 preparation configuration differs')
    data = dict(expected_source['data'], length=expected_source['model']['length'])
    _require(prep['dataset']['data_config'] == data and prep['dataset'].get('test_opened') is False
             and prep['dataset']['training_scene_ids'] == manifest['splits']['T_core']
             and prep['dataset']['validation_scene_ids'] == manifest['splits']['V_select'],
             '088 input adapter/preparation differs or opened test data')
    contract, normalizer = saved['data_contract'], saved['normalizer']
    _require(saved.get('input_adapter') == 'native_time_v1' and saved.get('split_unit') == 'scene'
             and isinstance(contract, list) and len(contract) == 2
             and normalizer.get('training_splits') == ['T_core']
             and normalizer.get('training_scene_ids') == manifest['splits']['T_core']
             and summary['normalizer'] == normalizer, '088 source contract/normalizer differs')
    for rows, split, fragments in zip(contract, ('T_core', 'V_select'),
            (profile['train_fragments'], profile['validation_fragments'])):
        _require([row['scene_id'] for row in rows] == manifest['splits'][split],
                 f'088 saved {split} scene population/order differs')
        _require(sum(len(row['node_ids']) for row in rows) == fragments,
                 f'088 saved {split} fragment population differs')
        for row in rows:
            sid = row['scene_id']
            _require(row['split'] == split and row['component_id'] == manifest['components'][sid]
                     and row.get('split_unit') == 'scene' and row.get('input_adapter') == 'native_time_v1'
                     and len(row['grid']) == data['length']
                     and row.get('max_observations') == data['max_observations']
                     and row.get('time_unit_seconds') == data['time_unit_seconds']
                     and row['reference_alignment'].get('longitude_policy', 'linear') == data['reference_longitude_policy']
                     and alignment.get(sid) == {key: row['reference_alignment'][key]
                                               for key in ('time_offset_seconds', 'max_gap_seconds')},
                     f'088 scene semantics/alignment differs: {sid}')
    history, coverage = saved['history'], saved['coverage']['TCM']
    batch = expected_source['training']['batch_tracks']
    expected_batches = [min(batch, profile['train_fragments'] - start)
                        for start in range(0, profile['train_fragments'], batch)]
    _require(len(expected_batches) == updates
             and [(row['stage'], row['update']) for row in history] == [('TCM', i) for i in range(1, updates + 1)]
             and [row['track_presentations'] for row in history] == expected_batches
             and all(math.isfinite(row[key]) for row in history for key in ('loss', 'gradient_norm'))
             and coverage['track_presentations'] == profile['train_fragments']
             and coverage['unique_track_count'] == profile['train_fragments']
             and coverage['unique_scene_count'] == profile['train_scenes']
             and coverage['completed_epochs'] == 1,
             '088 successful ordered one-pass exposure differs')
    evaluation = _json(source / 'completion-evaluation.json')
    _require(evaluation['scene_count'] == profile['validation_scenes']
             and evaluation['fragment_count'] == profile['validation_fragments']
             and len(evaluation['per_fragment']) == profile['validation_fragments']
             and [row['scene_id'] for row in evaluation['per_scene']] == manifest['splits']['V_select']
             and [row['fragment_count'] for row in evaluation['per_scene']] == [len(row['node_ids']) for row in contract[1]],
             '088 completion evaluation omits validation population')
    return dict(source=str(source.resolve()), phase=phase, profile=profile, target=target,
                checkpoint=saved, validation_pairs=sum(row['pair_count'] for row in contract[1]))


def verify_endpoint(output, bound):
    """Check real frozen weights and whole phase outputs; no score-based gate."""
    import torch
    from association.utils import runtime
    output, source, phase = Path(output), bound['checkpoint'], bound['phase']
    saved = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
    run, summary = _json(output / 'run.json'), _json(output / 'summary.json')
    stage = 'TAM_FROZEN' if phase == 'tam' else 'TFM_FROZEN'
    updates = bound['target']['training']['updates'][stage]
    _require(run.get('status') == 'COMPLETED' and run.get('final_evaluation_status') == 'COMPLETED'
             and run.get('phase') == phase and saved.get('phase') == phase
             and run.get('test_opened') is False and summary.get('test_opened') is False
             and saved['training_stages'] == [stage]
             and saved['cursor']['completed_updates'] == updates
             and saved['config'] == bound['target']['training']
             and [(row['stage'], row['update']) for row in saved['history']] == [(stage, i) for i in range(1, updates + 1)],
             '092 endpoint phase/update/final-evaluation contract differs')
    _require(saved['normalizer'] == source['normalizer'] == summary['normalizer']
             and saved['data_contract'] == source['data_contract']
             and saved['initialization']['source_run'] == bound['source'],
             '092 source, full data contract or normalizer differs')
    frozen = []
    for key, value in saved['states'][0].items():
        if key in ('positions', 'heading_mean_std') or key.startswith(('input_embedding.', 'tcm_encoder.', 'tcm_decoder.')):
            _require(torch.equal(value, source['states'][0][key]), f'frozen TCM tensor changed: {key}')
            frozen.append(key)
    for state in saved['states']:
        _require(all(bool(torch.isfinite(value).all()) for value in state.values()), 'nonfinite endpoint parameter')
    coverage = saved['coverage'][stage]
    exposures = updates * bound['target']['training']['pair_batch_size']
    _require(coverage['pair_presentations'] == exposures
             and coverage['positive_pair_presentations'] == (exposures // 2 if phase == 'tam' else exposures),
             '092 actual pair exposures differ')
    for other, row in saved['coverage'].items():
        if other != stage:
            _require(row['track_presentations'] == row['pair_presentations'] == 0, 'inactive phase received training')
    rows = source['data_contract'][1]
    ids = [row['scene_id'] for row in rows]
    if phase == 'tam':
        evaluation = summary['evaluation']
        association = evaluation['association']
        _require(evaluation['physical_fusion_evaluated'] is False
                 and association['coverage']['all_scenes'] == len(ids)
                 and association['pair_count'] == bound['validation_pairs']
                 and [row['scene_id'] for row in association['per_scene']] == ids
                 and len(summary['validation_score_records']) == 1
                 and summary['validation_score_records'][0]['update'] == updates,
                 'TAM must score the fixed final full V without physical fusion')
        records = runtime.load_records(summary['endpoint_score_records'])
        _require([row['scene_id'] for row in records] == ids, 'TAM score population/order differs')
        for record, row in zip(records, rows):
            _require(record['node_ids'] == row['node_ids'] and len(record['pairs']) == row['pair_count']
                     and record['identities'].tolist() == row['identities']
                     and record['sources'].tolist() == row['sources'], 'TAM saved candidates differ')
    else:
        _require(summary['selection_performed'] is False and summary['fixed_final_update'] == updates
                 and run['margin_calibration'] is None
                 and [row['update'] for row in summary['fusion_evaluations']] == [0, updates],
                 'TFM requires only initial/fixed-final oracle endpoints without TAM calibration')
        for endpoint in summary['fusion_evaluations']:
            evaluation = _json(endpoint['evaluation'])
            _require(endpoint['status'] == 'COMPLETED' and evaluation['oracle_only'] is True
                     and evaluation['association'] is None and evaluation['selection'] is None
                     and 'predicted' not in evaluation['fusion']
                     and [row['scene_id'] for row in evaluation['per_scene']] == ids,
                     'TFM must evaluate full V oracle-only without untrained TAM')
            index_path = Path(endpoint['predictions'])
            index = _json(index_path)
            _require([row['scene_id'] for row in index['scenes']] == ids,
                     'TFM predictions omit validation population')
            for item, row in zip(index['scenes'], rows):
                metadata = _json(index_path.parent / item['metadata'])
                _require((index_path.parent / item['npz']).is_file()
                         and metadata['methods']['oracle']['target_count'] == len(row['target_ids']),
                         'TFM prediction artifacts omit targets or arrays')
    return dict(status='COMPLETED', phase=phase, source_run=bound['source'], source_phase='tcm',
        source_tcm_updates=source['cursor']['completed_updates'], completed_updates=updates,
        frozen_tcm_tensors=len(frozen), full_source_contract_equal=True, normalizer_equal=True,
        validation_scene_count=len(ids), validation_fragment_count=sum(len(row['node_ids']) for row in rows),
        validation_candidate_count=bound['validation_pairs'], pair_presentations=exposures,
        full_system_evaluated=False, test_opened=False,
        interpretation='phase qualification only; method performance and training sufficiency require review')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('tam', 'tfm'))
    for name in ('source-tcm', 'manifest', 'alignment', 'config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--check-source-only', action='store_true')
    parser.add_argument('--archive', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    output_owned = False
    try:
        if args.cpu_threads < 1:
            raise ValueError('--cpu-threads must be positive')
        if not args.check_source_only and (not args.archive or args.output is None or not args.device):
            raise ValueError('fit requires --archive, --output and explicit --device')
        bound = check_source(args.source_tcm, args.manifest, args.alignment, args.config, args.phase,
                             cpu_threads=args.cpu_threads)
        import torch
        torch.set_num_threads(args.cpu_threads)
        if args.check_source_only:
            print(json.dumps(dict(status='READY_DEPENDENCIES', phase=args.phase,
                source_run=bound['source'], validation_pairs=bound['validation_pairs'])), flush=True)
            return 0
        if args.output.exists():
            raise FileExistsError(f'Output already exists: {args.output}')
        from tools.experiments import tmtaff_complete_fit
        fit_args = ['--phase', args.phase, '--initialize-tcm-from', str(args.source_tcm),
            '--manifest', str(args.manifest), '--alignment', str(args.alignment),
            '--config', str(args.config), '--output', str(args.output), '--device', args.device,
            '--cpu-threads', str(args.cpu_threads), '--time-policy', 'advisory',
            '--archive', *map(str, args.archive)]
        if args.phase == 'tfm':
            fit_args.append('--save-predictions')
        output_owned = True
        code = tmtaff_complete_fit.main(fit_args)
        if code:
            return code
        from association.utils import runtime
        runtime.write_json(args.output / 'downstream-verification.json', verify_endpoint(args.output, bound))
        print(f'COMPLETED verified 092 {args.phase}: {args.output.resolve()}', flush=True)
        return 0
    except WaitingDependencies as error:
        print(f'WAITING_DEPENDENCIES: {error}', flush=True)
        return WAITING
    except (Exception, KeyboardInterrupt) as error:
        if output_owned and args.output.is_dir():
            from association.utils import runtime
            runtime.write_json(args.output / 'downstream-verification.json',
                dict(status='FAILED', phase=args.phase, failure_type=type(error).__name__,
                     failure=str(error), source_run=str(args.source_tcm.resolve())))
        print(f'BLOCKED_OR_FAILED {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
