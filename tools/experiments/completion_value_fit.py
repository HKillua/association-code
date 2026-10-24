"""093/094 frozen088 completion-value arms, with actual-sample audit.

Each arm is a single independent fresh TAM or TFM stage. It does not wait for
its partner, select a completion policy from V, retry or assemble a full system.
Exit75 preserves the 088 dependency boundary without opening active weights.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.experiments import tmtaff_downstream_fit as source_checks

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ('learned', 'state_hold')


def _experiment_contract(phase, policy):
    suffix = 'hold' if policy == 'state_hold' else 'learned'
    return dict(source=source_checks._json(ROOT / 'configs/tmtaff-tcm-level1.json'),
        target=source_checks._json(ROOT / f'configs/tmtaff-{phase}-{suffix}-level1.json'),
        train_scenes=2100, validation_scenes=300,
        train_fragments=155134, validation_fragments=24599)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same_tree(left, right):
    """Actual saved values, not a digest or tolerance-based acceptance."""
    import torch
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left, right))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(_same_tree(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_same_tree(a, b) for a, b in zip(left, right)))
    return left == right


def _sample_evidence(saved, initial, phase):
    """Check ordered successful draws against the saved semantic pair universe.

    This checks actual pair order/labels and count journals. It does not claim
    the old data_contract contains every input float or per-query truth mask.
    """
    population = saved['data_contract'][0]
    batch = saved['config']['pair_batch_size']
    total, positive, unique = 0, 0, set()
    for row in saved['history']:
        groups = row.get('actual_samples')
        _require(isinstance(groups, list) and groups, 'actual successful sample journal is missing')
        labels = []
        for group in groups:
            index = group.get('scene_index')
            _require(type(index) is int and 0 <= index < len(population), 'sample scene index invalid')
            scene = population[index]
            count = len(scene['node_ids'])
            indices, pairs, truth = (group.get(key) for key in ('pair_indices', 'pairs', 'labels'))
            _require(group.get('scene_id') == scene['scene_id']
                     and isinstance(indices, list) and isinstance(pairs, list) and isinstance(truth, list)
                     and len(indices) == len(pairs) == len(truth) and len(indices) > 0,
                     'sample groups omit ordered pair evidence')
            for pair_index, pair, label in zip(indices, pairs, truth):
                _require(isinstance(pair, list) and len(pair) == 2
                         and all(type(value) is int for value in pair)
                         and 0 <= pair[0] < pair[1] < count
                         and type(pair_index) is int and 0 <= pair_index < scene['pair_count'],
                         'sample pair or pair index invalid')
                left, right = pair
                expected_index = left * (2 * count - left - 1) // 2 + right - left - 1
                expected_label = int(scene['identities'][left] == scene['identities'][right])
                _require(pair_index == expected_index and label in (0, 1) and label == expected_label,
                         'sample actual endpoints/index/label disagree')
                unique.add((index, pair_index))
            labels.extend(truth)
        positives = sum(labels)
        expected_positives = batch if phase == 'tfm' else batch // 2
        counts = row.get('supervision_counts', {})
        _require(len(labels) == batch and positives == expected_positives
                 and counts.get('association_labels') == (batch if phase == 'tam' else 0)
                 and counts.get('positive_labels') == positives
                 and counts.get('negative_labels') == batch - positives,
                 'successful sample supervision counts differ')
        if phase == 'tfm':
            _require(counts.get('fusion_attributes') == row.get('fusion_valid_counts')
                     and len(counts['fusion_attributes']) == 4
                     and all(type(value) is int and 0 <= value <= batch * saved['model_config']['length']
                             for value in counts['fusion_attributes']),
                     'TFM actual supervision counts differ')
        else:
            _require(counts.get('fusion_attributes') == [0, 0, 0, 0],
                     'TAM unexpectedly reports fusion supervision')
        total += len(labels)
        positive += positives
    _require(initial['cursor']['completed_updates'] == 0 and not initial['history']
             and initial['optimizer'] is not None and not initial['optimizer']['state'],
             'paired initial snapshot is not a fresh optimizer at update0')
    _require(saved['optimizer'] is not None and saved['optimizer']['state'],
             'final successful optimizer state is missing')
    steps = [float(row['step']) for row in saved['optimizer']['state'].values() if 'step' in row]
    updates = saved['cursor']['completed_updates']
    _require(steps and all(math.isfinite(step) and 0 < step <= updates for step in steps)
             and max(steps) == updates, 'final optimizer step evidence differs')
    return dict(successful_updates=updates, pair_presentations=total,
                positive_pair_presentations=positive, negative_pair_presentations=total-positive,
                unique_sampled_pairs=len(unique),
                checked='ordered scene/pair indices/endpoints/labels and successful task-count journals',
                not_claimed='full input-float or per-query reference-mask replay from semantic data_contract')


def validate_completed_run(output, phase, policy, *, bound=None):
    """Read-only single-arm artifact validation, also reused by the paired table."""
    import torch
    from association.models.tmtaff import ModelConfig
    output = Path(output).resolve()
    _require(phase in ('tam', 'tfm') and policy in POLICIES, 'unknown completion-value arm')
    run = source_checks._json(output / 'run.json')
    preparation = source_checks._json(output / 'preparation.json')
    saved = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
    initial = torch.load(output / 'pairing-initial.pt', map_location='cpu', weights_only=True)
    if bound is None:
        source_path = Path(saved['initialization']['source_run']).resolve()
        source_run = source_checks._json(source_path / 'run.json')
        _require(source_run.get('status') == 'COMPLETED' and source_run.get('phase') == 'tcm'
                 and source_run.get('final_evaluation_status') == 'COMPLETED',
                 'source088 is not completed')
        source = torch.load(source_path / 'checkpoint.pt', map_location='cpu', weights_only=True)
        bound = dict(source=str(source_path), phase=phase, checkpoint=source,
            target=source_checks._canonical(preparation['configuration'], phase),
            validation_pairs=sum(row['pair_count'] for row in source['data_contract'][1]))
    report = source_checks.verify_endpoint(output, bound)
    expected_model = ModelConfig(**bound['target']['model'])
    actual_model = ModelConfig(**saved['model_config'])
    source_model = ModelConfig(**bound['checkpoint']['model_config'])
    _require(expected_model.completion_policy == actual_model.completion_policy == policy
             and asdict(actual_model) == asdict(expected_model)
             and asdict(replace(actual_model, completion_policy='learned')) == asdict(source_model),
             'completion policy is not the sole source-model change')
    _require(run.get('completion_policy') == policy and run.get('pairing_audit') is True
             and saved.get('completion_policy') == policy and saved.get('pairing_audit') is True
             and initial.get('completion_policy') == policy and initial.get('pairing_audit') is True
             and run['requested_model_config'] == saved['model_config']
             and source_checks._canonical(preparation['configuration'], phase) == bound['target']
             and preparation.get('phase') == phase,
             'run, preparation and snapshot policy/audit differ')
    for key in ('phase', 'training_stages', 'config', 'model_config', 'fusion_config',
                'normalizer', 'data_contract', 'initialization', 'effective_margin', 'margin_calibration'):
        _require(_same_tree(saved[key], initial[key]), 'initial/final binding differs: ' + key)
    frozen = ('input_embedding.', 'tcm_encoder.', 'tcm_decoder.')
    for key, tensor in initial['states'][0].items():
        if key in ('positions', 'heading_mean_std') or key.startswith(frozen):
            _require(torch.equal(tensor, bound['checkpoint']['states'][0][key]),
                     'paired initial completion differs from source088: ' + key)
    if phase == 'tfm':
        _require(_same_tree(saved['states'][0], initial['states'][0]),
                 'TFM must freeze every association-model parameter')
        changed = any(not torch.equal(tensor, initial['states'][1][key])
                      for key, tensor in saved['states'][1].items())
    else:
        _require(_same_tree(saved['states'][1:], initial['states'][1:]),
                 'TAM must freeze all fusion and objective parameters')
        changed = any(not torch.equal(tensor, initial['states'][0][key])
                      for key, tensor in saved['states'][0].items() if key.startswith('tam_encoder.'))
    _require(changed, 'requested downstream module has no actual parameter change')
    for snapshot in (initial, saved):
        _require(all(bool(torch.isfinite(value).all()) for state in snapshot['states'] for value in state.values()),
                 'nonfinite initial or final state')
        _require(all(key in snapshot['rng'] for key in ('torch', 'numpy', 'tracks', 'tfm', 'pairs', 'python', 'cuda')),
                 'paired RNG evidence incomplete')
    samples = _sample_evidence(saved, initial, phase)
    report.update(scope='post088-completion-value', completion_policy=policy, pairing_audit=True,
        initial_snapshot='pairing-initial.pt', source_completion_policy='learned',
        model_change='completion_policy only', actual_samples=samples, downstream_parameters_changed=True,
        initial_predictions_required_equal=False, test_opened=False,
        interpretation=('one-pass088-conditioned pilot; not full TMTAFF reproduction; '
                        'TAM has separately T-calibrated margin and V-selected threshold'))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('tam', 'tfm'))
    parser.add_argument('--completion-policy', required=True, choices=POLICIES)
    for name in ('source-tcm', 'manifest', 'alignment', 'config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--check-source-only', action='store_true')
    parser.add_argument('--archive', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'))
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    output_owned = False
    try:
        if args.cpu_threads < 1:
            raise ValueError('--cpu-threads must be positive')
        if not args.check_source_only and (not args.archive or args.output is None or not args.device):
            raise ValueError('fit requires --archive, --output and explicit --device')
        bound = source_checks.check_source(args.source_tcm, args.manifest, args.alignment,
            args.config, args.phase, cpu_threads=args.cpu_threads,
            profile=_experiment_contract(args.phase, args.completion_policy))
        _require(bound['target']['model'].get('completion_policy') == args.completion_policy,
                 'CLI and fixed configuration policy disagree')
        if args.check_source_only:
            print(json.dumps(dict(status='READY_DEPENDENCIES', phase=args.phase,
                completion_policy=args.completion_policy, source_run=bound['source'])), flush=True)
            return 0
        if args.output.exists():
            raise FileExistsError(f'Output already exists: {args.output}')
        from tools.experiments import tmtaff_complete_fit
        fit_args = ['--phase', args.phase, '--initialize-tcm-from', str(args.source_tcm),
            '--pairing-audit', '--manifest', str(args.manifest), '--alignment', str(args.alignment),
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
        report = validate_completed_run(args.output, args.phase, args.completion_policy, bound=bound)
        runtime.write_json(args.output / 'completion-value-verification.json', report)
        print(f'COMPLETED verified {args.phase}/{args.completion_policy}: {args.output.resolve()}', flush=True)
        return 0
    except source_checks.WaitingDependencies as error:
        print(f'WAITING_DEPENDENCIES: {error}', flush=True)
        return source_checks.WAITING
    except (Exception, KeyboardInterrupt) as error:
        if output_owned and args.output.is_dir():
            from association.utils import runtime
            runtime.write_json(args.output / 'completion-value-verification.json',
                dict(status='FAILED', phase=args.phase, completion_policy=args.completion_policy,
                     failure_type=type(error).__name__, failure=str(error),
                     source_run=str(args.source_tcm.resolve())))
        print(f'BLOCKED_OR_FAILED {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
