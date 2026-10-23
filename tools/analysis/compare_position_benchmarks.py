"""071 CPU closeout: actual-cache comparison of paired gates and sequence pilots.

No inference, training, archive access, threshold selection, or test access.
Every output group is checked before unique-target rows are evaluated.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.sequence_position import _weighted, REGIONS
from association.evaluation.support_gate_linear import _verify_route
from association.evaluation.support_gate_objectives import _read, _artifact, _structure, _check_artifact
from association.evaluation.support_gate_replication import SAMPLE_KEYS, verify_pairing
from association.evaluation.system_attribution import _same
from association.training.evidence_fusion import TRAIN_MODES
from association.training.support_gate import load_checkpoint as gate_checkpoint
from association.training.sequence_position import load_checkpoint as sequence_checkpoint
from association.utils.runtime import read_json, write_json, write_json_gzip

ROOT = Path(__file__).resolve().parents[2]
CONTROLS = ('anchor', 'mean_supported', 'attribute_hybrid', 'support_switch')
ARM_CONFIGS = {
    '071A_log': 'support-gate-replica-log-level1.json',
    '071B_linear': 'support-gate-replica-linear-level1.json',
    '071C_lstm': 'sequence-position-lstm-level1.json',
    '071D_transformer': 'sequence-position-transformer-level1.json',
    '070_linear': 'support-gate-linear-level1.json',
}
INPUT_KEYS = ('node_ids', 'identities', 'target_identity_ids', 'native_states', 'support',
              'nearest_seconds', 'exact_observation', 'query_seconds', 'learned_assoc_group_offsets',
              'learned_assoc_group_node_indices', 'group_anchor_node_indices')
METHODS = (*CONTROLS, 'fixed_prior095', *ARM_CONFIGS)


def _sequence_sampling(path):
    """Reconstruct exposure only from logged successful optimizer updates."""
    path = Path(path)
    run = read_json(path/'run.json')
    config = run['configuration']
    steps, size = config['training']['updates'], config['training']['batch_size']
    scenes = run.get('training_scene_ids', [])
    if (run.get('status') != 'COMPLETED' or run.get('optimizer_updates') != steps
            or run.get('test_opened') is not False or not scenes or len(set(scenes)) != len(scenes)):
        raise ValueError('sequence pairing requires completed actual updates and ordered training scenes')
    rows = [json.loads(line) for line in (path/'training.jsonl').read_text().splitlines()]
    if len(rows) != steps:
        raise ValueError('sequence pairing requires every actual successful batch')
    exposure = dict(successful_updates=0, sampled_anchors=0, valid_position_points=0, scenes={}, anchors={},
        modes={mode: dict(updates=0, anchors=0, valid_position_points=0) for mode in TRAIN_MODES})
    eligibility = dict(eligible_query_points=0, eligible_valid_position_points=0)
    for step, row in enumerate(rows, 1):
        mode = TRAIN_MODES[(step-1) % len(TRAIN_MODES)]
        if (row.get('step') != step or row.get('mode') != mode or len(row.get('batch', [])) != size
                or not np.isfinite(row.get('loss', np.nan)) or not np.isfinite(row.get('gradient_norm', np.nan))):
            raise ValueError('sequence successful update order/mode/batch is invalid')
        exposure['successful_updates'] += 1
        exposure['sampled_anchors'] += size
        exposure['modes'][mode]['updates'] += 1
        exposure['modes'][mode]['anchors'] += size
        for sample in row['batch']:
            if (set(sample) != SAMPLE_KEYS or sample['scene_id'] not in scenes
                    or type(sample['anchor_index']) is not int or sample['anchor_index'] < 0
                    or not isinstance(sample['node_id'], str) or not sample['node_id']
                    or not sample['members'] or sample['members'][0] != sample['anchor_index']
                    or len(set(sample['members'])) != len(sample['members'])
                    or any(type(x) is not int or x < 0 for x in sample['members'])):
                raise ValueError('sequence actual scene/anchor/member record is invalid')
            for key in (*eligibility, 'valid_position_points'):
                if type(sample[key]) is not int or not 0 <= sample[key] <= config['data']['length']:
                    raise ValueError('sequence sample count exceeds fixed query population')
            if sample['eligible_valid_position_points'] > min(sample['eligible_query_points'], sample['valid_position_points']):
                raise ValueError('sequence eligible valid points exceed sampled population')
            for key in eligibility:
                eligibility[key] += sample[key]
            exposure['valid_position_points'] += sample['valid_position_points']
            exposure['modes'][mode]['valid_position_points'] += sample['valid_position_points']
            for key, name in (('scenes', sample['scene_id']), ('anchors', sample['node_id'])):
                exposure[key][name] = exposure[key].get(name, 0) + 1
    # Gradient counts are properties of different backbones, not sampling.
    actual = run.get('exposure', {})
    if any(actual.get(key) != value for key, value in exposure.items()):
        raise ValueError('sequence successful batches do not reproduce recorded sampling exposure')
    return run, rows, dict(exposure, **eligibility)


def verify_sequence_sampling(lstm_fit, transformer_fit, gate_fit=None):
    a, ar, ae = _sequence_sampling(lstm_fit)
    b, br, be = _sequence_sampling(transformer_fit)
    ca, cb = deepcopy(a['configuration']), deepcopy(b['configuration'])
    if ca['model'].pop('backbone') != 'lstm' or cb['model'].pop('backbone') != 'transformer':
        raise ValueError('sequence pairing must compare the declared two backbones')
    if ca != cb or a['training_scene_ids'] != b['training_scene_ids']:
        raise ValueError('sequence configs may differ only in backbone; training scenes must agree')
    sampled = lambda rows: [{key: row[key] for key in ('step', 'mode', 'batch')} for row in rows]
    if sampled(ar) != sampled(br) or ae != be:
        raise ValueError('sequence actual successful batch pairing differs')
    cross_gate = None
    if gate_fit is not None:
        # Gate exposure has extra opportunity/gradient counters. Compare only
        # actual common batch fields; do not equate incompatible aggregates.
        gate_rows = [json.loads(line) for line in (Path(gate_fit)/'training.jsonl').read_text().splitlines()]
        cross_gate = sampled(ar) == sampled(gate_rows)
    return dict(status='PAIR_VERIFIED', actual_successful_batches_exact=True,
        ordered_training_scene_ids_exact=True, sampling_exposure_exact=True,
        initial_parameters_compared=False, matched_updates=len(ar), matched_anchors=ae['sampled_anchors'],
        seed=ca['training']['seed'], exposure=ae, gate_actual_batch_sequence_exact=cross_gate,
        interpretation='Backbones and initial parameters differ; only actual sampling is paired.')


def _metadata(path, arm):
    path = Path(path).resolve()
    run, summary, fit = [read_json(path/name) for name in ('run.json', 'summary.json', 'fit/run.json')]
    config = read_json(ROOT/'configs'/ARM_CONFIGS[arm])
    if config['training']['updates'] != 1000:
        raise ValueError('071 comparison requires declared1000-update budgets')
    for record in (run, summary, fit):
        if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                or record.get('optimizer_updates') != 1000 or record.get('test_opened') is not False):
            raise ValueError('all actual run/configuration/updates must describe completed frozen jobs')
    if (run.get('source_system') != summary.get('source_system')
            or run.get('validation_selection') is not False or fit.get('final_checkpoint') != 'final.pt'):
        raise ValueError('completed job provenance or fixed-endpoint selection differs')
    expected_endpoints = ['final'] if arm == '070_linear' else ['initial', 'final']
    if run.get('completed_endpoints') != expected_endpoints:
        raise ValueError('full required endpoint chain has not completed')
    sequence = arm in ('071C_lstm', '071D_transformer')
    loader = sequence_checkpoint if sequence else gate_checkpoint
    with torch.random.fork_rng(devices=[]):
        model, checkpoint = loader(path/'fit/final.pt', device='cpu')
    del model
    if (checkpoint['configuration'] != config or checkpoint['step'] != 1000
            or checkpoint['actual_updates'] != 1000):
        raise ValueError('actual final checkpoint/configuration/update mismatch')
    final = read_json(path/'final/summary.json')
    for record in (read_json(path/'final/run.json'), final):
        endpoint = record.get('checkpoint', {})
        if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                or record.get('source_system') != run['source_system'] or record.get('device') != 'cpu'
                or record.get('optimizer_updates') != 0 or record.get('test_opened') is not False
                or record.get('groups_recomputed') is not False or record.get('threshold_reselected') is not False
                or endpoint.get('step') != 1000 or endpoint.get('actual_updates') != 1000
                or endpoint.get('configured_final_updates') != 1000
                or Path(endpoint.get('path', '')).resolve() != path/'fit/final.pt'):
            raise ValueError('final CPU output does not bind the actual frozen endpoint')
    manifest = read_json(ROOT/'configs/data/mtad-level1-scene3000.json')
    if (final['scene_ids'] != manifest['splits']['V_select']
            or fit.get('training_scene_ids') != manifest['splits']['T_core']):
        raise ValueError('training and validation must retain complete declared Level1 populations')
    components = {sid: str(component) for component, group in final['components'].items() for sid in group['scene_ids']}
    if components != {sid: str(manifest['components'][sid]) for sid in final['scene_ids']}:
        raise ValueError('endpoint component registry differs from frozen manifest')
    for directory in ('predictions', 'system-predictions'):
        if {p.name for p in (path/'final'/directory).glob('*.npz')} != {f'states-{s}.npz' for s in final['scene_ids']}:
            raise ValueError('final cache inventory is incomplete or has extra scenes')
    return dict(path=path, method='learned_sequence' if sequence else 'learned_gate',
                source_system=run['source_system'], scene_ids=final['scene_ids'], components=components,
                summary=final)


def _arrays(directory, sid):
    return tuple(_read(Path(directory)/sub/f'states-{sid}.npz')
                 for sub in ('predictions', 'system-predictions'))


def scene_arrays(arms, sid, initial_directory):
    """Both schemas share actual full-scene artifacts, not summary claims."""
    reference, system = _arrays(arms['071A_log']['path']/'final', sid)
    base = _artifact(system, 'anchor')
    _, _, regions, population = _structure(reference, base)
    data = {'region_'+key: value for key, value in regions.items()}
    for control in CONTROLS:
        data[control+'_position_error'] = _check_artifact(reference, _artifact(system, control), base, control)
    initial, initial_system = _arrays(initial_directory, sid)
    _verify_route(initial_directory, sid, initial=True)
    prior = _artifact(initial_system, 'learned_gate')
    for key in INPUT_KEYS:
        _same(initial[key], reference[key], '071 shared initial input/'+key)
    data['fixed_prior095_position_error'] = _check_artifact(initial, prior, base, 'learned_gate')
    for name, arm in arms.items():
        fragment, saved = _arrays(arm['path']/'final', sid)
        for key in INPUT_KEYS:
            _same(fragment[key], reference[key], '071 cross-model physical input/'+key)
        for region in REGIONS:
            _same(fragment['region_'+region], data['region_'+region], '071 cross-model support stratum/'+region)
        for control in CONTROLS:
            current, fixed = _artifact(saved, control), _artifact(system, control)
            if set(current) != set(fixed):
                raise ValueError('ordinary complete artifact fields differ')
            for key in fixed:
                _same(current[key], fixed[key], '071 ordinary full artifact/'+control+'/'+key)
            _check_artifact(fragment, current, base, control)
        artifact = _artifact(saved, arm['method'])
        error = _check_artifact(fragment, artifact, base, arm['method'])
        _same(np.isfinite(error), np.isfinite(data['anchor_position_error']), '071 identical target masks')
        _same(artifact['group_states'][..., 2:], _artifact(saved, 'mean_supported')['group_states'][..., 2:],
              '071 all output-group speed/course remain Mean')
        if arm['method'] == 'learned_gate':
            _verify_route(arm['path']/'final', sid)
        else:
            fixed = _artifact(saved, 'fixed_prior095')
            if set(fixed) != set(prior):
                raise ValueError('sequence fixed-prior artifact schema differs')
            for key in prior:
                _same(fixed[key], prior[key], '071 sequence shared .95 control/'+key)
            _check_artifact(fragment, fixed, base, 'fixed_prior095')
        data[name+'_position_error'] = error
    t = len(base['target_ids'])
    available = base['aligned_truth_valid'][:t, :, :2].all(-1)
    covered = available & base['aligned_prediction_valid'][:t, :, :2].all(-1)
    position_population = dict(available_target_points=int(available.sum()),
        covered_target_points=int(covered.sum()), missing_prediction_points=int((available & ~covered).sum()),
        available_target_trajectories=int(available.any(-1).sum()),
        covered_target_trajectories=int(covered.any(-1).sum()))
    return dict(scene_id=str(sid), population=population, position_population=position_population), data


def _pool(rows, arrays):
    position_population = {key: sum(row['position_population'][key] for row in rows)
                           for key in rows[0]['position_population']}
    available = position_population['available_target_points']
    position_population['point_coverage'] = position_population['covered_target_points']/available if available else None
    return dict(scene_count=len(rows), scene_ids=[r['scene_id'] for r in rows], position_population=position_population,
        population={key: sum(row['population'][key] for row in rows) for key in rows[0]['population']},
        position_weightings=_weighted(rows, arrays, methods=METHODS))


def run(args):
    inputs = dict(zip(ARM_CONFIGS, (args.log_run071A, args.linear_run071B, args.lstm_run071C,
                                  args.transformer_run071D, args.source_070)))
    inputs = {key: Path(value).resolve() for key, value in inputs.items()}
    output = Path(args.output).resolve()
    if args.cpu_threads != 4:
        raise ValueError('071 CPU closeout fixes CPU threads to4')
    torch.set_num_threads(4)
    if output.exists():
        raise FileExistsError('output already exists; no retry/overwrite')
    sources = [Path(read_json(path/'run.json')['source_system']).resolve().parent for path in inputs.values()]
    for source in (*inputs.values(), *sources):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('output must be separate from all immutable sources')
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='RUNNING', model_forwards=0, optimizer_updates=0, test_opened=False,
                 checkpoint_read_device='cpu', completed_scenes=0)
    write_json(output/'run.json', state)
    try:
        pairing = verify_pairing(inputs['071A_log']/'fit', inputs['071B_linear']/'fit')
        if pairing.get('status') != 'PAIR_VERIFIED':
            raise ValueError('actual A/B pairing must succeed before cross-model comparison')
        sequence_pairing = verify_sequence_sampling(inputs['071C_lstm']/'fit', inputs['071D_transformer']/'fit',
                                                    inputs['071A_log']/'fit')
        arms = {name: _metadata(path, name) for name, path in inputs.items()}
        first = arms['071A_log']
        for arm in arms.values():
            if any(arm[key] != first[key] for key in ('source_system', 'scene_ids', 'components')):
                raise ValueError('all arms require the same source, complete V population and components')
        rows, arrays = [], []
        for sid in first['scene_ids']:
            row, data = scene_arrays(arms, sid, inputs['071A_log']/'initial')
            row['component_id'] = first['components'][sid]
            rows.append(row); arrays.append(data)
            state['completed_scenes'] = len(rows)
            write_json(output/'run.json', state)
        summary = _pool(rows, arrays)
        # Cross-check actual reconstructed complete-point metrics against every producer.
        for name, arm in arms.items():
            actual = summary['position_weightings']['all'][name]['point_equal']
            expected = arm['summary']['methods'][arm['method']]['full_target_metrics']['all']['position']
            if actual['unit_count'] != expected['intersection_count'] or any(
                    not np.isclose(actual[key], expected[key], rtol=1e-10, atol=1e-9) for key in ('mae', 'rmse')):
                raise ValueError('reconstructed state metrics differ from producer summary/'+name)
        components = {}
        for component in sorted(set(first['components'].values())):
            indices = [i for i, row in enumerate(rows) if row['component_id'] == component]
            components[component] = _pool([rows[i] for i in indices], [arrays[i] for i in indices])
        largest = min(components, key=lambda key: (-components[key]['position_weightings']['all']['anchor']['point_equal']['unit_count'], key))
        rest = [i for i, row in enumerate(rows) if row['component_id'] != largest]
        summary.update(status='COMPLETED', methods=list(METHODS), pairing=pairing,
            sequence_pairing=sequence_pairing, components=components,
            largest_component_id=largest, remaining_components=_pool([rows[i] for i in rest], [arrays[i] for i in rest]) if rest else None,
            source_system=first['source_system'], inputs={key: str(path) for key, path in inputs.items()},
            model_forwards=0, optimizer_updates=0, test_opened=False, old066_acceptance='REJECTED',
            interpretation='Paired second-seed loss intervention plus bounded generic temporal baselines; reused V, not held-out superiority.',
            comparability='Gate acts only in support opportunity; sequence has full-Q 2D residual within base-point hemisphere (not constrained to Anchor–Mean path). Shared physical inputs/outputs, different capacity.',
            speed_heading='Exactly ordinary Mean for every learned output group; no learned speed/course claim.')
        scene_rows = [dict(_pool([row], [data]), component_id=row['component_id']) for row, data in zip(rows, arrays)]
        write_json_gzip(output/'scene-summary.json.gz', scene_rows)
        write_json_gzip(output/'summary.json.gz', summary)
        table = ['# 071冻结完整人口位置比较', '', '| 方法 | MAE(m) | RMSE(m) | P95(m) | P99(m) | 有效点 |', '|---|---:|---:|---:|---:|---:|']
        for method in METHODS:
            item = summary['position_weightings']['all'][method]['point_equal']
            table.append('| '+method+' | '+' | '.join(f'{item[key]:.4f}' for key in ('mae', 'rmse', 'p95', 'p99'))+f" | {item['unit_count']} |")
        table += ['', '固定同一056分组、首代表、目标人口及有效掩码。四权重、支撑层、逐场与共享身份组件详见JSON。',
                  '这是复用V上的有界pilot；一般LSTM/Transformer控制不等于作者方法复现，也不证明已收敛。']
        (output/'report.md').write_text('\n'.join(table)+'\n')
        state.update(status='COMPLETED', paired=True, output_population=summary['population'])
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='FAILED', paired=False, error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('log-run071A', 'linear-run071B', 'lstm-run071C', 'transformer-run071D', 'source-070', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    result = run(args)
    print(result['status']+': '+str(Path(args.output).resolve()), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
