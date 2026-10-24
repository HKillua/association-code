"""090 complete cached CPU comparison; no archive, inference or training."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.frozen_position_adapter import exact
from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.sequence_position import REGIONS, WEIGHTINGS
from association.evaluation.support_gate_objectives import _artifact, _check_artifact, _structure
from association.models.frozen_position_adapter import FrozenPositionAdapter, initialize_from_teacher
from association.models.sequence_position import tangent_position
from association.training.frozen_position_adapter import load_checkpoint, source_configuration, _exact
from association.training.sequence_position import load_checkpoint as load_teacher
from association.utils.runtime import read_json, write_json, write_json_gzip
from tools.analysis.compare_position_benchmarks import ROOT, CONTROLS, INPUT_KEYS, _metadata, _pool, _sequence_sampling
from tools.analysis.compare_position_harm import _cache, _visible_opportunity, harm_statistics

NAMES = {'all_queries': 'adapter_all', 'anchor_unsupported': 'adapter_missing', 'full_finetune': 'finetune'}


def declared_manifest():
    return read_json(ROOT/'configs/data/mtad-level1-scene3000.json')


def adapter_metadata(path):
    from tools.experiments.frozen_position_adapter_fit import frozen_configuration
    path = Path(path).resolve()
    run, summary, fit = [read_json(path/name) for name in ('run.json', 'summary.json', 'fit/run.json')]
    config = frozen_configuration(run['configuration']); steps = config['training']['updates']
    manifest = declared_manifest()
    for record in (run, summary, fit):
        if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                or record.get('optimizer_updates') != steps or record.get('test_opened') is not False):
            raise ValueError('090 completed configuration/update chain mismatch')
    if (run.get('completed_endpoints') != ['initial', 'final'] or run.get('validation_selection') is not False
            or run.get('source_system') != summary.get('source_system')
            or run.get('paired_control') != summary.get('paired_control')
            or fit.get('training_scene_ids') != manifest['splits']['T_core'] or fit.get('final_checkpoint') != 'final.pt'):
        raise ValueError('090 complete T population/provenance mismatch')
    for label, step, filename in (('initial', 0, 'step-0.pt'), ('final', steps, 'final.pt')):
        endpoint = read_json(path/label/'summary.json')
        for record in (read_json(path/label/'run.json'), endpoint):
            saved = record.get('checkpoint', {})
            if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                    or record.get('source_system') != run['source_system'] or record.get('device') != 'cpu'
                    or record.get('optimizer_updates') != 0 or record.get('test_opened') is not False
                    or record.get('groups_recomputed') is not False or record.get('threshold_reselected') is not False
                    or record.get('initial') != (label == 'initial') or saved.get('step') != step
                    or saved.get('actual_updates') != step or saved.get('configured_final_updates') != steps
                    or Path(saved.get('path', '')).resolve() != path/'fit'/filename):
                raise ValueError('090 endpoint does not bind actual complete CPU checkpoint')
        if endpoint['scene_ids'] != manifest['splits']['V_select']:
            raise ValueError('090 full V population/order mismatch')
        components = {sid: str(cid) for cid, item in endpoint['components'].items() for sid in item['scene_ids']}
        if components != {sid: str(manifest['components'][sid]) for sid in endpoint['scene_ids']}:
            raise ValueError('090 component registry mismatch')
        for sub in ('predictions', 'system-predictions'):
            if {p.name for p in (path/label/sub).glob('*.npz')} != {f'states-{sid}.npz' for sid in endpoint['scene_ids']}:
                raise ValueError('090 complete endpoint cache inventory mismatch')
    return NAMES[config['model']['mode']], dict(path=path, configuration=config, method='learned_sequence',
        source_system=run['source_system'], scene_ids=endpoint['scene_ids'], components=components,
        summary=endpoint, paired_control=Path(run['paired_control']).resolve())


def verify_pairing(arms):
    reference = arms['linear']['path']
    old, old_rows, old_exposure = _sequence_sampling(reference/'fit')
    _, old_final = load_teacher(reference/'fit/final.pt')
    _, old_initial = load_teacher(reference/'fit/step-0.pt')
    results, initial_states = {}, {}
    samples = lambda rows: [{k: row[k] for k in ('step', 'mode', 'batch')} for row in rows]
    for name, arm in arms.items():
        if name == 'linear': continue
        fit = arm['path']/'fit'; new, rows, exposure = _sequence_sampling(fit)
        config = new['configuration']; mode = config['model']['mode']
        if (source_configuration(config) != old['configuration'] or samples(rows) != samples(old_rows)
                or exposure != old_exposure or new['training_scene_ids'] != old['training_scene_ids']
                or arm['paired_control'] != reference):
            raise ValueError('090 source configuration or actual successful batches/exposure differ')
        snapshots = []
        for filename, step in (('step-0.pt', 0), ('final.pt', config['training']['updates'])):
            model, snapshot = load_checkpoint(fit/filename)
            if (snapshot['configuration'] != config or snapshot['step'] != step
                    or Path(snapshot['source_teacher']['checkpoint']).resolve() != reference/'fit/final.pt'
                    or not _exact(snapshot['source_teacher']['model'], old_final['model'])):
                raise ValueError('090 checkpoint fails actual source-final binding')
            snapshots.append(snapshot)
        initial, final = snapshots
        if initial['numpy_rng'] != old_initial['numpy_rng']:
            raise ValueError('090 original sampler initial state differs')
        expected = FrozenPositionAdapter(**config['model'])
        initialize_from_teacher(expected, old_final['model'], branch_seed=config['training']['seed']+90000)
        if not _exact(initial['model'], expected.state_dict()):
            raise ValueError('090 source-final/zero-head/branch-seed initialization differs')
        initial_states[name] = initial['model']
        teacher_final = {k.removeprefix('teacher.'): v for k, v in final['model'].items() if k.startswith('teacher.')}
        unchanged = _exact(teacher_final, old_final['model'])
        if (mode != 'full_finetune') != unchanged:
            raise ValueError('090 teacher freezing/full-model update contract differs')
        if final['numpy_rng'] != old_final['numpy_rng']:
            raise ValueError('090 original sampler final state differs')
        results[name] = dict(source_final_parameters_exact=True, fresh_adam=True,
            actual_successful_batches_exact=True, sampling_exposure_exact=True, teacher_unchanged=unchanged,
            matched_updates=len(rows), matched_anchors=exposure['sampled_anchors'], exposure=exposure,
            parameter_count=new['parameter_count'], trainable_parameter_count=new['trainable_parameter_count'],
            source_updates=old_final['step'], added_updates=len(rows), route_exposure=new['route_exposure'])
    if 'adapter_all' in initial_states and 'adapter_missing' in initial_states and not _exact(
            initial_states['adapter_all'], initial_states['adapter_missing']):
        raise ValueError('090A/B initial tensors must be identical')
    return dict(status='PAIR_VERIFIED', matched_updates=len(old_rows), arms=results,
        ab_initial_parameters_exact=('adapter_all' in initial_states and 'adapter_missing' in initial_states))


def _fixed_artifact(artifact, original, label, *, predictions=False):
    if set(artifact) != set(original): raise ValueError('090 artifact schema mismatch/'+label)
    for key in original:
        if predictions or key not in ('group_states', 'aligned_prediction'):
            exact(artifact[key], original[key], label+'/'+key)


def _geometry(fragment, system, reference, source_system, mode, initial):
    states = _artifact(system, 'learned_sequence')['group_states']
    teacher = _artifact(source_system, 'learned_sequence')['group_states']
    support = reference['support'][reference['group_anchor_node_indices']]
    exact(fragment['group_anchor_support'], support, 'visible support cache')
    exact(fragment['group_source_teacher_states'], teacher, 'source final full-group predictions')
    exact(fragment['group_source_teacher_residual_m'], reference['group_tangent_residual_m'], 'source final residuals')
    exact(states[..., 2:], _artifact(system, 'mean_supported')['group_states'][..., 2:], 'Mean attributes')
    if mode == 'full_finetune':
        base = _artifact(system, 'fixed_prior095')['group_states']; delta = fragment['group_tangent_residual_m']
    else:
        base = teacher; delta = fragment['group_teacher_increment_m']
        allowed = np.ones_like(support) if mode == 'all_queries' else ~support
        exact(delta, fragment['group_adapter_raw_increment_m']*allowed[..., None], 'actual visible increment')
        if mode == 'anchor_unsupported': exact(states[support], teacher[support], 'B inside unchanged')
        if initial: exact(delta, np.zeros_like(delta), 'zero adapter initial output')
    flat_base = torch.from_numpy(base.reshape(-1, 4)); flat_delta = torch.from_numpy(delta.reshape(-1, 2))
    exact(states, tangent_position(flat_base, flat_delta).numpy().reshape(states.shape), 'cached tangent geometry')
    if initial: exact(states, teacher, 'initial equals teacher FINAL, all groups')


def posthoc_masked_artifact(fragment, candidate, teacher):
    """Apply one predeclared visible mask to full arrays, including duplicate rows."""
    support = fragment['support'][fragment['group_anchor_node_indices']]
    artifact = {k: v.copy() for k, v in candidate.items()}
    artifact['group_states'] = np.where(support[..., None], teacher['group_states'], candidate['group_states'])
    aligned = np.full_like(artifact['aligned_prediction'], np.nan)
    for row, gid in enumerate(artifact['aligned_group_indices']):
        if gid >= 0: aligned[row] = artifact['group_states'][gid]
    artifact['aligned_prediction'] = aligned
    return artifact


def scene_pair(arms, sid):
    reference, source = _cache(arms['linear']['path']/'final', sid)
    reference['group_eligible'] = _visible_opportunity(reference)
    base = _artifact(source, 'anchor'); _, _, regions, population = _structure(reference, base)
    data = {'region_'+name: values for name, values in regions.items()}
    for name in (*CONTROLS, 'fixed_prior095'):
        data[name+'_position_error'] = _check_artifact(reference, _artifact(source, name), base, name)
    teacher = _artifact(source, 'learned_sequence')
    data['linear_position_error'] = _check_artifact(reference, teacher, base, 'learned_sequence')
    for name, arm in arms.items():
        if name == 'linear': continue
        for label in ('initial', 'final'):
            fragment, system = _cache(arm['path']/label, sid)
            for key in INPUT_KEYS: exact(fragment[key], reference[key], 'identical input/'+key)
            for region in REGIONS: exact(fragment['region_'+region], data['region_'+region], 'region/'+region)
            for control in (*CONTROLS, 'fixed_prior095'):
                _fixed_artifact(_artifact(system, control), _artifact(source, control), control, predictions=True)
                _check_artifact(fragment, _artifact(system, control), base, control)
            current = _artifact(system, 'learned_sequence')
            _fixed_artifact(current, teacher, 'unchanged target/population/masks')
            error = _check_artifact(fragment, current, base, 'learned_sequence')
            _geometry(fragment, system, reference, source, arm['configuration']['model']['mode'], label == 'initial')
            exact(np.isfinite(error), np.isfinite(data['linear_position_error']), 'full common valid population')
            if label == 'initial':
                _fixed_artifact(current, teacher, 'initial teacher artifact', predictions=True)
            else:
                data[name+'_position_error'] = error
                if name == 'adapter_all':
                    masked = posthoc_masked_artifact(fragment, current, teacher)
                    t = len(base['target_ids']); valid = np.isfinite(error)
                    masked_error = attribute_errors(masked['aligned_prediction'][:t], base['aligned_truth'][:t])['position']
                    payload = dict(fragment, posthoc_mask_A_group_states=masked['group_states'],
                        posthoc_mask_A_position_error=np.where(valid, masked_error, np.nan))
                    data['posthoc_mask_A_position_error'] = _check_artifact(payload, masked, base, 'posthoc_mask_A')
    t = len(base['target_ids']); available = base['aligned_truth_valid'][:t, :, :2].all(-1)
    covered = available & base['aligned_prediction_valid'][:t, :, :2].all(-1)
    return dict(scene_id=sid, population=population, position_population=dict(available_target_points=int(available.sum()),
        covered_target_points=int(covered.sum()), missing_prediction_points=int((available & ~covered).sum()),
        available_target_trajectories=int(available.any(-1).sum()), covered_target_trajectories=int(covered.any(-1).sum()))), data


def summarize(rows, arrays, methods):
    result = _pool(rows, arrays, methods)
    result['relative_harm'] = {}
    baselines = [m for m in ('fixed_prior095', 'linear', 'finetune', 'adapter_all', 'posthoc_mask_A') if m in methods]
    for region in REGIONS:
        values = {m: np.concatenate([a[m+'_position_error'][a['region_'+region] &
            np.isfinite(a['linear_position_error'])] for a in arrays]) for m in methods}
        result['relative_harm'][region] = {name+'__minus__'+base: harm_statistics(values[name], values[base])
            for name in methods if name not in CONTROLS for base in baselines if name != base}
    return result


def screen(summary, scenes, method):
    table = summary['position_weightings']['all']; new, old = table[method], table['linear']; eps = 1e-6
    delta = {k: new['point_equal'][k]-old['point_equal'][k] for k in ('mae', 'rmse', 'p95', 'p99')}
    fraction = 1-new['point_equal']['mae']/old['point_equal']['mae']
    other = all(new[k]['mae'] <= old[k]['mae']+eps for k in WEIGHTINGS if k != 'point_equal')
    def improves(item):
        tab = item['position_weightings']['all']; return tab[method]['point_equal']['mae'] < tab['linear']['point_equal']['mae']-eps
    scene_count = sum(improves(item) for item in scenes)
    component_count = sum(improves(item) for item in summary['components'].values())
    rest = summary['remaining_components']
    rest_ok = bool(rest and rest['position_weightings']['all'][method]['point_equal']['mae'] <= rest['position_weightings']['all']['linear']['point_equal']['mae']+eps)
    original_pass = bool(fraction >= .03 and delta['rmse'] <= eps and delta['p99'] <= eps and other and rest_ok
        and scene_count > len(scenes)/2 and component_count > len(summary['components'])/2)
    budget_pass, budget_delta = None, None
    if 'finetune' in table and method != 'finetune':
        control = table['finetune']; budget_delta = {k: new['point_equal'][k]-control['point_equal'][k] for k in ('mae', 'rmse', 'p95', 'p99')}
        budget_rest = bool(rest and rest['position_weightings']['all'][method]['point_equal']['mae'] <= rest['position_weightings']['all']['finetune']['point_equal']['mae']+eps)
        budget_pass = bool(budget_delta['mae'] < -eps and budget_delta['rmse'] <= eps and budget_delta['p99'] <= eps
            and all(new[k]['mae'] <= control[k]['mae']+eps for k in WEIGHTINGS) and budget_rest)
    return dict(original072C_screen_passed=original_pass, relative_mae_reduction=fraction, required_reduction=.03,
        deltas_vs072C=delta, nonpoint_mae_nonworse=bool(other), outside_largest_component_nonworse=rest_ok,
        improved_scenes=scene_count, total_scenes=len(scenes), improved_components=component_count,
        total_components=len(summary['components']), added_budget_control_passed=budget_pass,
        deltas_vs_added_budget=budget_delta, predeclared_method_candidate=method == 'adapter_missing',
        eligible_for_replication=bool(method == 'adapter_missing' and original_pass and budget_pass is True),
        teacher_inside_exact_verified=method == 'adapter_missing', numerical_guard_m=eps,
        interpretation='Single-seed resource screen; missing C cannot exclude added-budget explanation. posthoc_mask_A is a fixed diagnostic, not a nominated candidate.')


def run(args):
    if args.cpu_threads != 4: raise ValueError('090 CPU comparison requires CPU4')
    torch.set_num_threads(4); output = Path(args.output).resolve()
    paths = [Path(v).resolve() for v in (getattr(args, k, None) for k in ('all_run', 'missing_run', 'finetune_run')) if v]
    if not paths: raise ValueError('090 comparison requires at least one completed new arm')
    reference = Path(args.linear_run).resolve()
    if output.exists(): raise FileExistsError('090 comparison output exists; no retry/overwrite')
    # A runner may put this read-only derived table under its own completed run.
    # Never allow it to overwrite/contain a source run or enter frozen056.
    protected_source = Path(read_json(reference/'run.json')['source_system']).resolve().parent
    if output == protected_source or output in protected_source.parents or protected_source in output.parents:
        raise ValueError('090 comparison overlaps immutable056')
    for path in [reference, *paths]:
        if output == path or output in path.parents: raise ValueError('090 comparison contains an input run')
    output.mkdir(parents=True)
    state = dict(status='RUNNING', completed_scenes=0, model_forwards=0, optimizer_updates=0, test_opened=False)
    write_json(output/'run.json', state)
    try:
        arms = {'linear': _metadata(reference, '072C_transformer4000')}
        for path in paths:
            name, arm = adapter_metadata(path)
            if name in arms: raise ValueError('090 repeated arm in comparison')
            arms[name] = arm
        first = arms['linear']
        if any(any(arm[k] != first[k] for k in ('source_system', 'scene_ids', 'components')) for arm in arms.values()):
            raise ValueError('090 complete source/scene/component population differs')
        pairing = verify_pairing(arms)
        methods = (*CONTROLS, 'fixed_prior095', *arms, *(('posthoc_mask_A',) if 'adapter_all' in arms else ()))
        rows, arrays = [], []
        for sid in first['scene_ids']:
            row, data = scene_pair(arms, sid); row['component_id'] = first['components'][sid]
            rows.append(row); arrays.append(data); state['completed_scenes'] = len(rows); write_json(output/'run.json', state)
        summary = summarize(rows, arrays, methods)
        for name, arm in arms.items():
            actual = summary['position_weightings']['all'][name]['point_equal']
            expected = arm['summary']['methods']['learned_sequence']['full_target_metrics']['all']['position']
            if actual['unit_count'] != expected['intersection_count'] or any(not np.isclose(actual[k], expected[k], rtol=1e-10, atol=1e-9) for k in ('mae', 'rmse')):
                raise ValueError('090 cache does not reproduce producer metrics/'+name)
        scene_rows = [dict(summarize([row], [data], methods), component_id=row['component_id']) for row, data in zip(rows, arrays)]
        components = {}
        for cid in sorted(set(first['components'].values())):
            chosen = [i for i, r in enumerate(rows) if r['component_id'] == cid]
            components[cid] = summarize([rows[i] for i in chosen], [arrays[i] for i in chosen], methods)
        largest = min(components, key=lambda cid: (-components[cid]['position_population']['covered_target_points'], cid))
        rest = [i for i, row in enumerate(rows) if row['component_id'] != largest]
        summary.update(status='COMPLETED', methods=list(methods), pairing=pairing, components=components,
            largest_component_id=largest, remaining_components=summarize([rows[i] for i in rest], [arrays[i] for i in rest], methods) if rest else None,
            source_system=first['source_system'], inputs={k: str(v['path']) for k, v in arms.items()},
            model_forwards=0, optimizer_updates=0, test_opened=False, old066_acceptance='REJECTED',
            initial_output_source_final_exact=True, protected_inside_teacher_exact=('adapter_missing' in arms),
            posthoc_mask_A=dict(present='adapter_all' in arms, input='visible anchor support only',
                whole_group_and_aligned_arrays_reconstructed=True, quantiles_from_full_error_arrays=True,
                candidate=False, model_forwards=0, training_updates=0),
            interpretation='Additional-budget frozen-teacher adaptation and full fine-tuning. Reused V/shared identities/single seed; no independent significance or convergence claim.')
        summary['scientific_screen'] = {name: screen(summary, scene_rows, name) for name in arms if name != 'linear'}
        write_json_gzip(output/'summary.json.gz', summary); write_json_gzip(output/'scene-summary.json.gz', scene_rows)
        table = ['# 090 强模型保持、缺失区修正与追加预算对照', '', '| 方法 | MAE m | RMSE m | P95 m | P99 m |', '|---|---:|---:|---:|---:|']
        for name in methods:
            metric = summary['position_weightings']['all'][name]['point_equal']
            table.append('| '+name+' | '+' | '.join(f'{metric[k]:.6f}' for k in ('mae', 'rmse', 'p95', 'p99'))+' |')
        table += ['', 'A/B 的 initial 是072C final；新增4000成功更新使用 fresh Adam。posthoc_mask_A只是固定的零训练控制。',
            '完整人口、三互斥支撑区、四权重、H/I、逐场/身份组件及最大组件外见 JSON；不按区域择优。']
        (output/'report.md').write_text('\n'.join(table)+'\n')
        state.update(status='COMPLETED', population=summary['population']); write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}'); write_json(output/'run.json', state); raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--linear-run', required=True)
    for name in ('all-run', 'missing-run', 'finetune-run'): parser.add_argument('--'+name)
    parser.add_argument('--output', required=True); parser.add_argument('--cpu-threads', type=int, default=4)
    result = run(parser.parse_args(argv)); print(result['status'], flush=True); return 0


if __name__ == '__main__': raise SystemExit(main())
