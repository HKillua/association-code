"""098 one-seed cached comparison: relation prior, generic set and budget control.

All groups and target masks are checked before scoring. No model forward,
training, archive access, validation selection or automatic promotion occurs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.frozen_position_adapter import exact
from association.evaluation.helper_sequence_position import helper_population, BUCKETS, POPULATIONS
from association.evaluation.sequence_position import REGIONS, WEIGHTINGS
from association.evaluation.support_gate_objectives import _artifact, _check_artifact
from association.models.query_relation import QueryRelationPosition, initialize_from_reference
from association.models.sequence_position import tangent_position
from association.training.query_relation import load_checkpoint, source_configuration
from association.training.sequence_position import load_checkpoint as load_teacher
from association.training.frozen_position_adapter import _exact
from association.utils.runtime import read_json, write_json, write_json_gzip
from tools.analysis import compare_frozen_position_adapter as budget_compare
from tools.analysis.compare_helper_sequence import helper_cohorts
from tools.analysis.compare_position_benchmarks import (
    ROOT, CONTROLS, INPUT_KEYS, _metadata, _pool, _sequence_sampling,
)
from tools.analysis.compare_position_harm import _cache, harm_statistics

SCOPE = '098-query-relation-cached-comparison'
METHODS = (*CONTROLS, 'fixed_prior095', 'linear', 'finetune', 'generic', 'relation', 'relation_off')
GROUP_COHORTS = ('any_multi_query', 'no_multi_query')
EPS_M = 1e-6


def declared_manifest():
    return read_json(ROOT/'configs/data/mtad-level1-scene3000.json')


def frozen_configuration(config):
    # Avoid a runner/comparison import cycle and retain a narrow tiny-fixture seam.
    from tools.experiments.query_relation_fit import frozen_configuration as check
    return check(config)


def relation_metadata(path):
    path = Path(path).resolve()
    run, summary, fit = [read_json(path/name) for name in ('run.json', 'summary.json', 'fit/run.json')]
    config = frozen_configuration(run['configuration'])
    mode, steps = config['model']['mode'], config['training']['updates']
    labels = ['initial', 'final'] + (['relation_off'] if mode == 'relation' else [])
    for record in (run, summary, fit):
        if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                or record.get('optimizer_updates') != steps or record.get('test_opened') is not False):
            raise ValueError('098 completed configuration/update chain mismatch')
    if (run.get('completed_endpoints') != labels or run.get('validation_selection') is not False
            or summary.get('source_system') != run.get('source_system')
            or summary.get('paired_control') != run.get('paired_control')
            or summary.get('budget_control') != run.get('budget_control')
            or not run.get('paired_control') or not run.get('budget_control')
            or fit.get('final_checkpoint') != 'final.pt'):
        raise ValueError('098 complete endpoint/source/budget provenance mismatch')
    manifest = declared_manifest()
    if fit.get('training_scene_ids') != manifest['splits']['T_core']:
        raise ValueError('098 complete training population differs')
    endpoints = {}
    for label in labels:
        step = 0 if label == 'initial' else steps
        filename = 'step-0.pt' if label == 'initial' else 'final.pt'
        enabled = mode == 'relation' and label != 'relation_off'
        endpoint = read_json(path/label/'summary.json')
        for record in (read_json(path/label/'run.json'), endpoint):
            provenance = record.get('checkpoint', {})
            if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                    or record.get('source_system') != run['source_system'] or record.get('device') != 'cpu'
                    or record.get('optimizer_updates') != 0 or record.get('test_opened') is not False
                    or record.get('groups_recomputed') is not False or record.get('threshold_reselected') is not False
                    or record.get('initial') != (label == 'initial')
                    or record.get('relation_enabled') is not enabled
                    or provenance.get('step') != step or provenance.get('actual_updates') != step
                    or provenance.get('configured_final_updates') != steps
                    or Path(provenance.get('path', '')).resolve() != path/'fit'/filename):
                raise ValueError('098 endpoint does not bind its actual checkpoint/bias state')
        if endpoint['scene_ids'] != manifest['splits']['V_select']:
            raise ValueError('098 full V population/order mismatch')
        components = {sid: str(cid) for cid, item in endpoint['components'].items() for sid in item['scene_ids']}
        if components != {sid: str(manifest['components'][sid]) for sid in endpoint['scene_ids']}:
            raise ValueError('098 component registry mismatch')
        for directory in ('predictions', 'system-predictions'):
            if {p.name for p in (path/label/directory).glob('*.npz')} != {f'states-{sid}.npz' for sid in endpoint['scene_ids']}:
                raise ValueError('098 endpoint cache inventory mismatch')
        endpoints[label] = endpoint
    return mode, dict(path=path, configuration=config, method='learned_sequence',
        source_system=run['source_system'], scene_ids=endpoint['scene_ids'], components=components,
        summary=endpoints['final'], endpoints=endpoints,
        paired_control=Path(run['paired_control']).resolve(), budget_control=Path(run['budget_control']).resolve())


def verify_pairing(arms):
    """Rebuild actual sampling and source-final initialization for all three fits."""
    baseline = budget_compare.verify_pairing({k: arms[k] for k in ('linear', 'finetune')})
    reference = arms['linear']['path']
    old, old_rows, old_exposure = _sequence_sampling(reference/'fit')
    with torch.random.fork_rng(devices=[]):
        _, original_initial = load_teacher(reference/'fit/step-0.pt')
        _, original_final = load_teacher(reference/'fit/final.pt')
    samples = lambda rows: [{k: r[k] for k in ('step', 'mode', 'batch')} for r in rows]
    initial_states, results = {}, {}
    for name in ('generic', 'relation'):
        arm = arms[name]
        run, rows, exposure = _sequence_sampling(arm['path']/'fit')
        config = run['configuration']
        if (source_configuration(config) != old['configuration'] or samples(rows) != samples(old_rows)
                or exposure != old_exposure or run['training_scene_ids'] != old['training_scene_ids']
                or arm['paired_control'] != reference or arm['budget_control'] != arms['finetune']['path']):
            raise ValueError('098 actual source/budget, successful batches or exposure differ')
        snapshots = []
        for filename, step in (('step-0.pt', 0), ('final.pt', config['training']['updates'])):
            with torch.random.fork_rng(devices=[]):
                model, saved = load_checkpoint(arm['path']/'fit'/filename, device='cpu')
            source = saved['source_teacher']
            if (saved['configuration'] != config or saved['step'] != step
                    or Path(source['checkpoint']).resolve() != reference/'fit/final.pt'
                    or source['configuration'] != old['configuration'] or source['step'] != original_final['step']
                    or not _exact(source['model'], original_final['model'])):
                raise ValueError('098 checkpoint fails actual source-final binding')
            count = sum(p.numel() for p in model.parameters())
            if (run.get('parameter_count') != count or run.get('trainable_parameter_count') != count
                    or not all(p.requires_grad for p in model.parameters())):
                raise ValueError('098 all original and branch parameters must be trainable')
            snapshots.append(saved)
        initial, final = snapshots
        with torch.random.fork_rng(devices=[]):
            expected = QueryRelationPosition(**config['model'])
            initialize_from_reference(expected, original_final['model'], branch_seed=config['training']['seed']+98000)
        if (not _exact(initial['model'], expected.state_dict())
                or initial['numpy_rng'] != original_initial['numpy_rng']
                or final['numpy_rng'] != original_final['numpy_rng']):
            raise ValueError('098 source-final/branch initialization or sampler RNG differs')
        initial_states[name] = initial['model']
        results[name] = dict(actual_successful_batches_exact=True, sampling_exposure_exact=True,
            source_final_parameters_exact=True, initial_sampler_rng_exact=True, final_sampler_rng_exact=True,
            matched_updates=len(rows), parameter_count=count, trainable_parameter_count=count,
            helper_support_exposure=run.get('helper_support_exposure'),
            source_updates=original_final['step'], added_updates=len(rows),
            final_parameters_changed=not _exact(initial['model'], final['model']))
    if not _exact(initial_states['generic'], initial_states['relation']):
        raise ValueError('098 generic/relation initial tensors must be identical')
    if results['generic']['parameter_count'] != results['relation']['parameter_count']:
        raise ValueError('098 generic/relation parameter counts differ')
    return dict(status='PAIR_VERIFIED', seed=old['configuration']['training']['seed'],
        matched_updates=len(old_rows), matched_anchors=old_exposure['sampled_anchors'],
        budget_control=baseline, arms=results, generic_relation_initial_parameters_exact=True,
        generic_relation_parameter_count_equal=True, equal_compute_claimed=False,
        optimizer='fresh Adam for the added stage; original successful samples replayed')


def _geometry(fragment, artifact, source_system):
    states = artifact['group_states']
    base = _artifact(source_system, 'fixed_prior095')['group_states']
    residual = fragment['group_tangent_residual_m']
    exact(fragment['group_source_teacher_states'], _artifact(source_system, 'learned_sequence')['group_states'],
        '098 source final group predictions')
    exact(states[..., 2:], _artifact(source_system, 'mean_supported')['group_states'][..., 2:], '098 Mean attributes')
    rebuilt = tangent_position(torch.from_numpy(base.reshape(-1, 4)),
        torch.from_numpy(residual.reshape(-1, 2))).numpy().reshape(states.shape)
    exact(states, rebuilt, '098 cached tangent geometry')


def scene_pair(arms, sid):
    row, data = budget_compare.scene_pair({k: arms[k] for k in ('linear', 'finetune')}, sid)
    reference, original_system = _cache(arms['linear']['path']/'final', sid)
    original = _artifact(original_system, 'learned_sequence')
    base = _artifact(original_system, 'anchor')
    counts_reference, off_states, relation_states, population = None, None, None, None
    for name in ('generic', 'relation'):
        labels = ('initial', 'final') + (('relation_off',) if name == 'relation' else ())
        for label in labels:
            fragment, system = _cache(arms[name]['path']/label, sid)
            for key in INPUT_KEYS:
                exact(fragment[key], reference[key], '098 identical input/'+key)
            for region in REGIONS:
                exact(fragment['region_'+region], data['region_'+region], '098 region/'+region)
            for control in (*CONTROLS, 'fixed_prior095'):
                budget_compare._fixed_artifact(_artifact(system, control), _artifact(original_system, control),
                    '098 '+control, predictions=True)
                _check_artifact(fragment, _artifact(system, control), base, control)
            artifact = _artifact(system, 'learned_sequence')
            budget_compare._fixed_artifact(artifact, original, '098 fixed target/population/masks')
            errors = _check_artifact(fragment, artifact, base, 'learned_sequence')
            exact(np.isfinite(errors), np.isfinite(data['linear_position_error']), '098 full common valid population')
            _geometry(fragment, artifact, original_system)
            current_population = helper_population(fragment, system)
            exact(fragment['group_any_multi_helper'], (fragment['group_supported_helper_count'] >= 2).any(-1),
                '098 group-level multi-helper population')
            if population is None:
                population = current_population
                counts_reference = {key: fragment[key] for key in
                    ('group_supported_helper_count', 'group_helper_offsets', 'group_helper_node_indices')}
                data.update({'helper_cohort_'+k: v for k, v in helper_cohorts(fragment, system).items()})
                counts = fragment['group_supported_helper_count']
                group_multi = (counts >= 2).any(-1)
                selected = base['aligned_group_indices'][:len(base['target_ids'])]
                exists = selected >= 0
                target_multi = np.zeros(len(selected), bool)
                target_multi[exists] = group_multi[selected[exists]]
                shape = data['region_all'].shape
                data['group_cohort_any_multi_query'] = np.broadcast_to((exists & target_multi)[:, None], shape)
                data['group_cohort_no_multi_query'] = np.broadcast_to((exists & ~target_multi)[:, None], shape)
                row['group_relation_population'] = dict(all_output_groups=int(len(group_multi)),
                    groups_with_any_multi_query=int(group_multi.sum()), groups_with_no_multi_query=int((~group_multi).sum()))
            else:
                if current_population != population:
                    raise ValueError('098 helper population changes across endpoints/arms')
                for key, expected in counts_reference.items():
                    exact(fragment[key], expected, '098 helper membership/'+key)
            if label == 'initial':
                budget_compare._fixed_artifact(artifact, original, '098 initial equals teacher FINAL', predictions=True)
            else:
                method = 'relation_off' if label == 'relation_off' else name
                data[method+'_position_error'] = errors
                if method == 'relation': relation_states = artifact['group_states']
                if method == 'relation_off': off_states = artifact['group_states']
    no_multi = ~(counts_reference['group_supported_helper_count'] >= 2).any(-1)
    exact(relation_states[no_multi], off_states[no_multi], '098 no-multi group fixed-weight bias switch')
    row['helper_population'] = population
    row['no_multi_prediction_exact'] = True
    return row, data


def summarize(rows, arrays, methods=METHODS):
    result = _pool(rows, arrays, methods)
    result['helper_cohorts'] = {
        name: _pool(rows, [dict(a, region_all=a['helper_cohort_'+name]) for a in arrays], methods)['position_weightings']['all']
        for name in BUCKETS}
    result['group_cohorts'] = {
        name: _pool(rows, [dict(a, region_all=a['group_cohort_'+name]) for a in arrays], methods)['position_weightings']['all']
        for name in GROUP_COHORTS}
    result['group_relation_population'] = {key: sum(row['group_relation_population'][key] for row in rows)
        for key in rows[0]['group_relation_population']}
    result['no_multi_prediction_exact'] = all(row['no_multi_prediction_exact'] for row in rows)
    result['relative_harm'] = {}
    for region in REGIONS:
        values = {name: np.concatenate([a[name+'_position_error'][a['region_'+region]
            & np.isfinite(a['linear_position_error'])] for a in arrays]) for name in methods}
        result['relative_harm'][region] = {new+'__minus__'+old: harm_statistics(values[new], values[old])
            for new, old in (('generic', 'finetune'), ('relation', 'finetune'),
                            ('relation', 'generic'), ('relation', 'relation_off'))}
    return result


def scientific_screen(summary, scenes):
    """Frozen numerical gate; one seed alone never establishes a contribution."""
    table = summary['position_weightings']['all']
    new = table['relation']
    rest = summary.get('remaining_components')

    def versus(control, reduction, *, majority=False):
        old = table[control]
        fraction = (1-new['point_equal']['mae']/old['point_equal']['mae']
            if old['point_equal']['mae'] > 0 else 0.)
        deltas = {k: new['point_equal'][k]-old['point_equal'][k] for k in ('mae', 'rmse', 'p95', 'p99')}
        weights = all(new[k]['mae'] <= old[k]['mae']+EPS_M for k in WEIGHTINGS)
        outside = bool(rest and rest['position_weightings']['all']['relation']['point_equal']['mae']
            <= rest['position_weightings']['all'][control]['point_equal']['mae']+EPS_M)
        improves = lambda item: (item['position_weightings']['all']['relation']['point_equal']['mae']
            < item['position_weightings']['all'][control]['point_equal']['mae']-EPS_M)
        scenes_better = sum(improves(item) for item in scenes)
        components_better = sum(improves(item) for item in summary['components'].values())
        passed = bool(fraction >= reduction and deltas['mae'] < -EPS_M
            and deltas['rmse'] <= EPS_M and deltas['p99'] <= EPS_M and weights and outside)
        if majority:
            passed = bool(passed and scenes_better > len(scenes)/2
                and components_better > len(summary['components'])/2)
        return dict(passed=passed, relative_mae_reduction=fraction, required_reduction=reduction,
            deltas=deltas, all_mae_weightings_nonworse=bool(weights), outside_largest_component_nonworse=outside,
            improved_scenes=scenes_better, total_scenes=len(scenes), improved_components=components_better,
            total_components=len(summary['components']))

    budget = versus('finetune', .03, majority=True)
    generic = versus('generic', .01)
    off = table['relation_off']['point_equal']
    mechanism_delta = new['point_equal']['mae']-off['mae']
    multi_points = summary['group_cohorts']['any_multi_query']['relation']['point_equal']['unit_count']
    mechanism_pass = bool(mechanism_delta < -EPS_M and multi_points > 0
        and summary.get('no_multi_prediction_exact') is True)
    passed = bool(budget['passed'] and generic['passed'] and mechanism_pass)
    generic_better = table['generic']['point_equal']['mae'] < table['finetune']['point_equal']['mae']-EPS_M
    return dict(candidate='relation', seed_screen_passed=passed, versus_budget=budget, versus_generic=generic,
        fixed_weight_bias_intervention=dict(passed=mechanism_pass, delta_mae_m=mechanism_delta,
            groups_with_multi_query_valid_points=int(multi_points),
            no_multi_prediction_exact=summary.get('no_multi_prediction_exact'), checkpoint_selection=False),
        generic_point_mae_better_than_budget=bool(generic_better),
        outcome=('PASS_THIS_SEED_ONLY' if passed else
            ('GENERIC_BASELINE_SIGNAL_ONLY' if generic_better and not generic['passed'] else 'CURRENT_RELATION_RECIPE_NOT_ACCEPTED')),
        required_seeds=[20261023, 20261021], default_promoted=False, test_opened=False, numerical_guard_m=EPS_M,
        interpretation='Both preregistered seeds must pass. Generic-set gains alone are a task-baseline result; this screen does not establish early-pooling causality or held-out generalization.')


def run(args):
    if args.cpu_threads != 4:
        raise ValueError('098 cached comparison requires CPU4')
    torch.set_num_threads(4)
    paths = {key: Path(getattr(args, key+'_run')).resolve() for key in ('linear', 'budget', 'generic', 'relation')}
    if len(set(paths.values())) != len(paths):
        raise ValueError('098 comparison requires distinct declared runs')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('098 comparison exists; no overwrite/retry')
    for path in paths.values():
        source = Path(read_json(path/'run.json')['source_system']).resolve().parent
        for protected in (path, source):
            if output == protected or output in protected.parents or protected in output.parents:
                raise ValueError('098 comparison output overlaps an immutable source')
    output.mkdir(parents=True)
    state = dict(scope=SCOPE, status='RUNNING', model_forwards=0, optimizer_updates=0,
        test_opened=False, completed_scenes=0, default_promoted=False)
    write_json(output/'run.json', state)
    try:
        generic_name, generic = relation_metadata(paths['generic'])
        relation_name, relation = relation_metadata(paths['relation'])
        if (generic_name, relation_name) != ('generic', 'relation'):
            raise ValueError('098 requires exactly generic and relation arms')
        seed = generic['configuration']['training']['seed']
        if seed not in (20261023, 20261021) or relation['configuration']['training']['seed'] != seed:
            raise ValueError('098 mixed or undeclared seed comparison')
        profile = '090' if seed == 20261023 else '097'
        original_name = '072C_transformer4000' if seed == 20261023 else '073C_transformer4000_seed21'
        budget_name, budget = budget_compare.adapter_metadata(paths['budget'], profile=profile)
        if budget_name != 'finetune':
            raise ValueError('098 strong budget control must be full-model finetune')
        arms = dict(linear=_metadata(paths['linear'], original_name), finetune=budget,
                    generic=generic, relation=relation)
        first = arms['linear']
        if any(arm[k] != first[k] for arm in arms.values() for k in ('source_system', 'scene_ids', 'components')):
            raise ValueError('098 full source/population/component mismatch')
        pairing = verify_pairing(arms)
        rows, arrays = [], []
        for sid in first['scene_ids']:
            row, data = scene_pair(arms, sid)
            row['component_id'] = first['components'][sid]
            rows.append(row); arrays.append(data)
            state['completed_scenes'] = len(rows)
            write_json(output/'run.json', state)
        summary = summarize(rows, arrays)
        producers = {name: arm['summary'] for name, arm in arms.items()}
        producers['relation_off'] = relation['endpoints']['relation_off']
        for name, producer in producers.items():
            actual = summary['position_weightings']['all'][name]['point_equal']
            expected = producer['methods']['learned_sequence']['full_target_metrics']['all']['position']
            if actual['unit_count'] != expected['intersection_count'] or any(
                    not np.isclose(actual[k], expected[k], rtol=1e-10, atol=1e-9) for k in ('mae', 'rmse')):
                raise ValueError('098 cache does not reproduce producer metrics/'+name)
        scene_rows = [dict(summarize([row], [data]), component_id=row['component_id'])
            for row, data in zip(rows, arrays)]
        components = {}
        for cid in sorted(set(first['components'].values())):
            ids = [i for i, row in enumerate(rows) if row['component_id'] == cid]
            components[cid] = summarize([rows[i] for i in ids], [arrays[i] for i in ids])
        largest = min(components, key=lambda cid: (-components[cid]['position_population']['covered_target_points'], cid))
        rest = [i for i, row in enumerate(rows) if row['component_id'] != largest]
        populations = {name: dict(total=sum(row['helper_population'][name]['total'] for row in rows),
            counts={bucket: sum(row['helper_population'][name]['counts'][bucket] for row in rows) for bucket in BUCKETS})
            for name in POPULATIONS}
        summary.update(scope=SCOPE, status='COMPLETED', seed=seed, pairing=pairing,
            methods=list(METHODS), components=components, largest_component_id=largest,
            remaining_components=summarize([rows[i] for i in rest], [arrays[i] for i in rest]) if rest else None,
            helper_population=populations, source_system=first['source_system'],
            inputs={k: str(v['path']) for k, v in arms.items()}, model_forwards=0, optimizer_updates=0,
            test_opened=False, default_promoted=False, old066_acceptance='REJECTED',
            initial_output_source_final_exact=True,
            interpretation='One declared seed, source-final warm start plus fresh Adam replay. Reused V/shared identities; not independent generalization, not continuous-optimizer convergence. Speed/heading remain Mean.')
        summary['scientific_screen'] = scientific_screen(summary, scene_rows)
        write_json_gzip(output/'summary.json.gz', summary)
        write_json_gzip(output/'scene-summary.json.gz', scene_rows)
        lines = ['# 098 同预算位置融合比较：seed '+str(seed), '',
            '| 方法 | MAE m | RMSE m | P95 m | P99 m |', '|---|---:|---:|---:|---:|']
        for name in METHODS:
            metric = summary['position_weightings']['all'][name]['point_equal']
            lines.append('| '+name+' | '+' | '.join(f'{metric[k]:.6f}' for k in ('mae', 'rmse', 'p95', 'p99'))+' |')
        lines += ['', '本 seed 科学门槛：'+summary['scientific_screen']['outcome']+'。两个预定 seed 都通过才形成候选证据，不自动晋升默认或开放测试。',
            'relation_off 使用 relation 的同一 final 权重关闭固定关系偏置，零新增训练，不用于选模。',
            '完整人口、四种权重、尾部、逐场／组件、最大组件外、helper 数与整组可交互人口见 JSON。',
            '通用 set 单独改善只能记为更强任务基线；本轮不证明早聚合是既有方法的根因。']
        (output/'report.md').write_text('\n'.join(lines)+'\n')
        state.update(status='COMPLETED', seed=seed, population=summary['population'],
            seed_screen_passed=summary['scientific_screen']['seed_screen_passed'])
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('linear-run', 'budget-run', 'generic-run', 'relation-run', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    result = run(parser.parse_args(argv))
    print(result['status'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
