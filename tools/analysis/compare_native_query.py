"""100 complete cached comparison: strong090C, budget, native mean and query.

Read-only inputs; no model forward, archive read, regrouping or model selection.
Checkpoint loading verifies actual parameters, not version hashes or filenames.
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
from association.evaluation.position_sources import CONTROLS, INPUT_KEYS, ROOT, _pool, _sequence_sampling
from association.evaluation.sequence_position import REGIONS, WEIGHTINGS
from association.evaluation.support_gate_objectives import _artifact, _check_artifact, _structure
from association.models.sequence_position import tangent_position
from association.training.frozen_position_adapter import _exact, load_checkpoint as load_strong
from association.training.native_query import ARMS, SCOPE, load_checkpoint
from association.utils.runtime import read_json, write_json, write_json_gzip
from tools.analysis.compare_frozen_position_adapter import adapter_metadata, declared_manifest, _fixed_artifact
from tools.analysis.compare_position_harm import _cache, _visible_opportunity, harm_statistics

METHODS = (*CONTROLS, 'fixed_prior095', 'strong', *ARMS)
PAIRS = (('budget', 'strong'), ('mean', 'strong'), ('query', 'strong'),
         ('mean', 'budget'), ('query', 'budget'), ('query', 'mean'))


def declared_configuration(arm):
    if arm not in ARMS:
        raise ValueError('100 unknown comparison arm')
    return read_json(ROOT/'configs'/f'native-query-{arm}-level1.json')


def native_metadata(path, arm, strong):
    path = Path(path).resolve()
    run, summary, fit = [read_json(path/name) for name in ('run.json', 'summary.json', 'fit/run.json')]
    config = declared_configuration(arm); steps = config['training']['updates']
    manifest = declared_manifest()
    for record in (run, summary, fit):
        if (record.get('status') != 'COMPLETED' or record.get('scope') != SCOPE
                or record.get('configuration') != config or record.get('optimizer_updates') != steps
                or record.get('test_opened') is not False):
            raise ValueError('100 incomplete or mismatched successful fixed-endpoint chain/'+arm)
    expected_source = dict(run=str(strong['path']), checkpoint=str(strong['path']/'fit/final.pt'),
                           source_system=strong['source_system'], source_total_updates=8000)
    if (run.get('completed_endpoints') != ['final'] or run.get('validation_selection') is not False
            or run.get('source_system') != strong['source_system']
            or run.get('strong_run') != str(strong['path'])
            or summary.get('strong_run') != str(strong['path'])
            or summary.get('source_system') != strong['source_system']
            or fit.get('source') != expected_source or fit.get('final_checkpoint') != 'final.pt'
            or fit.get('training_scene_ids') != manifest['splits']['T_core']):
        raise ValueError('100 strong source, successful training population or fixed endpoint differs/'+arm)
    endpoint = read_json(path/'final/summary.json')
    expected_checkpoint = dict(path=str(path/'fit/final.pt'), actual_updates=steps, step=steps,
        configured_final_updates=steps, strong_source=str(strong['path']/'fit/final.pt'))
    for record in (read_json(path/'final/run.json'), endpoint):
        if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                or record.get('checkpoint') != expected_checkpoint or record.get('device') != 'cpu'
                or record.get('source_system') != strong['source_system']
                or record.get('optimizer_updates') != 0 or record.get('test_opened') is not False
                or record.get('initial') is not False or record.get('groups_recomputed') is not False
                or record.get('threshold_reselected') is not False):
            raise ValueError('100 final CPU cache must bind actual final parameters/'+arm)
    components = {sid: str(cid) for cid, row in endpoint['components'].items() for sid in row['scene_ids']}
    if endpoint['scene_ids'] != manifest['splits']['V_select'] or components != strong['components']:
        raise ValueError('100 complete V scene/component population differs/'+arm)
    for sub in ('predictions', 'system-predictions'):
        if {p.name for p in (path/'final'/sub).glob('*.npz')} != {f'states-{sid}.npz' for sid in endpoint['scene_ids']}:
            raise ValueError('100 complete output array inventory differs/'+arm)
    return dict(path=path, configuration=config, source_system=run['source_system'],
        scene_ids=endpoint['scene_ids'], components=components, summary=endpoint, run=run, fit=fit)


def verify_initial_states(initials, base_state):
    """Compare actual initialization tensors; a matching seed is insufficient."""
    if set(initials) != set(ARMS):
        raise ValueError('100 requires all three actual initial checkpoints')
    if any(not _exact(saved['base_state'], base_state) for saved in initials.values()):
        raise ValueError('100 actual frozen base tensors differ from strong090C')
    if not _exact(initials['budget']['branch_state'], base_state):
        raise ValueError('100 added-budget initial tensors differ from strong090C')
    states = {arm: {k: v for k, v in initials[arm]['branch_state'].items() if not k.startswith('pooler.')}
              for arm in ('mean', 'query')}
    if not _exact(states['mean'], states['query']):
        raise ValueError('100 mean/query common initial tensors differ outside pooler')
    for arm in ('mean', 'query'):
        if any(torch.count_nonzero(initials[arm]['branch_state'][key]) for key in ('output.weight', 'output.bias')):
            raise ValueError('100 native branch initial output must equal frozen b')
    return dict(common_initial_tensors_exact=True, common_tensor_count=len(states['mean']),
                budget_initial_equals_strong=True, native_zero_heads=True)


def verify_pairing(arms):
    strong_path = arms['strong']['path']
    original, original_rows, original_exposure = _sequence_sampling(strong_path/'fit')
    with torch.random.fork_rng(devices=[]):
        strong_model, strong_saved = load_strong(strong_path/'fit/final.pt')
    if original['configuration']['model']['mode'] != 'full_finetune':
        raise ValueError('100 strong control must be090C full fine-tuning')
    base_state = strong_model.teacher.state_dict()
    base_config = strong_saved['source_teacher']['configuration']['model']
    source = dict(run=str(strong_path), checkpoint=str(strong_path/'fit/final.pt'),
                  source_system=arms['strong']['source_system'], source_total_updates=8000)
    samples = lambda rows: [{key: row[key] for key in ('step', 'mode', 'batch')} for row in rows]
    initials, reports = {}, {}
    for arm in ARMS:
        fit = arms[arm]['path']/'fit'
        run, rows, exposure = _sequence_sampling(fit)
        config = arms[arm]['configuration']; steps = config['training']['updates']
        if (run['configuration'] != config or samples(rows) != samples(original_rows)
                or exposure != original_exposure or run['training_scene_ids'] != original['training_scene_ids']):
            raise ValueError('100 actual successful batches or complete exposure differ/'+arm)
        snapshots = []
        for filename, step in (('step-0.pt', 0), ('final.pt', steps)):
            with torch.random.fork_rng(devices=[]):
                model, saved = load_checkpoint(fit/filename, device='cpu')
            if (saved['configuration'] != config or saved['step'] != step or saved['actual_updates'] != step
                    or saved['source'] != source or saved['base_model_config'] != base_config
                    or not _exact(saved['base_state'], base_state)):
                raise ValueError('100 actual checkpoint/base/source binding differs/'+arm)
            optimizer = saved['optimizer']; groups = optimizer.get('param_groups', [])
            entries = optimizer.get('state', {})
            if (len(groups) != 1 or groups[0]['lr'] != config['training']['learning_rate']
                    or len(groups[0]['params']) != len(list(model.branch.parameters()))
                    or groups[0]['weight_decay'] != 0 or groups[0]['betas'] != (0.9, .999)
                    or (step == 0 and entries)
                    or (step > 0 and (set(entries) != set(groups[0]['params'])
                        or any(float(entry.get('step', -1)) != step for entry in entries.values())))):
                raise ValueError('100 fresh Adam or actual optimizer update state differs/'+arm)
            snapshots.append(saved)
        initials[arm] = snapshots[0]
        clipped = sum(row['gradient_norm'] > config['training']['gradient_clip'] for row in rows)
        if (any(row.get('gradient_clipped') != (row['gradient_norm'] > config['training']['gradient_clip']) for row in rows)
                or run['gradient_clipping']['clipped_updates'] != clipped):
            raise ValueError('100 clipping accounting differs from successful gradients/'+arm)
        counts = dict(parameter_count=sum(p.numel() for p in model.parameters()),
                      trainable_parameter_count=sum(p.numel() for p in model.branch.parameters()))
        if any(run[key] != value for key, value in counts.items()):
            raise ValueError('100 parameter count differs from actual checkpoint/'+arm)
        losses = np.array([row['loss'] for row in rows], np.float64)
        window = min(100, len(losses))
        reports[arm] = dict(**counts, actual_successful_batches_exact=True,
            source_updates=strong_saved['step']+strong_saved['source_teacher']['step'],
            added_updates=steps, frozen_base_tensors_exact=True, fresh_adam=True,
            branch_changed=not _exact(snapshots[0]['branch_state'], snapshots[1]['branch_state']),
            clipped_updates=clipped, clip_fraction=clipped/len(rows),
            first_window_mean_loss=float(losses[:window].mean()), last_window_mean_loss=float(losses[-window:].mean()),
            loss_window_updates=window, fit_elapsed_seconds=run.get('elapsed_seconds'),
            total_elapsed_seconds=arms[arm]['run'].get('elapsed_seconds'),
            successful_update_seconds=float(sum(row['seconds'] for row in rows)))
    return dict(status='PAIR_VERIFIED', **verify_initial_states(initials, base_state),
        matched_updates=len(original_rows), matched_anchors=original_exposure['sampled_anchors'],
        exposure=original_exposure, arms=reports,
        interpretation='Actual scene/anchor/member and successful exposure pairing; wall time/parameter count need not match.')


def scene_pair(arms, sid):
    reference, source = _cache(arms['strong']['path']/'final', sid)
    reference['group_eligible'] = _visible_opportunity(reference)
    base = _artifact(source, 'anchor'); _, _, regions, population = _structure(reference, base)
    data = {'region_'+name: mask for name, mask in regions.items()}
    for control in (*CONTROLS, 'fixed_prior095'):
        data[control+'_position_error'] = _check_artifact(reference, _artifact(source, control), base, control)
    teacher = _artifact(source, 'learned_sequence')
    data['strong_position_error'] = _check_artifact(reference, teacher, base, 'learned_sequence')
    for arm in ARMS:
        fragment, system = _cache(arms[arm]['path']/'final', sid)
        for key in INPUT_KEYS:
            exact(fragment[key], reference[key], '100 identical input/'+key)
        for region in REGIONS:
            exact(fragment['region_'+region], data['region_'+region], '100 region/'+region)
        for control in (*CONTROLS, 'fixed_prior095'):
            current = _artifact(system, control)
            _fixed_artifact(current, _artifact(source, control), '100 control/'+control, predictions=True)
            _check_artifact(fragment, current, base, control)
        current = _artifact(system, 'learned_sequence')
        _fixed_artifact(current, teacher, '100 target/mask/group population')
        data[arm+'_position_error'] = _check_artifact(fragment, current, base, 'learned_sequence')
        exact(np.isfinite(data[arm+'_position_error']), np.isfinite(data['strong_position_error']), '100 common errors')
        exact(fragment['group_frozen_base_states'], teacher['group_states'], '100 frozen base versus090C final')
        exact(current['group_states'][..., 2:], teacher['group_states'][..., 2:], '100 unchanged speed/course')
        origin = _artifact(system, 'fixed_prior095')['group_states'] if arm == 'budget' else teacher['group_states']
        residual = fragment['group_native_increment_m']
        if residual.shape != (*origin.shape[:2], 2) or not np.isfinite(residual).all():
            raise ValueError('100 invalid complete native residual cache')
        rebuilt = tangent_position(torch.from_numpy(origin.reshape(-1, 4)),
                                   torch.from_numpy(residual.reshape(-1, 2))).numpy().reshape(origin.shape)
        exact(current['group_states'], rebuilt, '100 declared tangent output geometry')
    t = len(base['target_ids']); available = base['aligned_truth_valid'][:t, :, :2].all(-1)
    covered = available & base['aligned_prediction_valid'][:t, :, :2].all(-1)
    return dict(scene_id=sid, population=population, position_population=dict(
        available_target_points=int(available.sum()), covered_target_points=int(covered.sum()),
        missing_prediction_points=int((available & ~covered).sum()), available_target_trajectories=int(available.any(-1).sum()),
        covered_target_trajectories=int(covered.any(-1).sum()))), data


def summarize(rows, arrays):
    result = _pool(rows, arrays, METHODS)
    result['pairwise'] = {}
    pairs = (*PAIRS, *((arm, control) for arm in ARMS for control in (*CONTROLS, 'fixed_prior095')))
    for left, right in pairs:
        result['pairwise'][left+'__minus__'+right] = {}
        for region in REGIONS:
            values = {name: np.concatenate([a[name+'_position_error'][a['region_'+region] &
                np.isfinite(a['strong_position_error'])] for a in arrays]) for name in (left, right)}
            result['pairwise'][left+'__minus__'+right][region] = harm_statistics(values[left], values[right])
    return result


def scientific_screen(summary, scenes):
    """A single-seed development screen; never authorize further computation."""
    eps = 1e-6; results = {}
    def metric(item, arm): return item['position_weightings']['all'][arm]
    def finite(value): return value is not None and np.isfinite(value)
    def nonworse(left, right): return finite(left) and finite(right) and left <= right+eps
    def pair_mae(item, left, right):
        a, b = metric(item, left)['point_equal']['mae'], metric(item, right)['point_equal']['mae']
        return (a, b) if finite(a) and finite(b) else None
    for arm in ('mean', 'query'):
        comparisons = {}
        for control in ('strong', 'budget'):
            new, old = metric(summary, arm), metric(summary, control)
            current, baseline = new['point_equal']['mae'], old['point_equal']['mae']
            reduction = 1-current/baseline if finite(current) and finite(baseline) and baseline > 0 else None
            tails = all(nonworse(new['point_equal'][key], old['point_equal'][key]) for key in ('rmse', 'p99'))
            weights = all(nonworse(new[key]['mae'], old[key]['mae']) for key in WEIGHTINGS)
            scene_pairs = [pair_mae(item, arm, control) for item in scenes]
            component_pairs = [pair_mae(item, arm, control) for item in summary['components'].values()]
            scene_wins = sum(pair is not None and pair[0] < pair[1]-eps for pair in scene_pairs)
            component_wins = sum(pair is not None and pair[0] < pair[1]-eps for pair in component_pairs)
            rest = summary['remaining_components']; outside_pair = pair_mae(rest, arm, control) if rest else None
            outside = outside_pair is not None and nonworse(*outside_pair)
            passed = reduction is not None and reduction >= .03 and tails and weights and outside \
                and scene_wins > len(scenes)/2 and component_wins > len(summary['components'])/2
            comparisons[control] = dict(screen_passed=bool(passed), relative_mae_reduction=reduction,
                required_reduction=.03, tails_nonworse=bool(tails), four_weight_mae_nonworse=bool(weights),
                outside_largest_component_nonworse=bool(outside), improved_scenes=scene_wins,
                total_scenes=len(scenes), unevaluable_scenes=sum(pair is None for pair in scene_pairs),
                improved_components=component_wins, total_components=len(component_pairs),
                unevaluable_components=sum(pair is None for pair in component_pairs))
        passed = all(item['screen_passed'] for item in comparisons.values())
        results[arm] = dict(status='PROMISING_SINGLE_SEED' if passed else 'NO_CLEAR_SINGLE_SEED_ADVANTAGE',
            comparisons=comparisons, automatic_new_training=False,
            missing_population_policy='Retain unevaluable scenes/components in total counts; they are not wins.')
    return results


def run(args):
    if getattr(args, 'cpu_threads', 4) != 4: raise ValueError('100 comparison fixes CPU4')
    torch.set_num_threads(4)
    paths = {name: Path(getattr(args, name+'_run')).resolve() for name in ('strong', *ARMS)}
    if len(set(paths.values())) != 4: raise ValueError('100 requires distinct strong and three experiment runs')
    output = Path(args.output).resolve()
    protected = [*paths.values(), Path(read_json(paths['strong']/'run.json')['source_system']).resolve().parent]
    if output.exists(): raise FileExistsError('100 comparison output already exists; no overwrite')
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError('100 derived output must not overlap immutable inputs')
    output.mkdir(parents=True)
    state = dict(status='RUNNING', model_forwards=0, optimizer_updates=0, test_opened=False, completed_scenes=0)
    write_json(output/'run.json', state)
    try:
        label, strong = adapter_metadata(paths['strong'])
        if label != 'finetune': raise ValueError('100 strong input must be090C full-finetune')
        arms = {'strong': strong, **{arm: native_metadata(paths[arm], arm, strong) for arm in ARMS}}
        pairing = verify_pairing(arms); rows, arrays = [], []
        for sid in strong['scene_ids']:
            row, data = scene_pair(arms, sid); row['component_id'] = strong['components'][sid]
            rows.append(row); arrays.append(data)
            state['completed_scenes'] = len(rows); write_json(output/'run.json', state)
        summary = summarize(rows, arrays)
        for arm in arms:
            actual = summary['position_weightings']['all'][arm]['point_equal']
            expected = arms[arm]['summary']['methods']['learned_sequence']['full_target_metrics']['all']['position']
            if actual['unit_count'] != expected['intersection_count'] or any(not np.isclose(actual[key], expected[key], rtol=1e-10, atol=1e-9) for key in ('mae', 'rmse')):
                raise ValueError('100 arrays do not reproduce producer physical metrics/'+arm)
        scenes = [dict(summarize([row], [data]), component_id=row['component_id']) for row, data in zip(rows, arrays)]
        components = {}
        for cid in sorted(set(strong['components'].values())):
            chosen = [i for i, row in enumerate(rows) if row['component_id'] == cid]
            components[cid] = summarize([rows[i] for i in chosen], [arrays[i] for i in chosen])
        largest = min(components, key=lambda cid: (-components[cid]['position_population']['covered_target_points'], cid))
        rest = [i for i, row in enumerate(rows) if row['component_id'] != largest]
        summary.update(status='COMPLETED', methods=list(METHODS), pairing=pairing, components=components,
            largest_component_id=largest,
            remaining_components=summarize([rows[i] for i in rest], [arrays[i] for i in rest]) if rest else None,
            source_system=strong['source_system'], inputs={k: str(p) for k, p in paths.items()},
            model_forwards=0, optimizer_updates=0, test_opened=False, old066_acceptance='REJECTED',
            frozen_base_arrays_exact=True, population_masks_controls_exact=True,
            interpretation='Single-seed repeated-V screen. T1/T2 differ in pooler and 64 parameters; total compute is not identical. Native representation attribution still needs T1P and replication. No peer victory or automatic new experiment.',
            quantile_definition='P95/P99 use complete point-error arrays; other weightings report their established MAE/RMSE, not quantiles of scene averages.')
        summary['scientific_screen'] = scientific_screen(summary, scenes)
        write_json_gzip(output/'summary.json.gz', summary); write_json_gzip(output/'scene-summary.json.gz', scenes)
        table = ['# 100 原生观测 Mean/Query 与同更新预算对照', '',
                 '| 方法 | MAE m | RMSE m | P95 m | P99 m |', '|---|---:|---:|---:|---:|']
        for arm in METHODS:
            values = summary['position_weightings']['all'][arm]['point_equal']
            table.append('| '+arm+' | '+' | '.join(f'{values[k]:.6f}' if values[k] is not None else 'NA'
                                                 for k in ('mae', 'rmse', 'p95', 'p99'))+' |')
        table += ['', '完整人口、覆盖/重复、四权重、机会区、逐场/共享组件及最大组件外见 JSON。',
                  '这是单 seed 开发比较；通过仅记 promising，不自动追加训练，不等于同行优势。']
        (output/'report.md').write_text('\n'.join(table)+'\n')
        state.update(status='COMPLETED', population=summary['population']); write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}'); write_json(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('strong-run', 'budget-run', 'mean-run', 'query-run', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    result = run(parser.parse_args(argv)); print(result['status'], flush=True); return 0


if __name__ == '__main__': raise SystemExit(main())
