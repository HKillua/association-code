"""074 paired final4000 harm-penalty vs linear Transformer, cached CPU audit."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.frozen_fusion_transfer import _changes
from association.evaluation.sequence_position import REGIONS
from association.evaluation.support_gate_objectives import _artifact, _check_artifact, _read, _structure
from association.evaluation.system_attribution import _same
from association.training.sequence_position import load_checkpoint
from association.utils.runtime import read_json, write_json, write_json_gzip
from association.evaluation.position_sources import CONTROLS, INPUT_KEYS, _metadata, _pool, _sequence_sampling

METHODS = (*CONTROLS, 'fixed_prior095', 'linear', 'harm')
HARM_THRESHOLDS_M = (100., 1000., 10000.)
HIGH_SSE_SCENES = {'20261023': ('3035', '4335', '282'),
                   '20261021': ('3035', '282', '256')}


def verify_loss_pair(new_fit, old_fit):
    new, nr, ne = _sequence_sampling(new_fit)
    old, ore, oe = _sequence_sampling(old_fit)
    expected = deepcopy(old['configuration'])
    expected['position_loss'] = new['configuration']['position_loss']
    expected['scope'] = 'sequence-position-relative-harm-v1'
    if (old['configuration']['position_loss'] != 'linear'
            or new['configuration']['position_loss'] != 'linear_relative_harm'
            or expected != new['configuration']
            or new['training_scene_ids'] != old['training_scene_ids']):
        raise ValueError('074 matched configurations may differ only in declared loss/scope')
    samples = lambda rows: [{k: r[k] for k in ('step', 'mode', 'batch')} for r in rows]
    if samples(nr) != samples(ore) or ne != oe:
        raise ValueError('074 actual successful sampling or exposure differs')
    snapshots = []
    for path, record in ((new_fit, new), (old_fit, old)):
        with torch.random.fork_rng(devices=[]):
            model, saved = load_checkpoint(Path(path)/'step-0.pt', device='cpu')
        del model
        if saved['configuration'] != record['configuration'] or saved['step'] != 0:
            raise ValueError('074 initial checkpoint configuration/step differs from completed fit')
        snapshots.append(saved)
    left, right = snapshots
    if (left['actual_updates'] != 0 or right['actual_updates'] != 0
            or set(left['model']) != set(right['model']) or any(
                left['model'][k].dtype != right['model'][k].dtype
                or not torch.equal(left['model'][k], right['model'][k]) for k in left['model'])
            or left['numpy_rng'] != right['numpy_rng']):
        raise ValueError('074 initial model or sampler RNG differs')
    return dict(status='PAIR_VERIFIED', matched_updates=len(nr), matched_anchors=ne['sampled_anchors'],
        seed=new['configuration']['training']['seed'], initial_parameters_exact=True,
        initial_sampler_rng_exact=True, actual_successful_batches_exact=True, exposure=ne)


def _cache(directory, sid):
    return tuple(_read(Path(directory)/sub/f'states-{sid}.npz')
                 for sub in ('predictions', 'system-predictions'))


def _visible_opportunity(fragment):
    offsets, nodes = fragment['learned_assoc_group_offsets'], fragment['learned_assoc_group_node_indices']
    groups = [nodes[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    return np.asarray([~fragment['support'][g[0]] & fragment['support'][g[1:]].any(0) for g in groups])


def scene_pair(arms, sid):
    reference, saved = _cache(arms['linear']['path']/'final', sid)
    # Sequence caches save support/partition, not Gate-specific opportunity.
    # Derive it from visible inputs solely for the shared structural validator.
    reference['group_eligible'] = _visible_opportunity(reference)
    base = _artifact(saved, 'anchor')
    _, _, regions, population = _structure(reference, base)
    data = {'region_'+name: value for name, value in regions.items()}
    for method in (*CONTROLS, 'fixed_prior095'):
        data[method+'_position_error'] = _check_artifact(reference, _artifact(saved, method), base, method)
    for name, arm in arms.items():
        fragment, system = _cache(arm['path']/'final', sid)
        for key in INPUT_KEYS:
            _same(fragment[key], reference[key], '074 identical visible input/'+key)
        for region in REGIONS:
            _same(fragment['region_'+region], data['region_'+region], '074 visible region/'+region)
        for method in (*CONTROLS, 'fixed_prior095'):
            current, original = _artifact(system, method), _artifact(saved, method)
            if set(current) != set(original):
                raise ValueError('074 ordinary cache schema differs')
            for key in original:
                _same(current[key], original[key], '074 unchanged control/'+method+'/'+key)
            _check_artifact(fragment, current, base, method)
        prediction = _artifact(system, 'learned_sequence')
        data[name+'_position_error'] = _check_artifact(fragment, prediction, base, 'learned_sequence')
        _same(np.isfinite(data[name+'_position_error']), np.isfinite(data['anchor_position_error']),
              '074 common target-point population')
        _same(prediction['group_states'][..., 2:], _artifact(system, 'mean_supported')['group_states'][..., 2:],
              '074 frozen Mean speed/heading')
        initial, initial_system = _cache(arm['path']/'initial', sid)
        for key in INPUT_KEYS:
            _same(initial[key], reference[key], '074 initial visible input/'+key)
        prior = _artifact(system, 'fixed_prior095')
        initial_output = _artifact(initial_system, 'learned_sequence')
        if set(prior) != set(initial_output):
            raise ValueError('074 initial/prior schema differs')
        for key in prior:
            _same(initial_output[key], prior[key], '074 step0 equals fixed .95/'+key)
        _check_artifact(initial, initial_output, base, 'learned_sequence')
    t = len(base['target_ids'])
    available = base['aligned_truth_valid'][:t, :, :2].all(-1)
    covered = available & base['aligned_prediction_valid'][:t, :, :2].all(-1)
    return dict(scene_id=sid, population=population, position_population=dict(
        available_target_points=int(available.sum()), covered_target_points=int(covered.sum()),
        missing_prediction_points=int((available & ~covered).sum()),
        available_target_trajectories=int(available.any(-1).sum()),
        covered_target_trajectories=int(covered.any(-1).sum()))), data


def harm_statistics(error, baseline):
    error, baseline = np.asarray(error, np.float64), np.asarray(baseline, np.float64)
    if error.shape != baseline.shape or not np.isfinite(error).all() or not np.isfinite(baseline).all():
        raise ValueError('harm statistics require common finite errors')
    delta = error-baseline
    result = _changes(delta)
    result['severe_harm'] = {str(int(threshold)): dict(
        threshold_m=threshold, count=int((delta > threshold).sum()),
        cumulative_harm_m=float(delta[delta > threshold].sum())) for threshold in HARM_THRESHOLDS_M}
    result['squared_error_change_sum_m2'] = float(np.sum((error-baseline)*(error+baseline)))
    result['net_identity_residual_m'] = result['signed_error_change_sum']-(result['harm_sum']-result['improvement_sum'])
    return result


def summarize(rows, arrays):
    result = _pool(rows, arrays, METHODS)
    result['relative_harm'] = {}
    for region in REGIONS:
        values = {m: np.concatenate([a[m+'_position_error'][
            a['region_'+region] & np.isfinite(a['anchor_position_error'])] for a in arrays]) for m in METHODS}
        result['relative_harm'][region] = {
            name: harm_statistics(values[name], values['fixed_prior095']) for name in ('linear', 'harm')}
        result['relative_harm'][region]['harm_minus_linear'] = harm_statistics(values['harm'], values['linear'])
    return result


def scientific_screen(summary):
    """Numerical directions are a screen, not a significance or safety claim."""
    old = summary['position_weightings']['all']['linear']['point_equal']
    new = summary['position_weightings']['all']['harm']['point_equal']
    h = summary['relative_harm']['all']
    delta = {key: new[key]-old[key] for key in ('mae', 'rmse', 'p95', 'p99')}
    delta['cumulative_harm_m'] = h['harm']['harm_sum']-h['linear']['harm_sum']
    # Guard numerical signs at sub-micrometre average scale, not an invented
    # engineering minimum effect. Actual sizes and distributions remain primary.
    eps, n = 1e-6, new['unit_count']
    passed = (delta['mae'] <= eps and delta['rmse'] < -eps and delta['p99'] <= eps
              and delta['cumulative_harm_m'] < -eps*n)
    return dict(direction_screen_passed=passed, deltas=delta, numerical_guard_m=eps,
        cumulative_guard_m=eps*n, interpretation='Both seeds must pass; this numerical screen does not establish a practically meaningful or statistically independent gain.')


def run(args):
    seed = int(args.seed)
    names = {20261023: ('074A_harm4000_seed23', '072C_transformer4000'),
             20261021: ('074B_harm4000_seed21', '073C_transformer4000_seed21')}
    if seed not in names or args.cpu_threads != 4:
        raise ValueError('074 comparison fixes one of the two declared seeds and CPU4')
    torch.set_num_threads(4)
    output = Path(args.output).resolve()
    paths = {'harm': Path(args.harm_run).resolve(), 'linear': Path(args.linear_run).resolve()}
    if output.exists():
        raise FileExistsError('074 comparison output exists; no retry/overwrite')
    for path in paths.values():
        source = Path(read_json(path/'run.json')['source_system']).resolve().parent
        for protected in (path, source):
            if output == protected or output in protected.parents or protected in output.parents:
                raise ValueError('074 output overlaps an immutable source')
    output.mkdir(parents=True)
    state = dict(status='RUNNING', model_forwards=0, optimizer_updates=0, test_opened=False, completed_scenes=0)
    write_json(output/'run.json', state)
    try:
        arms = {name: _metadata(paths[name], key) for name, key in zip(('harm', 'linear'), names[seed])}
        first = arms['linear']
        if any(arms['harm'][k] != first[k] for k in ('source_system', 'scene_ids', 'components')):
            raise ValueError('074 source population or components differ')
        pairing = verify_loss_pair(paths['harm']/'fit', paths['linear']/'fit')
        if pairing['seed'] != seed or pairing['matched_updates'] != 4000:
            raise ValueError('074 requires the declared matched4000 seed')
        rows, arrays = [], []
        for sid in first['scene_ids']:
            row, data = scene_pair(arms, sid)
            row['component_id'] = first['components'][sid]
            rows.append(row); arrays.append(data)
            state['completed_scenes'] = len(rows)
            write_json(output/'run.json', state)
        summary = summarize(rows, arrays)
        for name, arm in arms.items():
            actual = summary['position_weightings']['all'][name]['point_equal']
            expected = arm['summary']['methods']['learned_sequence']['full_target_metrics']['all']['position']
            if actual['unit_count'] != expected['intersection_count'] or any(
                    not np.isclose(actual[k], expected[k], rtol=1e-10, atol=1e-9) for k in ('mae', 'rmse')):
                raise ValueError('074 cache does not reproduce producer physical metrics/'+name)
        scene_rows = [dict(summarize([r], [a]), component_id=r['component_id']) for r, a in zip(rows, arrays)]
        components = {}
        for cid in sorted(set(first['components'].values())):
            indices = [i for i, row in enumerate(rows) if row['component_id'] == cid]
            components[cid] = summarize([rows[i] for i in indices], [arrays[i] for i in indices])
        largest = min(components, key=lambda cid: (-components[cid]['position_population']['covered_target_points'], cid))
        rest = [i for i, row in enumerate(rows) if row['component_id'] != largest]
        summary.update(status='COMPLETED', seed=seed, pairing=pairing, methods=list(METHODS),
            components=components, largest_component_id=largest,
            remaining_components=summarize([rows[i] for i in rest], [arrays[i] for i in rest]) if rest else None,
            predeclared_high_sse_scenes=[r for r in scene_rows if r['scene_ids'][0] in HIGH_SSE_SCENES[str(seed)]],
            numerical_screen=scientific_screen(summary), inputs={k: str(v) for k, v in paths.items()},
            model_forwards=0, optimizer_updates=0, test_opened=False, old066_acceptance='REJECTED',
            interpretation='Fixed endpoint paired loss intervention; harm is relative to .95, not absolute tail-risk protection. Reused V and shared identities; no independent significance claim.')
        write_json_gzip(output/'summary.json.gz', summary)
        write_json_gzip(output/'scene-summary.json.gz', scene_rows)
        table = ['# 074 相对初值退化惩罚：同seed固定终点', '',
                 '| 方法 | MAE m | RMSE m | P95 m | P99 m |', '|---|---:|---:|---:|---:|']
        for name in METHODS:
            metrics = summary['position_weightings']['all'][name]['point_equal']
            table.append('| '+name+' | '+' | '.join(f'{metrics[k]:.4f}' for k in ('mae', 'rmse', 'p95', 'p99'))+' |')
        table += ['', '数值方向筛查：'+str(summary['numerical_screen']),
                  '两seed必须全部报告。H/I、100/1000/10000m额外伤害、逐场/组件与高SSE既定场景见JSON。']
        (output/'report.md').write_text('\n'.join(table)+'\n')
        state.update(status='COMPLETED', population=summary['population'])
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('harm-run', 'linear-run', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--seed', type=int, choices=(20261023, 20261021), required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    result = run(parser.parse_args(argv))
    print(result['status'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
