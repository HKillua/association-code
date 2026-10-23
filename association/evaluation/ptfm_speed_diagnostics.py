"""076A: read-only speed quality and dynamics on frozen 074C outputs.

No network/checkpoint/archive access. Zero speed is constructed for every output
before the frozen first-representative mapping; truth only defines diagnostics.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.physical import _check_valid_states
from association.evaluation.scene import _regions
from association.evaluation.support_gate_objectives import _artifact, _read, _structure, _check_artifact
from association.evaluation.system_attribution import ATTRIBUTES, _same, assert_metrics, pool_metrics

SOURCE_METHODS = ('anchor', 'mean_supported', 'attribute_hybrid', 'support_switch', 'fixed_prior095', 'ptfm_adapted')
METHODS = ('anchor', 'mean_supported', 'ptfm_adapted', 'zero_speed')
BINS = ('exactly_zero', 'gt0_le1', 'gt1_le5', 'gt5_le20', 'gt20')
STRATA = ('all', 'supported_helper_present', 'supported_helper_absent')


def distribution(values):
    values = np.asarray(values, np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError('diagnostic distributions require finite declared values')
    return dict(count=len(values), mean=float(values.mean()) if len(values) else None,
        std=float(values.std()) if len(values) else None,
        **{key: float(np.quantile(values, q)) if len(values) else None
           for key, q in (('min', 0), ('p01', .01), ('p05', .05), ('median', .5), ('p95', .95), ('p99', .99), ('max', 1))})


def pearson(a, b):
    """Undefined for fewer than two points or exactly zero centered variance."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('correlation requires same finite population')
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    x, y = a - a.mean(), b - b.mean()
    xx, yy = float(x @ x), float(y @ y)
    if xx == 0 or yy == 0:
        return None
    return float(np.clip((x @ y) / np.sqrt(xx * yy), -1, 1))


def error_stats(error):
    error = np.asarray(error, np.float64)
    return dict(point_count=int(error.size), mae=float(np.abs(error).mean()) if error.size else None,
        mse=float(np.square(error).mean()) if error.size else None,
        rmse=float(np.sqrt(np.square(error).mean())) if error.size else None,
        absolute_error=distribution(np.abs(error)))


def dynamics(prediction, truth, valid, query_seconds):
    """Per scene-target dynamics; adjacency never jumps over an invalid query."""
    prediction, truth = np.asarray(prediction, float), np.asarray(truth, float)
    valid, times = np.asarray(valid), np.asarray(query_seconds, float)
    if prediction.shape != truth.shape or valid.shape != truth.shape or valid.dtype != bool or truth.ndim != 2:
        raise ValueError('dynamics requires target-by-query arrays and boolean masks')
    times = np.broadcast_to(times, truth.shape)
    if not np.isfinite(times).all() or np.any(np.diff(times, axis=1) <= 0):
        raise ValueError('query times must be finite strictly increasing seconds')
    if not np.isfinite(prediction[valid]).all() or not np.isfinite(truth[valid]).all():
        raise ValueError('dynamics nonfinite valid input')
    pred_std, truth_std, corr, centered_error, increments, rates, dt_values = [], [], [], [], [], [], []
    target_count = 0
    for pred, ref, use, clock in zip(prediction, truth, valid, times):
        n = int(use.sum())
        if n:
            target_count += 1
        if n >= 2:
            p, r = pred[use], ref[use]
            pred_std.append(float(p.std()) if np.ptp(p) else 0.); truth_std.append(float(r.std()) if np.ptp(r) else 0.)
            corr.append(pearson(p, r))
            centered_error.extend((p-p.mean())-(r-r.mean()))
        adjacent = use[:-1] & use[1:]
        dt = np.diff(clock)[adjacent]
        delta_error = (np.diff(pred)[adjacent] - np.diff(ref)[adjacent])
        increments.extend(delta_error); rates.extend(delta_error/dt); dt_values.extend(dt)
    defined = [x for x in corr if x is not None]
    return dict(targets_with_valid_points=target_count, targets_with_at_least_two_points=len(pred_std),
        prediction_within_target_std=distribution(pred_std), truth_within_target_std=distribution(truth_std),
        target_pearson=dict(defined_target_count=len(defined), undefined_target_count=len(corr)-len(defined),
            macro_mean=float(np.mean(defined)) if defined else None, values=distribution(defined)),
        centered_error=error_stats(centered_error), adjacent_increment_error=error_stats(increments),
        adjacent_rate_error_per_second=error_stats(rates), adjacent_query_dt_seconds=distribution(dt_values))


def load_scene(endpoint, scene_id):
    """Validate full original partitions, all six outputs and every error mask."""
    endpoint, sid = Path(endpoint), str(scene_id)
    fragment = _read(endpoint/'predictions'/f'states-{sid}.npz')
    system = _read(endpoint/'system-predictions'/f'states-{sid}.npz')
    base = _artifact(system, 'anchor')
    if set(system) != {method+'_'+key for method in SOURCE_METHODS for key in base}:
        raise ValueError('074C requires all six complete original artifacts')
    offsets, flat = fragment['learned_assoc_group_offsets'], fragment['learned_assoc_group_node_indices']
    groups = [flat[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    # Existing structural validator expects an opportunity array absent in074C.
    # Derive from observable support only; every cached region is still checked.
    supported = np.array([fragment['support'][g[1:]].any(0) for g in groups])
    fragment['group_eligible'] = ~fragment['support'][fragment['group_anchor_node_indices']] & supported
    groups, selected, _, population = _structure(fragment, base)
    _same(supported, fragment['group_helper_present'], '076A visible helper presence')
    t = len(base['target_ids']); q = len(fragment['query_seconds'])
    if not np.isfinite(fragment['query_seconds']).all() or np.any(np.diff(fragment['query_seconds']) <= 0):
        raise ValueError('076A actual query seconds must increase')
    _same(base['split'], np.asarray('V_select'), '076A V split')
    _same(base['grouping_kind'], np.asarray('predicted'), '076A predicted partition')
    _same(base['correspondence_policy'], np.asarray('anchor_identity_first'), '076A fixed correspondence')
    replay, speed = {}, {}
    full = {}
    for method in SOURCE_METHODS:
        artifact = _artifact(system, method)
        _check_artifact(fragment, artifact, base, method)
        _check_valid_states(artifact['group_states'], base['group_valid'], '076A full output/'+method)
        full[method] = _regions(artifact['aligned_prediction'], base['aligned_truth'], base['aligned_truth_valid'],
            base['aligned_prediction_valid'], base['aligned_observed'])
        errors = attribute_errors(artifact['aligned_prediction'][:t], base['aligned_truth'][:t])
        common = base['aligned_truth_valid'][:t] & base['aligned_prediction_valid'][:t]
        for attr, cols in ATTRIBUTES.items():
            valid = common[..., list(cols)].all(-1)
            saved = fragment[method+'_'+attr+'_error']
            _same(np.isfinite(saved), valid, '076A cached '+attr+' mask/'+method)
            if not np.allclose(errors[attr][valid], saved[valid], rtol=0, atol=1e-9):
                raise ValueError('076A cached physical error differs/'+method+'/'+attr)
        replay[method] = dict(population=population, full_target_metrics=full[method])
        if method in METHODS: speed[method] = artifact['aligned_prediction'][:t, :, 2]
    # Zero control is generated on ALL output groups, including duplicates and
    # reference-unavailable groups. The unchanged original map is applied later.
    zero_groups = np.zeros((len(groups), q), dtype=np.float64)
    aligned = np.full(base['aligned_group_indices'].shape+(q,), np.nan)
    represented = base['aligned_group_indices'] >= 0
    aligned[represented] = zero_groups[base['aligned_group_indices'][represented]]
    speed['zero_speed'] = aligned[:t]
    helper = np.zeros((t, q), bool)
    helper[selected >= 0] = supported[selected[selected >= 0]]
    _same(helper, fragment['region_supported_helper_present'], '076A helper stratum')
    _same((selected[:, None] >= 0) & ~helper, fragment['region_supported_helper_absent'], '076A no-helper stratum')
    return dict(scene_id=sid, population=population, replay=replay, methods=speed,
        truth=base['aligned_truth'][:t, :, 2], valid=(base['aligned_truth_valid'][:t, :, 2] & base['aligned_prediction_valid'][:t, :, 2]),
        truth_valid=base['aligned_truth_valid'][:t, :, 2], represented=np.broadcast_to(selected[:, None] >= 0, (t, q)),
        helper=helper, query_seconds=np.broadcast_to(fragment['query_seconds'], (t, q)),
        native_speed=fragment['native_states'][..., 2],
        group_speed={m: _artifact(system, m)['group_states'][..., 2] for m in METHODS[:-1]} | {'zero_speed': zero_groups},
        group_helper=supported, fragment_count=len(fragment['node_ids']),
        zero_control_all_groups_verified=True)


def _cat(values):
    return np.concatenate(values, axis=0)


def summarize(records):
    """Pooled point metrics and target/scene denominators stay separate."""
    truth, valid, helper, represented, times = [_cat([r[k] for r in records]) for k in ('truth', 'valid', 'helper', 'represented', 'query_seconds')]
    methods = {m: _cat([r['methods'][m] for r in records]) for m in METHODS}
    regions = dict(all=np.ones_like(valid), supported_helper_present=helper, supported_helper_absent=represented & ~helper)
    bins = dict(exactly_zero=truth == 0, gt0_le1=(truth > 0)&(truth <= 1), gt1_le5=(truth > 1)&(truth <= 5),
        gt5_le20=(truth > 5)&(truth <= 20), gt20=truth > 20)
    if int(sum((v & valid).sum() for v in bins.values())) != int(valid.sum()):
        raise ValueError('fixed native speed bins must partition valid nonnegative truth')
    result = dict(scene_count=len(records), scene_ids=[r['scene_id'] for r in records],
        population={key: sum(r['population'][key] for r in records) for key in records[0]['population']},
        available_truth_point_count=int(_cat([r['truth_valid'] for r in records]).sum()), common_speed_point_count=int(valid.sum()),
        native_input_distribution=distribution(_cat([r['native_speed'].ravel() for r in records])),
        input_population='all native reconstructed fragment/query states, not original raw observations',
        all_output_group_distributions={m: distribution(_cat([r['group_speed'][m].ravel() for r in records])) for m in METHODS},
        all_output_group_helper_counts={key: int(sum((r['group_helper'] if key == 'present' else ~r['group_helper']).sum() for r in records))
            for key in ('present', 'absent')}, strata={}, fixed_truth_speed_bins={})
    for region, select in regions.items():
        use = valid & select
        result['strata'][region] = dict(point_count=int(use.sum()), truth_distribution=distribution(truth[use]), methods={})
        for method, pred in methods.items():
            by_scene = [pearson(r['methods'][method][r['valid'] & (np.ones_like(r['valid']) if region == 'all' else
                r['helper'] if region == 'supported_helper_present' else r['represented'] & ~r['helper'])],
                r['truth'][r['valid'] & (np.ones_like(r['valid']) if region == 'all' else
                r['helper'] if region == 'supported_helper_present' else r['represented'] & ~r['helper'])]) for r in records]
            defined = [x for x in by_scene if x is not None]
            result['strata'][region]['methods'][method] = dict(error_stats(pred[use]-truth[use]),
                prediction_distribution=distribution(pred[use]), pearson=pearson(pred[use], truth[use]),
                scene_pearson=dict(defined_scene_count=len(defined), undefined_scene_count=len(records)-len(defined),
                    macro_mean=float(np.mean(defined)) if defined else None),
                dynamics=dynamics(pred, truth, use, times))
    for name, select in bins.items():
        use = valid & select
        result['fixed_truth_speed_bins'][name] = dict(point_count=int(use.sum()), methods={m: error_stats(p[use]-truth[use]) for m,p in methods.items()})
    return result


def analyze(endpoint, summary, components, check=None):
    import gzip, json
    endpoint=Path(endpoint); records=[]; rows=[]
    with gzip.open(endpoint/'scene-summary.json.gz', 'rt', encoding='utf-8') as handle:
        original_rows=json.load(handle)
    if [r['scene_id'] for r in original_rows]!=summary['scene_ids']:
        raise ValueError('076A source per-scene summary order differs')
    source056=Path(summary['source_system'])
    for sid in summary['scene_ids']:
        if check: check()
        row=load_scene(endpoint,sid);row['component_id']=str(components[sid])
        previous=original_rows[len(records)]
        if str(previous['component_id'])!=row['component_id'] or previous['fragment_count']!=row['fragment_count']:
            raise ValueError('076A source scene metadata differs')
        for method in SOURCE_METHODS:
            if previous['methods'][method]['population']!=row['population']:
                raise ValueError('076A source scene population differs')
            assert_metrics(row['replay'][method]['full_target_metrics'],previous['methods'][method]['full_target_metrics'],'076A scene replay/'+sid+'/'+method)
        source_fragment=_read(source056/'predictions'/f'states-{sid}.npz')
        current_fragment=_read(endpoint/'predictions'/f'states-{sid}.npz')
        for key in ('node_ids','identities','target_identity_ids','native_states','support','nearest_seconds','exact_observation','query_seconds','learned_assoc_group_offsets','learned_assoc_group_node_indices'):
            _same(current_fragment[key],source_fragment[key],'076A frozen056 input/'+key)
        source_system=_read(source056/'system-predictions'/f'states-{sid}.npz')
        current_system=_read(endpoint/'system-predictions'/f'states-{sid}.npz')
        for method in SOURCE_METHODS[:3]:
            before,after=_artifact(source_system,'learned_assoc_'+method),_artifact(current_system,method)
            if not before or set(before)!=set(after):raise ValueError('076A original056 artifact schema differs')
            for key in before:_same(after[key],before[key],'076A original056 population and ordinary outputs/'+key)
        records.append(row)
        rows.append(dict(scene_id=sid,component_id=row['component_id'],diagnostics=summarize([row])))
    for method in SOURCE_METHODS:
        pooled = pool_metrics([r['replay'][method]['full_target_metrics'] for r in records])
        assert_metrics(pooled, summary['methods'][method]['full_target_metrics'], '076A source pooled/'+method)
        population={k:sum(r['population'][k] for r in records) for k in records[0]['population']}
        if population != summary['methods'][method]['population']: raise ValueError('076A source population differs')
        for component, original in summary['components'].items():
            selected=[r for r in records if r['component_id']==str(component)]
            assert_metrics(pool_metrics([r['replay'][method]['full_target_metrics'] for r in selected]),
                original['methods'][method]['full_target_metrics'],'076A source component/'+str(component)+'/'+method)
        for attr in ATTRIBUTES:
            count=summary['methods'][method]['attribute_quantiles'][attr]['all']['point_count']
            if count != pooled['all'][attr]['intersection_count']: raise ValueError('076A source quantile population differs')
    if sum(r['fragment_count'] for r in records) != summary['fragment_count']:
        raise ValueError('076A fragment population differs')
    result = dict(status='COMPLETED', methods=list(METHODS), pooled=summarize(records),
        components={c:summarize([r for r in records if r['component_id']==c]) for c in sorted(set(components.values()))},
        source_physical_replay={m:pool_metrics([r['replay'][m]['full_target_metrics'] for r in records]) for m in SOURCE_METHODS},
        source_population_and_mapping_verified=True, zero_control_all_groups_verified=True,
        limitations=['V_select reused; shared-identity components are not independent experiments.',
            'Native speeds retain original units; fixed bins are descriptive, not physical stopped/moving labels.',
            'Helper strata have different populations and do not establish causal benefit of helper input.',
            'Correlation/low error is insufficient to prove physical dynamics; all constants use only fixed zero, never fitted V statistics.',
            'Within-target centering uses truth only for diagnosis, never model input; adjacent differences never bridge invalid queries.',
            'Original measured observations and training-target statistics are unavailable in this cache; no T mean/median control is fabricated.'])
    return result,rows
