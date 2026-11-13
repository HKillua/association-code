"""C74 evaluation of frozen A/B/C outputs, with no prediction, fitting or I/O.

The public row contains no identities or coordinates. Flat arrays contain only
signed errors on unique target rows; duplicate outputs remain in the shared
physical evaluator's coverage accounting. Bias is prediction minus reference,
including shortest-arc heading error. Change rates use adjacent valid Q pairs,
actual seconds, and the region at the later query; they never bridge missing
references. Reference-derived near-stop/turn flags are evaluation only.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.tmtaff import aggregate_fusion_reports


METHODS = ('A', 'B', 'C')
COMPARISONS = (('B', 'C'), ('A', 'B'), ('A', 'C'))
REGIONS = ('all', 'active', 'internal_gap', 'near_endpoint',
           'reference_near_stop', 'reference_turn', 'active_no_low_speed_fallback')
FEATURES = ('speed', 'heading', 'speed_change_rate', 'heading_change_rate')
UNITS = dict(speed='mps', heading='degrees', speed_change_rate='mps_per_second',
             heading_change_rate='degrees_per_second')
POPULATION_KEYS = ('target_identity_count', 'predicted_group_count',
                   'duplicate_prediction_group_count', 'anchor_matched_target_count',
                   'reference_unavailable_target_count')
COVERAGE_KEYS = ('available_target_count', 'available_prediction_count', 'intersection_count',
                 'missing_prediction_count', 'prediction_without_target_count',
                 'available_target_trajectory_count', 'covered_target_trajectory_count',
                 'missing_target_trajectory_count')
COUNT_KEYS = ('available_reference_points', 'common_points', 'gained_points', 'lost_points',
              'neither_points', 'common_targets', 'gained_targets', 'lost_targets')


def _wrap(value):
    return (value + 180.) % 360. - 180.


def _stats(errors):
    errors = np.asarray(errors, np.float64)
    if errors.ndim != 1 or not np.isfinite(errors).all():
        raise ValueError('finite one-dimensional signed error vectors required')
    absolute = np.abs(errors)
    n = len(errors)
    return dict(count=n, mae=float(absolute.mean()) if n else None,
                rmse=float(np.sqrt(np.mean(errors**2))) if n else None,
                p95=float(np.quantile(absolute, .95)) if n else None,
                p99=float(np.quantile(absolute, .99)) if n else None,
                bias=float(errors.mean()) if n else None)


def _pair(base, candidate, counts):
    if base.shape != candidate.shape or counts['common_points'] != len(base):
        raise ValueError('paired error populations differ')
    b, c = _stats(base), _stats(candidate)
    delta = np.abs(candidate) - np.abs(base)
    return dict(baseline=b, candidate=c, **counts,
                coverage=(counts['common_points']/counts['available_reference_points']
                          if counts['available_reference_points'] else None),
                **{'delta_'+key: c[key]-b[key] if len(base) else None
                   for key in ('mae', 'rmse', 'p95', 'p99', 'bias')},
                improved_points=int((delta < -1e-9).sum()),
                worsened_points=int((delta > 1e-9).sum()),
                tied_points=int((np.abs(delta) <= 1e-9).sum()))


def _vectors(artifact, target_count, query):
    prediction = artifact['aligned_prediction'][:target_count]
    truth = artifact['aligned_truth'][:target_count]
    tv = artifact['aligned_truth_valid'][:target_count]
    pv = artifact['aligned_prediction_valid'][:target_count]
    errors, available, own = {}, {}, {}
    for feature, column in (('speed', 2), ('heading', 3)):
        raw = prediction[..., column] - truth[..., column]
        errors[feature] = _wrap(raw) if feature == 'heading' else raw
        available[feature] = tv[..., column]
        own[feature] = tv[..., column] & pv[..., column]
        key = feature+'_change_rate'
        pd, td = np.diff(prediction[..., column], axis=1), np.diff(truth[..., column], axis=1)
        if feature == 'heading':
            pd, td = _wrap(pd), _wrap(td)
        errors[key] = np.zeros(truth.shape[:2], np.float64)
        errors[key][:, 1:] = (np.abs(pd)-np.abs(td))/np.diff(query)
        available[key], own[key] = np.zeros(truth.shape[:2], bool), np.zeros(truth.shape[:2], bool)
        available[key][:, 1:] = available[feature][:, :-1] & available[feature][:, 1:]
        own[key][:, 1:] = own[feature][:, :-1] & own[feature][:, 1:]
    return errors, available, own


def evaluate_scene(sid, component, nodes, groups, observation_support, nearest,
                   ev, predictions, group_masks, query_seconds):
    """Return one public row and flat errors for already-frozen [G,64,4] arms.

    Required group masks are active/internal_gap/near_endpoint. Optional
    B_low_speed and C_low_speed must be supplied together and lie within active;
    their absence is reported, not treated as evidence of no fallback.
    """
    nodes, query = np.asarray(nodes), np.asarray(query_seconds, np.float64)
    n, g = len(nodes), len(groups)
    support, nearest = np.asarray(observation_support), np.asarray(nearest)
    if (nodes.ndim != 1 or len(np.unique(nodes)) != n or query.shape != (64,)
            or not np.isfinite(query).all() or (np.diff(query) <= 0).any()
            or support.shape != (n, 64) or support.dtype != bool or nearest.shape != support.shape
            or not np.isfinite(nearest).all() or (nearest < 0).any()):
        raise ValueError('unique nodes, increasing physical Q64 and original support/nearest required')
    if set(predictions) != set(METHODS):
        raise ValueError('exactly the fixed A/B/C predictions required')
    for key in ('node_ids', 'query_seconds'):
        expected = nodes if key == 'node_ids' else query
        if key in ev and not np.array_equal(ev[key], expected):
            raise ValueError('evaluation '+key+' binding differs')
    identities, targets = np.asarray(ev['identities']), np.asarray(ev['target_ids'])
    if (identities.shape != (n,) or targets.ndim != 1
            or np.shape(ev['target_states']) != (len(targets), 64, 4)
            or np.shape(ev['target_valid']) != (len(targets), 64, 4)
            or np.asarray(ev['target_valid']).dtype != bool):
        raise ValueError('original reference population and Q64 masks required')
    required = {'active', 'internal_gap', 'near_endpoint'}
    fallback_keys = {'B_low_speed', 'C_low_speed'}
    keys = set(group_masks)
    if not required <= keys or keys - required not in (set(), fallback_keys):
        raise ValueError('fixed group mask names, with both optional fallback masks, required')
    masks = {key: np.asarray(value) for key, value in group_masks.items()}
    if any(value.shape != (g, 64) or value.dtype != bool for value in masks.values()):
        raise ValueError('group masks must be boolean [G,64]')
    fallback_known = fallback_keys <= keys
    if fallback_known and any((masks[key] & ~masks['active']).any() for key in fallback_keys):
        raise ValueError('low-speed fallback masks must be within active')
    outputs = {}
    for name in METHODS:
        states = np.asarray(predictions[name])
        if states.shape != (g, 64, 4) or not np.isfinite(states).all():
            raise ValueError(name+': finite states[G,64,4] required')
        if not np.array_equal(states[..., :2], np.asarray(predictions['A'])[..., :2]):
            raise ValueError('frozen position outputs differ')
        outputs[name] = emit_group_outputs(states, groups, node_count=n)
    reports, artifacts = evaluate_fusion_methods(outputs,
        prediction_speed_units={name: 'mps' for name in METHODS}, node_identities=identities,
        target_ids=targets, target_states=ev['target_states'], target_valid=ev['target_valid'],
        support=support, nearest_seconds=nearest, target_speed_unit='mps', split='V_select')
    baseline = artifacts['A']
    for name in METHODS[1:]:
        artifact = artifacts[name]
        for key in ('aligned_prediction_valid', 'aligned_truth_valid', 'aligned_group_indices',
                    'aligned_target_ids', 'group_offsets', 'group_node_indices',
                    'group_anchor_target_indices', 'aligned_observed'):
            if not np.array_equal(artifact[key], baseline[key]):
                raise ValueError(name+': common correspondence or mask changed: '+key)
        if not np.array_equal(artifact['aligned_prediction'][..., :2],
                              baseline['aligned_prediction'][..., :2], equal_nan=True):
            raise ValueError(name+': aligned position output changed')
        for key in POPULATION_KEYS + ('unmatched_target_ids',):
            if reports[name][key] != reports['A'][key]:
                raise ValueError(name+': coverage or duplicates changed')
        for region in ('all', 'observed', 'missing'):
            for feature in ('position', 'speed', 'heading'):
                for key in COVERAGE_KEYS:
                    if (reports[name]['full_target_metrics'][region][feature][key]
                            != reports['A']['full_target_metrics'][region][feature][key]):
                        raise ValueError(name+': physical feature coverage changed')
    aligned = baseline['aligned_group_indices'][:len(targets)]
    matched = aligned >= 0
    regions = dict(all=np.ones((len(targets), 64), bool))
    for key in required:
        regions[key] = np.zeros((len(targets), 64), bool)
        regions[key][matched] = masks[key][aligned[matched]]
    truth, valid = np.asarray(ev['target_states']), np.asarray(ev['target_valid'])
    regions['reference_near_stop'] = valid[..., 2] & (truth[..., 2] < .5)
    turn = np.zeros((len(targets), 64), bool)
    eligible = valid[..., 2:4].all(-1) & (truth[..., 2] >= .5)
    turn[:, 1:] = (eligible[:, :-1] & eligible[:, 1:]
                   & (np.abs(_wrap(np.diff(truth[..., 3], axis=1))) >= 15.))
    regions['reference_turn'] = turn
    regions['active_no_low_speed_fallback'] = np.zeros((len(targets), 64), bool)
    fallback_counts = None
    if fallback_known:
        b, c = masks['B_low_speed'], masks['C_low_speed']
        no_fallback = masks['active'] & ~(b | c)
        regions['active_no_low_speed_fallback'][matched] = no_fallback[aligned[matched]]
        fallback_counts = dict(B=int(b.sum()), C=int(c.sum()), both=int((b & c).sum()),
            either=int((b | c).sum()), B_only=int((b & ~c).sum()), C_only=int((c & ~b).sum()),
            neither_active=int(no_fallback.sum()))
    row = dict(scene_id=str(sid), component_id=str(component), node_count=n, group_count=g,
        invariants=dict(position_exact=True, masks_exact=True, correspondence_exact=True,
                        coverage_exact=True, duplicates_exact=True),
        group_query_counts={key: int(masks[key].sum()) for key in sorted(required)},
        fallback_masks_supplied=fallback_known, fallback_group_query_counts=fallback_counts,
        reference_region_counts={key: int(regions[key].sum())
                                 for key in ('reference_near_stop', 'reference_turn')},
        methods={}, comparisons={})
    arrays, vectors = {}, {}
    for name in METHODS:
        report = reports[name]
        errors, available, own = _vectors(artifacts[name], len(targets), query)
        vectors[name] = errors, available, own
        item = dict(population={key: int(report[key]) for key in POPULATION_KEYS},
            unmatched_target_count=len(report['unmatched_target_ids']),
            full_target_metrics=report['full_target_metrics'],
            paper_fusion_metrics=report['paper_fusion_metrics'], errors={})
        for region in REGIONS:
            item['errors'][region] = {}
            for feature in FEATURES:
                values = errors[feature][own[feature] & regions[region]].copy()
                arrays[f'{name}__{region}__{feature}'] = values
                item['errors'][region][feature] = dict(_stats(values), unit=UNITS[feature])
        row['methods'][name] = item
    for base, candidate in COMPARISONS:
        key = candidate+'_vs_'+base
        entry = dict(baseline_method=base, candidate_method=candidate, regions={})
        _, available, bm = vectors[base]
        _, _, cm = vectors[candidate]
        for region in REGIONS:
            entry['regions'][region] = {}
            for feature in FEATURES:
                b, c = bm[feature] & regions[region], cm[feature] & regions[region]
                counts = dict(available_reference_points=int((available[feature] & regions[region]).sum()),
                    common_points=int((b & c).sum()), gained_points=int((c & ~b).sum()),
                    lost_points=int((b & ~c).sum()),
                    neither_points=int((available[feature] & regions[region] & ~(b | c)).sum()),
                    common_targets=int((b.any(1) & c.any(1)).sum()),
                    gained_targets=int((~b.any(1) & c.any(1)).sum()),
                    lost_targets=int((b.any(1) & ~c.any(1)).sum()))
                if counts['gained_points'] or counts['lost_points']:
                    raise ValueError('fixed population gained/lost points')
                entry['regions'][region][feature] = dict(_pair(
                    arrays[f'{base}__{region}__{feature}'],
                    arrays[f'{candidate}__{region}__{feature}'], counts), unit=UNITS[feature])
        row['comparisons'][key] = entry
    return row, arrays


def _pool(rows, arrays):
    def cat(name, region, feature):
        return np.concatenate([a[f'{name}__{region}__{feature}'] for a in arrays])
    result = dict(scene_count=len(rows), scene_ids=[r['scene_id'] for r in rows],
        component_count=len({r['component_id'] for r in rows}),
        node_count=sum(r['node_count'] for r in rows), group_count=sum(r['group_count'] for r in rows),
        invariants={key: all(r['invariants'][key] for r in rows) for key in rows[0]['invariants']},
        group_query_counts={key: sum(r['group_query_counts'][key] for r in rows)
                            for key in rows[0]['group_query_counts']},
        fallback_masks_supplied=rows[0]['fallback_masks_supplied'],
        fallback_group_query_counts=None, methods={}, comparisons={})
    if result['fallback_masks_supplied']:
        result['fallback_group_query_counts'] = {
            key: sum(r['fallback_group_query_counts'][key] for r in rows)
            for key in rows[0]['fallback_group_query_counts']}
    result['reference_region_counts'] = {
        key: sum(r['reference_region_counts'][key] for r in rows)
        for key in rows[0]['reference_region_counts']}
    for name in METHODS:
        items = [r['methods'][name] for r in rows]
        result['methods'][name] = dict(
            population={key: sum(item['population'][key] for item in items) for key in POPULATION_KEYS},
            unmatched_target_count=sum(item['unmatched_target_count'] for item in items),
            physical=aggregate_fusion_reports(items),
            errors={region: {feature: dict(_stats(cat(name, region, feature)), unit=UNITS[feature])
                             for feature in FEATURES} for region in REGIONS})
    for base, candidate in COMPARISONS:
        key = candidate+'_vs_'+base
        entry = dict(baseline_method=base, candidate_method=candidate, regions={})
        for region in REGIONS:
            entry['regions'][region] = {}
            for feature in FEATURES:
                values = [r['comparisons'][key]['regions'][region][feature] for r in rows]
                counts = {field: sum(value[field] for value in values) for field in COUNT_KEYS}
                paired = _pair(cat(base, region, feature), cat(candidate, region, feature), counts)
                deltas = [value['delta_mae'] for value in values]
                paired['scene_directions'] = dict(
                    improved=sum(d is not None and d < -1e-9 for d in deltas),
                    worsened=sum(d is not None and d > 1e-9 for d in deltas),
                    tied=sum(d is not None and abs(d) <= 1e-9 for d in deltas),
                    undefined=sum(d is None for d in deltas))
                entry['regions'][region][feature] = dict(paired, unit=UNITS[feature])
        result['comparisons'][key] = entry
    return result


def summarize(rows, arrays):
    """Pool point errors, every component, and exclusion of the largest component."""
    if (not rows or len(rows) != len(arrays) or len({r['scene_id'] for r in rows}) != len(rows)
            or len({r['fallback_masks_supplied'] for r in rows}) != 1):
        raise ValueError('unique nonempty scenes with consistent fallback masks and paired arrays required')
    components = sorted({r['component_id'] for r in rows})
    grouped, sizes = {}, {}
    for component in components:
        selected = [i for i, row in enumerate(rows) if row['component_id'] == component]
        grouped[component] = _pool([rows[i] for i in selected], [arrays[i] for i in selected])
        sizes[component] = sum(rows[i]['methods']['A']['full_target_metrics']['all']['position']
                               ['available_target_count'] for i in selected)
    largest = min(components, key=lambda component: (-sizes[component], component))
    keep = [i for i, row in enumerate(rows) if row['component_id'] != largest]
    return dict(methods=list(METHODS), primary_comparison='C_vs_B', primary_region='all',
        all_components=_pool(rows, arrays), by_component=grouped,
        remove_largest_component=dict(excluded_component=largest,
            criterion='available reference position points; ties lexical component id',
            result=_pool([rows[i] for i in keep], [arrays[i] for i in keep]) if keep else None),
        definitions=dict(bias='prediction minus reference; heading shortest arc in [-180,180)',
            reference_turn='adjacent Q64 valid reference speed/course; both speeds >=0.5mps; circular change >=15deg',
            reference_near_stop='valid reference speed <0.5mps',
            change_rates='absolute adjacent speed/course change per physical second; region at later query; no missing-reference bridging',
            active_no_low_speed_fallback='active and neither B nor C low-speed fallback; undefined when optional masks absent',
            error_population='unique target rows; duplicates retained separately in shared full physical coverage'),
        limitations=['reused development V32, not independent testing or author reproduction',
                     'point-pooled errors and scene/target-equal paper metrics are distinct',
                     'reference speed need not equal the derivative of reference positions',
                     'reference strata are evaluation-only; no fit, selection or confidence interval'])
