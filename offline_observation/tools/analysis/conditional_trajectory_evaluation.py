"""C70 evaluation only: fixed outputs enter before identities/references do.

``evaluate_scene`` returns a JSON-safe, identity-free row and a flat dictionary
of numeric error vectors suitable for ``np.savez_compressed``. No coordinates,
identity strings, model calls, files, or selection decisions leave this module.
States are [G,64,4] in latitude/longitude/mps/course order. Group rows follow
increasing minimum input-node index, independent of the membership label values.
``summarize`` pools actual point errors, and also reports every component and the
population excluding the largest component (by available reference positions).
TRUE_* are explicit reference-identity diagnostics, never deployed methods.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.association import evaluate_scenes, _aggregate
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.tmtaff import aggregate_fusion_reports


METHODS = ('WS_MEAN', 'WS_FRAGMENT', 'WS_GROUP', 'J_GROUP', 'HOLDOUT_GROUP',
           'J_ALL_GROUP', 'TRUE_MEAN', 'TRUE_GROUP')
ASSOCIATION_ARMS = dict(WS='WS_GROUP', J='J_GROUP', HOLDOUT='HOLDOUT_GROUP',
                        J_ALL='J_ALL_GROUP')
# Ordered (baseline, candidate); the first comparison is the primary estimand.
COMPARISONS = (('J_GROUP', 'HOLDOUT_GROUP'), ('WS_MEAN', 'WS_FRAGMENT'),
               ('WS_FRAGMENT', 'WS_GROUP'), ('WS_MEAN', 'WS_GROUP'),
               ('WS_GROUP', 'J_GROUP'), ('WS_GROUP', 'HOLDOUT_GROUP'),
               ('J_GROUP', 'J_ALL_GROUP'), ('TRUE_MEAN', 'TRUE_GROUP'))
PRIMARY_COMPARISON = 'HOLDOUT_GROUP_vs_J_GROUP'
REGIONS = ('all', 'observed', 'missing')
FEATURES = ('position', 'latitude', 'longitude', 'speed', 'heading')
UNITS = dict(position='metres', latitude='degrees', longitude='degrees',
             speed='mps', heading='degrees')
POPULATION_KEYS = ('target_identity_count', 'predicted_group_count',
                   'duplicate_prediction_group_count', 'anchor_matched_target_count',
                   'reference_unavailable_target_count')
PAIR_COUNTS = ('common_points', 'gained_points', 'lost_points', 'neither_points',
               'common_targets', 'gained_targets', 'lost_targets')


def _stats(values):
    values = np.asarray(values, np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('finite one-dimensional error vectors required')
    n = len(values)
    return dict(count=n, mae=float(values.mean()) if n else None,
                rmse=float(np.sqrt(np.mean(values**2))) if n else None,
                p95=float(np.quantile(values, .95)) if n else None,
                p99=float(np.quantile(values, .99)) if n else None,
                max=float(values.max()) if n else None)


def _groups(membership, node_count):
    member = np.asarray(membership)
    if (member.shape != (node_count,) or member.dtype.kind not in 'iu'
            or (member < 0).any()):
        raise ValueError('one nonnegative integer membership per original node required')
    groups = [np.flatnonzero(member == label) for label in np.unique(member)]
    return sorted(groups, key=lambda group: int(group[0]))


def _same_partition(a, b, pairs):
    return np.array_equal(a[pairs[:, 0]] == a[pairs[:, 1]],
                          b[pairs[:, 0]] == b[pairs[:, 1]])


def _vectors(artifact, target_count):
    pred = artifact['aligned_prediction'][:target_count]
    truth = artifact['aligned_truth'][:target_count]
    tv = artifact['aligned_truth_valid'][:target_count]
    pv = artifact['aligned_prediction_valid'][:target_count]
    errors = attribute_errors(pred, truth)
    errors['latitude'] = np.abs(pred[..., 0] - truth[..., 0])
    errors['longitude'] = np.abs((pred[..., 1] - truth[..., 1] + 180.) % 360. - 180.)
    # Match the physical evaluator's circular convention, including full turns.
    errors['heading'] = np.abs((pred[..., 3] % 360. - truth[..., 3] % 360. + 180.) % 360. - 180.)
    columns = dict(position=[0, 1], latitude=[0], longitude=[1], speed=[2], heading=[3])
    available = {key: tv[..., cols].all(-1) for key, cols in columns.items()}
    own = {key: available[key] & pv[..., cols].all(-1) for key, cols in columns.items()}
    observed = artifact['aligned_observed'][:target_count]
    regions = dict(all=np.ones(observed.shape, bool), observed=observed, missing=~observed)
    return errors, available, own, regions


def _pair_result(baseline, candidate, gained, lost, feature):
    baseline, candidate = np.asarray(baseline), np.asarray(candidate)
    if baseline.shape != candidate.shape:
        raise ValueError('paired common-population errors differ in shape')
    delta = candidate - baseline
    result = dict(unit=UNITS[feature], baseline=_stats(baseline), candidate=_stats(candidate),
                  delta_mae=float(delta.mean()) if len(delta) else None,
                  delta_rmse=(float(np.sqrt(np.mean(candidate**2)) -
                                    np.sqrt(np.mean(baseline**2))) if len(delta) else None),
                  improved_points=int((delta < -1e-9).sum()),
                  worsened_points=int((delta > 1e-9).sum()),
                  tied_points=int((np.abs(delta) <= 1e-9).sum()),
                  gained_candidate_errors=_stats(gained), lost_baseline_errors=_stats(lost))
    if feature == 'position':
        result['tail'] = dict(
            harm_over_1000m=int((delta > 1000.).sum()),
            rescue_over_1000m=int((delta < -1000.).sum()),
            newly_over_1000m=int(((baseline <= 1000.) & (candidate > 1000.)).sum()),
            no_longer_over_1000m=int(((baseline > 1000.) & (candidate <= 1000.)).sum()),
            max_harm_m=float(np.maximum(delta, 0.).max()) if len(delta) else None,
            harm_definition='candidate minus baseline error on common target-query points')
    return result


def evaluate_scene(sid, component, nodes, sources, pairs, support, nearest, ev, predictions):
    """Evaluate the eight already-fixed arms; no prediction or I/O is performed.

    ``ev`` has identities, target_ids, target_states, target_valid. Its optional
    node_ids is checked against nodes. Each prediction is (membership, states).
    The caller must bind Q64/query_seconds before calling: this API has no times.
    Association scopes are the complete same-/cross-source graph, NOT temporal
    paper eligibility masks (intervals are intentionally absent from this API).
    """
    nodes, sources, pairs = np.asarray(nodes), np.asarray(sources), np.asarray(pairs)
    n = len(nodes)
    if (nodes.ndim != 1 or len(np.unique(nodes)) != n or sources.shape != (n,)
            or pairs.dtype.kind not in 'iu'
            or not np.array_equal(pairs, np.column_stack(np.triu_indices(n, 1)))):
        raise ValueError('unique ordered nodes, sources and exact complete upper-triangle pairs required')
    if set(predictions) != set(METHODS):
        raise ValueError('exactly the fixed eight C70 arms are required')
    support, nearest = np.asarray(support), np.asarray(nearest)
    if (support.shape != (n, 64) or support.dtype != bool or nearest.shape != support.shape
            or not np.isfinite(nearest).all() or (nearest < 0).any()):
        raise ValueError('original boolean Q64 support and finite nonnegative nearest seconds required')
    if 'node_ids' in ev and not np.array_equal(nodes, ev['node_ids']):
        raise ValueError('reference/input node order differs')
    identities = np.asarray(ev['identities'])
    targets = np.asarray(ev['target_ids'])
    if identities.shape != (n,) or np.shape(ev['target_states']) != (len(targets), 64, 4):
        raise ValueError('reference node/target population and Q64 must match')
    outputs, memberships = {}, {}
    truth_pair = identities[pairs[:, 0]] == identities[pairs[:, 1]]
    for name in METHODS:
        membership, states = predictions[name]
        memberships[name] = np.asarray(membership)
        groups = _groups(membership, n)
        if np.shape(states) != (len(groups), 64, 4):
            raise ValueError(name + ': states must follow canonical group order and Q64')
        outputs[name] = emit_group_outputs(states, groups, node_count=n)
        if name.startswith('TRUE_') and not np.array_equal(
                memberships[name][pairs[:, 0]] == memberships[name][pairs[:, 1]], truth_pair):
            raise ValueError(name + ': diagnostic requires the complete true-identity partition')
    for name in ('WS_MEAN', 'WS_FRAGMENT'):
        if not _same_partition(memberships[name], memberships['WS_GROUP'], pairs):
            raise ValueError('WS controls must share exactly the same partition')
    reports, artifacts = evaluate_fusion_methods(outputs,
        prediction_speed_units={name: 'mps' for name in METHODS},
        node_identities=identities, target_ids=targets, target_states=ev['target_states'],
        target_valid=ev['target_valid'], support=support, nearest_seconds=nearest,
        target_speed_unit='mps', split='T_core')
    row = dict(scene_id=str(sid), component_id=str(component), node_count=n,
               methods={}, association={}, comparisons={},
               association_scope='complete all-pair same/cross-source; no temporal paper mask',
               true_arms='TRUE_MEAN/TRUE_GROUP use identity partitions, diagnostic only')
    arrays, vectors = {}, {}
    for name in METHODS:
        report = reports[name]
        item = dict(population={key: int(report[key]) for key in POPULATION_KEYS},
                    unmatched_target_count=len(report['unmatched_target_ids']),
                    diagnostic_only=name.startswith('TRUE_'),
                    full_target_metrics=report['full_target_metrics'],
                    paper_fusion_metrics=report['paper_fusion_metrics'], absolute_errors={})
        errors, available, own, regions = _vectors(artifacts[name], len(targets))
        vectors[name] = errors, available, own, regions
        for region in REGIONS:
            item['absolute_errors'][region] = {}
            for feature in FEATURES:
                mask = own[feature] & regions[region]
                values = np.asarray(errors[feature][mask], np.float64)
                key = f'{name}__{region}__{feature}'
                arrays[key] = values.copy()
                item['absolute_errors'][region][feature] = dict(_stats(values), unit=UNITS[feature])
        row['methods'][name] = item
    for front, name in ASSOCIATION_ARMS.items():
        co = memberships[name][pairs[:, 0]] == memberships[name][pairs[:, 1]]
        result = evaluate_scenes([dict(scene_id=str(sid), component_id=str(component),
            sources=sources, pairs=pairs, labels=truth_pair, scores=co.astype(float))], .5)
        row['association'][front] = {
            scope: (result if scope == 'all' else result[scope])['per_scene'][0]
            for scope in ('all', 'itsa', 'mstsa')}
    for base, candidate in COMPARISONS:
        key = candidate + '_vs_' + base
        row['comparisons'][key] = dict(baseline_method=base, candidate_method=candidate, regions={})
        be, available, bm, regions = vectors[base]
        ce, _, cm, _ = vectors[candidate]
        for region in REGIONS:
            result = {}
            for feature in FEATURES:
                b, c = bm[feature] & regions[region], cm[feature] & regions[region]
                common, gained, lost = b & c, c & ~b, b & ~c
                vals = (be[feature][common], ce[feature][common], ce[feature][gained], be[feature][lost])
                for label, value in zip(('baseline', 'candidate', 'gained', 'lost'), vals):
                    arrays[f'{key}__{region}__{feature}__{label}'] = np.asarray(value, np.float64).copy()
                counts = dict(common_points=int(common.sum()), gained_points=int(gained.sum()),
                    lost_points=int(lost.sum()),
                    neither_points=int((available[feature] & regions[region] & ~b & ~c).sum()),
                    common_targets=int((b.any(1) & c.any(1)).sum()),
                    gained_targets=int((~b.any(1) & c.any(1)).sum()),
                    lost_targets=int((b.any(1) & ~c.any(1)).sum()))
                result[feature] = dict(_pair_result(*vals, feature), **counts)
            row['comparisons'][key]['regions'][region] = result
    return row, arrays


def _pool(rows, arrays):
    def cat(key):
        return np.concatenate([np.asarray(a[key], np.float64) for a in arrays])
    result = dict(scene_count=len(rows), component_count=len({r['component_id'] for r in rows}),
                  node_count=sum(r['node_count'] for r in rows), methods={}, association={}, comparisons={})
    for name in METHODS:
        items = [row['methods'][name] for row in rows]
        physical = aggregate_fusion_reports(items)
        result['methods'][name] = dict(
            population={key: sum(item['population'][key] for item in items) for key in POPULATION_KEYS},
            unmatched_target_count=sum(item['unmatched_target_count'] for item in items),
            diagnostic_only=name.startswith('TRUE_'), physical=physical,
            absolute_errors={region: {feature: dict(_stats(cat(f'{name}__{region}__{feature}')),
                unit=UNITS[feature]) for feature in FEATURES} for region in REGIONS})
    for front in ASSOCIATION_ARMS:
        result['association'][front] = {scope: _aggregate([
            row['association'][front][scope] for row in rows]) for scope in ('all', 'itsa', 'mstsa')}
    for base, candidate in COMPARISONS:
        key = candidate + '_vs_' + base
        entry = dict(baseline_method=base, candidate_method=candidate, regions={})
        for region in REGIONS:
            entry['regions'][region] = {}
            for feature in FEATURES:
                vals = [cat(f'{key}__{region}__{feature}__{label}')
                        for label in ('baseline', 'candidate', 'gained', 'lost')]
                counts = {field: sum(row['comparisons'][key]['regions'][region][feature][field]
                                    for row in rows) for field in PAIR_COUNTS}
                value = dict(_pair_result(*vals, feature), **counts)
                deltas = [row['comparisons'][key]['regions'][region][feature]['delta_mae'] for row in rows]
                value['scene_directions'] = dict(improved=sum(x is not None and x < -1e-9 for x in deltas),
                    worsened=sum(x is not None and x > 1e-9 for x in deltas),
                    tied=sum(x is not None and abs(x) <= 1e-9 for x in deltas),
                    undefined=sum(x is None for x in deltas))
                entry['regions'][region][feature] = value
        result['comparisons'][key] = entry
    return result


def summarize(rows, arrays):
    """Pool matching scene rows/error dictionaries; no bootstrap or IID claim."""
    if not rows or len(rows) != len(arrays) or len({row['scene_id'] for row in rows}) != len(rows):
        raise ValueError('nonempty unique scene rows and one matching array dictionary per scene required')
    components = sorted({row['component_id'] for row in rows})
    by_component, sizes = {}, {}
    for component in components:
        indices = [i for i, row in enumerate(rows) if row['component_id'] == component]
        by_component[component] = _pool([rows[i] for i in indices], [arrays[i] for i in indices])
        sizes[component] = sum(rows[i]['methods']['WS_GROUP']['full_target_metrics']['all']['position']
                               ['available_target_count'] for i in indices)
    largest = min(components, key=lambda component: (-sizes[component], component))
    retained = [i for i, row in enumerate(rows) if row['component_id'] != largest]
    return dict(methods=list(METHODS), primary_comparison=PRIMARY_COMPARISON,
        all_components=_pool(rows, arrays), by_component=by_component,
        remove_largest_component=(dict(excluded_component=largest,
            criterion='available reference position points; ties lexical component id',
            result=_pool([rows[i] for i in retained], [arrays[i] for i in retained]))
            if retained else dict(excluded_component=largest, result=None)),
        limitations=['T development description, not independent testing or author-protocol comparison',
                     'association uses complete all-pair scopes, no temporal paper eligibility mask',
                     'TRUE_* use identity partitions for diagnostics only',
                     'absolute longitude/heading errors wrap; paper ME retains its separate raw-difference definition',
                     'point-pooled errors and scene/target-equal paper statistics are distinct',
                     'no confidence interval or new fitted selection is produced'])
