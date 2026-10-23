"""080 fixed-score population comparison; no model, raw input or optimization.

T/V balanced views share eligibility, identity caps and negative availability
rules, not individual samples or random draw order. Standardized-prior AP is an
algebraic reference holding empirical class-conditional scores fixed, not a new
operating point, calibrated probability or observed deployment performance.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from association.evaluation.tsactfer_input_probe import (
    SCOPE as SOURCE_SCOPE, _metrics as original_metrics, _natural_eligible)
from association.evaluation.tsactfer_score_diagnostics import (
    SCOPES, _masks, _metrics as score_metrics, _quantiles, _validate)
from baselines.tsactfer_data import build_positive_pools

SCOPE = '080-tsa-population-probe-v1'
DEFAULT = dict(scope=SCOPE, seed=20261024, scene_count=32, pairs_per_head_scene=32,
               advisory_seconds=3600)
MARGIN_TOLERANCE = 1e-7
STANDARD_PRIOR = .5
TASKS = ('ITSA', 'MSTSA')
DOMAINS = ('same_scene', 'cross_scene')
MODES = ('full', 'train_crop')


def validate_config(config):
    if set(config) != set(DEFAULT) or config['scope'] != SCOPE:
        raise ValueError('080 configuration fields/scope differ')
    for key in ('seed', 'scene_count', 'pairs_per_head_scene', 'advisory_seconds'):
        if type(config[key]) is not int or config[key] < (0 if key == 'seed' else 1):
            raise ValueError('invalid080 '+key)
    if not 2 <= config['scene_count'] <= 32 or not 2 <= config['pairs_per_head_scene'] <= 32:
        raise ValueError('080 has a fixed finite cap of32 scenes and32 identities')
    return deepcopy(config)


def select_validation_roster(manifest, config=DEFAULT):
    validate_config(config)
    splits = {name: list(map(str, manifest['splits'][name])) for name in ('T_core', 'V_select', 'T_test')}
    if any(len(ids) != len(set(ids)) for ids in splits.values()):
        raise ValueError('080 split rosters must be distinct')
    if (set(splits['T_core']) & set(splits['V_select']) or set(splits['T_test']) &
            (set(splits['T_core']) | set(splits['V_select'])) or len(splits['V_select']) < config['scene_count']):
        raise ValueError('080 requires disjoint split scene lists and enoughV')
    indices = np.sort(np.random.default_rng(config['seed']).choice(
        len(splits['V_select']), config['scene_count'], replace=False))
    return [splits['V_select'][int(i)] for i in indices]


def standardized_ap(scores, labels, prior=STANDARD_PRIOR):
    """Tie-group AP at a declared prior; undefined if either class is absent."""
    scores, labels = np.asarray(scores, np.float64), np.asarray(labels)
    if (scores.ndim != 1 or labels.shape != scores.shape or not np.isfinite(scores).all()
            or not np.isin(labels, [0, 1]).all() or isinstance(prior, bool)
            or not np.isfinite(prior) or not 0 < prior < 1):
        raise ValueError('standardized AP requires finite scores, binary labels and interior prior')
    labels = labels.astype(bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    order = np.argsort(-scores, kind='stable')
    values, truth = scores[order], labels[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1]
    ends = np.r_[starts[1:], len(values)]
    pos = np.add.reduceat(truth.astype(np.int64), starts)
    neg = ends-starts-pos
    tpr, fpr = np.cumsum(pos)/positive, np.cumsum(neg)/negative
    precision = prior*tpr/(prior*tpr+(1-prior)*fpr)
    return float(np.sum((pos/positive)*precision))


def metrics(scores, labels, threshold):
    scores, labels = np.asarray(scores), np.asarray(labels, bool)
    result = score_metrics(scores, labels, threshold)
    result['average_precision_prior05'] = standardized_ap(scores, labels)
    return result


def _paired_metrics(positive, negative, threshold):
    return metrics(np.r_[positive, negative], np.r_[np.ones(len(positive), bool),
                                                  np.zeros(len(negative), bool)], threshold)


def pair_score(row, left, right):
    """Lookup in the original complete upper-triangle cache, in either direction."""
    n = len(row['node_ids'])
    if (isinstance(left, bool) or isinstance(right, bool) or int(left) != left or int(right) != right
            or left == right or min(left, right) < 0 or max(left, right) >= n):
        raise ValueError('080 invalid distinct node pair')
    i, j = sorted((int(left), int(right)))
    index = i*(2*n-i-1)//2+j-i-1
    if (np.asarray(row['pairs']).shape != (n*(n-1)//2, 2)
            or index >= len(row['scores']) or not np.array_equal(row['pairs'][index], [i, j])):
        raise ValueError('080 complete score lookup population differs')
    return float(row['scores'][index]), int(index)


def _same(actual, expected, path):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(path+' fields differ')
        for key in expected:
            _same(actual[key], expected[key], path+'/'+key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(path+' list differs')
        for i, (a, b) in enumerate(zip(actual, expected)):
            _same(a, b, path+'/'+str(i))
    elif isinstance(expected, float):
        if (isinstance(actual, bool) or not isinstance(actual, (float, int)) or not np.isfinite(actual)
                or not np.isclose(actual, expected, rtol=0., atol=1e-12)):
            raise ValueError(path+' numeric value differs')
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(path+' value differs')


def _candidates(rows, index, task, domain):
    left = rows[index]
    return [j for j, right in enumerate(rows) if right['identity'] != left['identity']
        and ((right['scene_id'] == left['scene_id']) == (domain == 'same_scene'))
        and ((right['right_source'] == left['left_source']) == (task == 'ITSA'))]


def _validate_view(row, task, manifest, allowed):
    sid = str(row['scene_id'])
    if sid not in allowed or str(row['component_id']) != str(manifest['components'][sid]):
        raise ValueError('080079 view scene/component differs')
    if (not isinstance(row['identity'], str) or not row['identity'] or row['left_node'] == row['right_node']
            or any(type(row[key]) is not int or row[key] < 0 for key in ('left_index', 'right_index'))
            or row['left_index'] == row['right_index']):
        raise ValueError('080079 positive view identity/node differs')
    for key in ('left_interval', 'right_interval', 'crop_left_interval', 'crop_right_interval'):
        value = np.asarray(row[key])
        if value.shape != (2,) or not np.isfinite(value).all() or value[1] < value[0]:
            raise ValueError('080079 view interval differs')
    if not _natural_eligible(row, row, task):
        raise ValueError('080079 positive is outside original task eligibility')
    crop = dict(row, left_interval=row['crop_left_interval'], right_interval=row['crop_right_interval'])
    if row['crop_retains_task_eligibility'] != _natural_eligible(crop, crop, task):
        raise ValueError('080079 cropped eligibility differs')


def validate_079(summary, saved, manifest):
    """Recompute existing079 summaries from immutable saved pairs and mappings."""
    if (summary.get('scope') != SOURCE_SCOPE or summary.get('status') != 'COMPLETED'
            or summary.get('frozen_weights_unchanged') is not True
            or summary.get('optimizer_updates') != 0 or summary.get('model_training') is not False
            or summary.get('threshold_selected') is not False or summary.get('groups_recomputed') is not False
            or set(saved) != set(TASKS) or set(summary['tasks']) != set(TASKS)):
        raise ValueError('080 requires completed frozen079')
    roster = list(map(str, summary['scene_ids']))
    if (len(roster) != len(set(roster)) or not set(roster) <= set(map(str, manifest['splits']['T_core']))):
        raise ValueError('080079 source roster must remain T_core')
    verified = {}
    for task in TASKS:
        rows, arrays, report = saved[task]['rows'], saved[task]['arrays'], summary['tasks'][task]
        for row in rows:
            _validate_view(row, task, manifest, set(roster))
        if len({(r['scene_id'], r['identity']) for r in rows}) != len(rows):
            raise ValueError('080079 repeated selected scene identity')
        mapping = np.asarray(arrays['mapping'])
        if (mapping.ndim != 2 or mapping.shape[1] != 3 or not np.issubdtype(mapping.dtype, np.integer)
                or (mapping < 0).any() or (mapping >= len(rows)).any()
                or len(set(mapping[:, 0])) != len(mapping)):
            raise ValueError('080079 mapping shape/index differs')
        excluded, expected_anchors = [], []
        for i, row in enumerate(rows):
            pools = {domain: _candidates(rows, i, task, domain) for domain in DOMAINS}
            if not all(pools.values()):
                excluded.append(dict(view_index=i, scene_id=row['scene_id'], reason=[
                    d+'_no_other_identity_matching_head_source' for d in DOMAINS if not pools[d]]))
            else:
                expected_anchors.append(i)
        if not np.array_equal(mapping[:, 0], expected_anchors):
            raise ValueError('080079 mapped anchor population differs')
        for i, same, cross in mapping:
            if same not in _candidates(rows, i, task, 'same_scene') or cross not in _candidates(rows, i, task, 'cross_scene'):
                raise ValueError('080079 negative source/global identity/domain differs')
        _same(excluded, report['excluded_anchors'], '080079 excluded anchors')
        for name, expected in dict(selected_positive_views=len(rows), paired_anchors=len(mapping),
            cropped_positive_eligible_count=sum(r['crop_retains_task_eligibility'] for r in rows),
            unique_same_negative_views=len(set(mapping[:, 1])), unique_cross_negative_views=len(set(mapping[:, 2])),
            available=bool(len(mapping)), global_identity_negative_exclusion=True).items():
            _same(report[name], expected, '080079 '+name)
        for domain_index, domain in enumerate(DOMAINS, 1):
            eligibility = np.array([_natural_eligible(rows[int(i)], rows[int(j)], task)
                for i, j in zip(mapping[:, 0], mapping[:, domain_index])], bool)
            if not np.array_equal(arrays[domain+'_negative_natural_eligible'], eligibility):
                raise ValueError('080079 negative eligibility differs')
            modes = {}
            for mode in MODES:
                key = domain+'__'+mode
                p, n = [np.asarray(arrays[key+'_'+sign]) for sign in ('positive', 'negative')]
                if (p.shape != (len(mapping),) or n.shape != p.shape or not np.isfinite(p).all()
                        or not np.isfinite(n).all() or (np.abs(p) > 1.00001).any() or (np.abs(n) > 1.00001).any()):
                    raise ValueError('080079 score shape/finite/cosine range differs')
                result = report['conditions'][key]
                _same(original_metrics(p, n), result['pooled'], '080079 pooled '+key)
                _same(original_metrics(p[eligibility], n[eligibility]), result['natural_eligible_paired_subset'],
                      '080079 eligible '+key)
                _same(int(eligibility.sum()), result['natural_eligible_negative_count'], '080079 eligible count')
                scene_ids = [rows[int(i)]['scene_id'] for i in mapping[:, 0]]
                scene_metrics = {sid: original_metrics(p[np.asarray(scene_ids) == sid], n[np.asarray(scene_ids) == sid])
                                 for sid in sorted(set(scene_ids))}
                _same(scene_metrics, result['scenes'], '080079 scene metrics '+key)
                modes[mode] = (p, n)
            p, n = modes['full']; cp, cn = modes['train_crop']
            difference = dict(positive_crop_minus_full=_quantiles(cp-p), negative_crop_minus_full=_quantiles(cn-n),
                margin_crop_minus_full=_quantiles((cp-cn)-(p-n)),
                mean_margin_full=float(np.mean(p-n)) if len(p) else None,
                mean_margin_crop=float(np.mean(cp-cn)) if len(p) else None)
            _same(difference, report['paired_crop_differences'][domain], '080079 paired differences')
        for mode in MODES:
            if not np.array_equal(arrays['same_scene__'+mode+'_positive'], arrays['cross_scene__'+mode+'_positive']):
                raise ValueError('080079 shared positive scores differ')
        verified[task] = dict(source_rows=len(rows), paired_anchors=len(mapping), excluded_anchors=excluded,
            cross_scene_availability_exclusions=sum(any('cross_scene' in reason for reason in row['reason'])
                                                  for row in excluded), all_saved_metrics_reproduced=True)
    return verified


def _balanced_report(positive, negative, anchors, roster, components, threshold):
    p, n = np.asarray(positive), np.asarray(negative)
    if p.shape != n.shape or p.shape != (len(anchors),):
        raise ValueError('080 balanced scores/population differ')
    by_scene = {sid: dict(scene_id=sid, component_id=str(components[sid]),
        metrics=_paired_metrics(p[mask], n[mask], threshold))
        for sid in roster for mask in [np.array([str(r['scene_id']) == sid for r in anchors], bool)]}
    by_component = {cid: dict(component_id=cid, scene_ids=[sid for sid in roster if str(components[sid]) == cid],
        metrics=_paired_metrics(p[mask], n[mask], threshold))
        for cid in sorted({str(components[sid]) for sid in roster})
        for mask in [np.array([str(r['component_id']) == cid for r in anchors], bool)]}
    return dict(pooled=_paired_metrics(p, n, threshold), scenes=by_scene, components=by_component,
                scene_count=len(roster), component_count=len(by_component))


def _margin_statistics(p, n, cp, cn):
    p, n, cp, cn = [np.asarray(x, np.float64) for x in (p, n, cp, cn)]
    positive, negative = cp-p, cn-n
    margin = positive-negative
    if not len(margin):
        return dict(count=0, mean_positive_change=None, mean_negative_change=None, mean_margin_change=None,
            mean_decomposition_residual=None, improvement_count=0, deterioration_count=0, tie_count=0,
            margin_change=_quantiles(margin))
    pmean, nmean, mmean = map(float, (positive.mean(), negative.mean(), margin.mean()))
    residual = mmean-(pmean-nmean)
    if abs(residual) > 1e-12:
        raise ValueError('080 mean margin decomposition fails')
    return dict(count=len(margin), mean_positive_change=pmean, mean_negative_change=nmean,
        mean_margin_change=mmean, mean_decomposition_residual=residual,
        improvement_count=int((margin > MARGIN_TOLERANCE).sum()),
        deterioration_count=int((margin < -MARGIN_TOLERANCE).sum()),
        tie_count=int((np.abs(margin) <= MARGIN_TOLERANCE).sum()), margin_change=_quantiles(margin))


def source_margin_decomposition(saved, roster, components):
    result = {}
    for task in TASKS:
        rows, arrays = saved[task]['rows'], saved[task]['arrays']
        anchors = [rows[int(i)] for i in arrays['mapping'][:, 0]]
        result[task] = {}
        for domain in DOMAINS:
            values = [arrays[domain+'__'+mode+'_'+sign] for mode in MODES for sign in ('positive', 'negative')]
            scenes = {sid: _margin_statistics(*[np.asarray(x)[mask] for x in values]) for sid in roster
                      for mask in [np.array([r['scene_id'] == sid for r in anchors], bool)]}
            component = {cid: _margin_statistics(*[np.asarray(x)[mask] for x in values])
                         for cid in sorted({str(components[sid]) for sid in roster})
                         for mask in [np.array([str(r['component_id']) == cid for r in anchors], bool)]}
            result[task][domain] = dict(pooled=_margin_statistics(*values), scenes=scenes, components=component,
                numerical_tie_tolerance=MARGIN_TOLERANCE, tolerance_is_scientific_effect_threshold=False)
    return result


def validation_balanced(records, config, threshold, roster, components):
    by_id = {row['scene_id']: row for row in records}
    reports, selections = {}, {}
    for task_index, task in enumerate(TASKS):
        select_rng = np.random.default_rng(np.random.SeedSequence([config['seed'], task_index, 1]))
        negative_rng = np.random.default_rng(np.random.SeedSequence([config['seed'], task_index, 3]))
        views, audit = [], []
        for sid in roster:
            row = by_id[sid]
            pools, stats = build_positive_pools([row]); available = sorted(pools[task])
            count = min(len(available), config['pairs_per_head_scene'])
            audit.append(dict(scene_id=sid, eligible_identities=len(available),
                eligible_pairs=stats['eligible_pair_counts'][task], selected_identities=count if count >= 2 else 0,
                excluded_reason='fewer_than_two_eligible_identities' if count < 2 else None))
            if count < 2:
                continue
            chosen = select_rng.choice(len(available), count, replace=False)
            for identity in [available[int(i)] for i in chosen]:
                choices = pools[task][identity]
                _, left, right = choices[int(select_rng.integers(len(choices)))]
                score, index = pair_score(row, left, right)
                views.append(dict(scene_id=sid, component_id=row['component_id'], identity=identity,
                    left_node=str(row['node_ids'][left]), right_node=str(row['node_ids'][right]),
                    left_index=int(left), right_index=int(right), left_source=str(row['sources'][left]),
                    right_source=str(row['sources'][right]), left_interval=row['intervals'][left].tolist(),
                    right_interval=row['intervals'][right].tolist(), positive_pair_index=index, positive_score=score))
        paired, excluded, positive, negative, anchors = [], [], [], [], []
        for i, view in enumerate(views):
            pools = {domain: _candidates(views, i, task, domain) for domain in DOMAINS}
            if not all(pools.values()):
                excluded.append(dict(view_index=i, scene_id=view['scene_id'], reason=[
                    d+'_no_other_identity_matching_head_source' for d in DOMAINS if not pools[d]]))
                continue
            chosen = int(negative_rng.choice(pools['same_scene']))
            score, pair_index = pair_score(by_id[view['scene_id']], view['left_index'], views[chosen]['right_index'])
            paired.append(dict(anchor_view_index=i, negative_view_index=chosen, negative_pair_index=pair_index,
                cross_scene_candidates_exist=True, negative_natural_eligible=_natural_eligible(view, views[chosen], task)))
            anchors.append(view); positive.append(view['positive_score']); negative.append(score)
        report = _balanced_report(positive, negative, anchors, roster, components, threshold)
        report.update(selected_positive_views=len(views), paired_anchors=len(paired), excluded_anchors=excluded,
            sampling=audit, unique_same_negative_views=len({r['negative_view_index'] for r in paired}),
            cross_scene_availability_exclusions=sum(any('cross_scene' in reason for reason in row['reason'])
                                                  for row in excluded),
            same_negative_natural_eligible_count=sum(row['negative_natural_eligible'] for row in paired))
        reports[task] = report
        selections[task] = dict(rows=views, mapping=paired)
    return reports, selections


def full_candidate_report(records, threshold):
    def summarize(rows):
        masks = [_masks(r) for r in rows]
        return {scope: metrics(np.concatenate([r['scores'][m[scope]] for r, m in zip(rows, masks)]),
            np.concatenate([r['labels'][m[scope]] for r, m in zip(rows, masks)]), threshold) for scope in SCOPES}
    scenes = {r['scene_id']: dict(scene_id=r['scene_id'], component_id=r['component_id'],
        node_count=len(r['node_ids']), scopes=summarize([r])) for r in records}
    components = {cid: dict(component_id=cid, scene_ids=[r['scene_id'] for r in records if r['component_id'] == cid],
        scopes=summarize([r for r in records if r['component_id'] == cid]))
        for cid in sorted({r['component_id'] for r in records})}
    return dict(scene_count=len(records), node_count=sum(len(r['node_ids']) for r in records),
                pooled=summarize(records), scenes=scenes, components=components)


def analyze_population(records, manifest, source079_summary, source079_saved, threshold, config=DEFAULT):
    config = validate_config(config)
    if threshold is not None and (isinstance(threshold, bool) or not np.isfinite(threshold)):
        raise ValueError('080 fixed threshold must be finite orNone')
    roster = select_validation_roster(manifest, config)
    # Require complete frozen V population even though only predeclared32 enter
    # this analysis; the CLI validates the score inventory against074D as well.
    seen, validated = set(), []
    for record in records:
        row = _validate(record, seen)
        if not np.array_equal(row['pairs'], np.column_stack(np.triu_indices(len(row['node_ids']), 1))):
            raise ValueError('080 requires complete original candidate population')
        if row['component_id'] != str(manifest['components'][row['scene_id']]):
            raise ValueError('080 validation component differs')
        validated.append(row)
    if [r['scene_id'] for r in validated] != list(map(str, manifest['splits']['V_select'])):
        raise ValueError('080 requires complete frozen V_select order')
    verification = validate_079(source079_summary, source079_saved, manifest)
    chosen = [row for row in validated if row['scene_id'] in set(roster)]
    if [row['scene_id'] for row in chosen] != roster:
        raise ValueError('080 selected validation order differs')
    balanced, selections = validation_balanced(chosen, config, threshold, roster, manifest['components'])
    t_roster = list(map(str, source079_summary['scene_ids']))
    train = {}
    for task in TASKS:
        item = source079_saved[task]
        anchors = [item['rows'][int(i)] for i in item['arrays']['mapping'][:, 0]]
        train[task] = _balanced_report(item['arrays']['same_scene__full_positive'],
            item['arrays']['same_scene__full_negative'], anchors, t_roster, manifest['components'], threshold)
    return dict(scope=SCOPE, configuration=config, threshold=threshold, threshold_reselected=False,
        model_forwards=0, optimizer_updates=0, raw_archives_read=False, checkpoints_opened=False,
        groups_recomputed=False, test_opened=False, train_scene_ids=t_roster, validation_scene_ids=roster,
        source079_verification=verification, train_balanced_full=train, validation_balanced_full=balanced,
        validation_full_candidates=full_candidate_report(chosen, threshold),
        source079_margin_decomposition=source_margin_decomposition(source079_saved, t_roster, manifest['components']),
        protocol=dict(standardized_prior=STANDARD_PRIOR, standardized_ap_is_observed_AP=False,
            standardized_ap_definition='Tie-group integral of precision at prior0.5 over empirical positive recall; both classes required.',
            negative_pool='Other globally distinct identity right views among selected eligible positives, matching head source relation.',
            eligibility_population='Anchors require same_scene and cross_scene negative availability; onlysame_scene scores are used forV.',
            selection='Fixed scene roster and identity cap; missing eligibility does not trigger replacement.',
            temporal_scope='Original full-input task eligibility; no croppedV or fabricated cross-scene scores.',
            limitations=['T/V share sampling rules, not individual examples, random draw order or equal geography/scene density.',
                'T is in-sample; scenes and identities are correlated. This is descriptive development evidence, not causal generalization.',
                '079T and074DV frozen scores may use different devices; tiny threshold-adjacent changes are not evidence of generalization.',
                'Balanced AP differs from full-candidate AP through prevalence and candidate composition.',
                'Standardized AP holds empirical class-conditional distributions fixed; it does not remove population confounding.',
                'AUC/mean margin improvement does not guarantee useful precision under rare positives or system recovery.'])), selections
