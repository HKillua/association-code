"""055A cached, fixed-score grouping-refinement diagnostics.

Truth identities classify already-fixed groups only. No threshold, score, group,
representative, or prediction is selected using these diagnostics.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import math

import numpy as np

from association.evaluation.fusion_controls import attribute_errors, ordinary_fusion
from association.evaluation.grouping import _partition
from association.evaluation.physical import group_membership_diagnostics
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.system_attribution import (
    ANCHOR, ATTRIBUTES, BASE, INDEPENDENT, _metrics, _regions, _same,
    analyze_scene, pool_metrics,
)

NEW_ANCHOR = 'refined_anchor_ols'
NEW_MEAN = 'refined_mean_supported'
METHODS = (BASE, ANCHOR, NEW_ANCHOR, NEW_MEAN, INDEPENDENT)
CLASSES = ('unrepresented', 'singleton', 'pure_multi', 'mixed_multi')
SIZE_BINS = ((1, 1, 'size_1'), (2, 4, 'size_2_4'),
             (5, 16, 'size_5_16'), (17, math.inf, 'size_ge_17'))


def _ratios(counts):
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    # Same convention as association._counts; singleton precision is undefined.
    return dict(counts, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else (1. if sum(counts.values()) else None))


def _class(members, identities):
    return 1 if len(members) == 1 else 2 if len(set(identities[members])) == 1 else 3


def _artifacts(arrays, name):
    return {key: arrays[name+'_'+key] for key in (
        'aligned_prediction', 'aligned_prediction_valid', 'aligned_group_indices',
        'group_states', 'group_valid', 'group_anchor_target_indices')}


def _pair(a, b, artifacts, truth, truth_valid, regions, arrays, *, prefix='common_'):
    t = len(truth)
    common = truth_valid.copy()
    for name in (a, b):
        common &= artifacts[name]['aligned_prediction_valid'][:t]
    errors = {name: attribute_errors(artifacts[name]['aligned_prediction'][:t], truth)
              for name in (a, b)}
    result = dict(methods={name: _metrics(artifacts[name]['aligned_prediction'][:t],
        truth, truth_valid, common, regions) for name in (a, b)},
        error_change_second_minus_first={})
    for region, mask in regions.items():
        result['error_change_second_minus_first'][region] = {}
        for attr, cols in ATTRIBUTES.items():
            use = common[..., list(cols)].all(-1) & mask
            delta = (errors[b][attr]-errors[a][attr])[use]
            worse, better = delta > 1e-9, delta < -1e-9
            result['error_change_second_minus_first'][region][attr] = dict(
                common_point_count=len(delta), second_worse_point_count=int(worse.sum()),
                second_better_point_count=int(better.sum()),
                tied_point_count=int((~worse & ~better).sum()),
                harm_sum=float(delta[worse].sum()), benefit_sum=float(-delta[better].sum()),
                signed_error_change_sum=float(delta.sum()))
    arrays[prefix+a+'__vs__'+b+'_valid'] = common
    return result


def _group_metadata(groups, nodes, support, parents, parent_groups, native):
    q = support.shape[1]
    classes = np.asarray([_class(g, nodes) for g in groups], np.uint8)
    sizes = np.asarray([len(g) for g in groups], np.int64)
    parent_ids = np.asarray([parents[g[0]] for g in groups], np.int64)
    parent_sizes = np.asarray([len(parent_groups[i]) for i in parent_ids], np.int64)
    helper = np.asarray([support[g[1:]].any(0) if len(g) > 1 else np.zeros(q, bool)
                         for g in groups], bool).reshape(len(groups), q)
    effective = np.broadcast_to(helper[..., None], (len(groups), q, 4)).copy()
    angular_fallback = np.zeros((len(groups), q, 2), bool)
    no_supported = np.zeros((len(groups), q), bool)
    for gid, group in enumerate(groups):
        weights = support[group]
        count = weights.sum(0)
        no_supported[gid] = count == 0
        for output_column, column in enumerate((1, 3)):
            radians = np.deg2rad(native[group, :, column].astype(np.float64))
            divisor = np.maximum(count, 1)
            magnitude = np.hypot((np.sin(radians)*weights).sum(0)/divisor,
                                 (np.cos(radians)*weights).sum(0)/divisor)
            angular_fallback[gid, :, output_column] = (magnitude <= 1e-12) & (count > 0)
            effective[gid, :, column] &= ~angular_fallback[gid, :, output_column]
    parent_classes = np.asarray([_class(parent_groups[i], nodes) for i in parent_ids], np.uint8)
    return dict(group_class=classes, group_size=sizes, parent_group_index=parent_ids,
                parent_size=parent_sizes, helper_active=helper, parent_class=parent_classes,
                effective_helper_valid=effective, angular_fallback=angular_fallback,
                no_supported_fallback=no_supported)


def _group_regions(meta):
    q = meta['helper_active'].shape[1]
    broadcast = lambda value: np.broadcast_to(value[:, None], (len(value), q))
    regions = dict(all=np.ones(meta['helper_active'].shape, bool), helper_active=meta['helper_active'],
                   no_helper_active=~meta['helper_active'])
    for code, name in enumerate(CLASSES[1:], 1):
        regions[name] = broadcast(meta['group_class'] == code)
        regions[name+'_helper_active'] = regions[name] & meta['helper_active']
    for low, high, name in SIZE_BINS:
        for key, label in (('group_size', 'group_'), ('parent_size', 'parent_')):
            regions[label+name] = broadcast((meta[key] >= low) & (meta[key] <= high))
    regions['old_mixed_to_new_pure_multi'] = broadcast(
        (meta['parent_class'] == 3) & (meta['group_class'] == 2))
    regions['old_mixed_to_new_singleton'] = broadcast(
        (meta['parent_class'] == 3) & (meta['group_class'] == 1))
    return regions


def _selected_metadata(artifact, meta, target_count, q):
    selected = artifact['aligned_group_indices'][:target_count]
    have = selected >= 0
    result = {}
    for key in ('group_class', 'group_size', 'parent_size', 'parent_class'):
        result[key] = np.zeros(target_count, meta[key].dtype)
        result[key][have] = meta[key][selected[have]]
    result['helper_active'] = np.zeros((target_count, q), bool)
    result['helper_active'][have] = meta['helper_active'][selected[have]]
    result['effective_helper_valid'] = np.zeros((target_count, q, 4), bool)
    result['effective_helper_valid'][have] = meta['effective_helper_valid'][selected[have]]
    return result


def _group_population(groups, nodes, truth_valid, meta):
    membership = np.empty(len(nodes), np.int64)
    for gid, members in enumerate(groups):
        membership[members] = gid
    d = group_membership_diagnostics(membership, nodes)
    group_count, identities = len(groups), len({nodes[g[0]] for g in groups})
    out = dict(group_count=group_count, represented_identity_count=identities,
        pair_count=len(nodes)*(len(nodes)-1)//2,
        duplicate_group_count=group_count-identities,
        group_class_counts={name: int((meta['group_class'] == code).sum())
                            for code, name in enumerate(CLASSES[1:], 1)},
        group_size_counts={name: int(((meta['group_size'] >= low) & (meta['group_size'] <= high)).sum())
                           for low, high, name in SIZE_BINS},
        co_membership=_ratios(dict(tp=d['correct_same_identity_pairs'], fp=d['merge_pairs'],
                                  fn=d['split_pairs'], tn=d['correct_different_identity_pairs'])),
        capacity={})
    out['fusion_support_audit'] = dict(
        supported_helper_group_points=int(meta['helper_active'].sum()),
        no_supported_member_fallback_group_points=int(meta['no_supported_fallback'].sum()),
        longitude_fallback_group_points=int(meta['angular_fallback'][..., 0].sum()),
        heading_fallback_group_points=int(meta['angular_fallback'][..., 1].sum()))
    for attr, cols in ATTRIBUTES.items():
        available = int(truth_valid[..., list(cols)].all(-1).any(-1).sum())
        capacity = min(group_count, available)
        out['capacity'][attr] = dict(available_target_trajectory_count=available,
            group_capacity_target_count=capacity, group_count_deficit=available-capacity,
            trajectory_coverage_upper_bound=capacity/available if available else None)
    return out


def _transition(old, new, truth_valid, arrays, name):
    out = {}
    t = len(truth_valid)
    for attr, cols in ATTRIBUTES.items():
        valid = truth_valid[..., list(cols)].all(-1)
        a = valid & old['aligned_prediction_valid'][:t, :, list(cols)].all(-1)
        b = valid & new['aligned_prediction_valid'][:t, :, list(cols)].all(-1)
        out[attr] = dict(rescued_target_count=int((~a.any(1) & b.any(1)).sum()),
            lost_target_count=int((a.any(1) & ~b.any(1)).sum()),
            gained_target_points=int((b & ~a).sum()), lost_target_points=int((a & ~b).sum()))
        arrays[name+'_gained_'+attr+'_points'] = b & ~a
        arrays[name+'_lost_'+attr+'_points'] = a & ~b
    return out


def analyze_refinement(fragment, saved, old_groups, new_groups, expected=None, source_names=None):
    """Replay054, evaluate exactly two new arms, and audit pure refinement."""
    report, arrays = analyze_scene(fragment, saved, expected, source_names)
    nodes = np.asarray(fragment['identities']).astype(str)
    targets = np.asarray(fragment['target_identity_ids']).astype(str)
    native = np.asarray(fragment['native_states'])
    support, nearest = np.asarray(fragment['support']), np.asarray(fragment['nearest_seconds'])
    n, q, _ = native.shape
    t = len(targets)
    truth, valid = saved['aligned_truth'][:t], saved['aligned_truth_valid'][:t]
    old_groups, new_groups = _partition(old_groups, n), _partition(new_groups, n)
    offsets, flat = fragment['ols_group_offsets'], fragment['ols_group_node_indices']
    cached = [flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    if old_groups != cached:
        raise ValueError('old_groups differ from replayed053A groups')
    parents = np.empty(n, np.int64)
    for gid, group in enumerate(old_groups):
        parents[group] = gid
    if any(len(set(parents[g])) != 1 for g in new_groups):
        raise ValueError('refinement cannot merge nodes across old groups')
    if not {g[0] for g in old_groups}.issubset({g[0] for g in new_groups}):
        raise ValueError('refinement must retain every old anchor')
    original_mean = ordinary_fusion(native, nearest, support, old_groups)['predictions']['mean_supported']
    _same(original_mean, fragment[BASE], 'original ordinary fusion implementation')
    fixed_regions = _regions(arrays['target_input_support'], arrays['target_nearest_input_seconds'],
                             arrays['target_exact_observation'])
    fragment_regions = _regions(support, nearest, fragment['exact_observation'])
    sources = np.asarray(fragment['sources']).astype(str)
    for source in sorted(set(sources)) if source_names is None else source_names:
        fragment_regions['source_'+str(source)] = np.broadcast_to((sources == str(source))[:, None], (n, q))
    fused = ordinary_fusion(native, nearest, support, new_groups)
    for name, values in ((NEW_ANCHOR, native), (NEW_MEAN, fused['predictions']['mean_supported'])):
        system = evaluate_full_scene([dict(node_indices=g, states=values[g[0]],
            prediction_valid=np.isfinite(values[g[0]])) for g in new_groups],
            node_identities=nodes, target_identities=targets, target_states=truth, target_valid=valid,
            node_observed=support, split='V_select', grouping_kind='predicted',
            correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=support, nearest_seconds=nearest))
        artifact = system.pop('artifact')
        if name == NEW_ANCHOR:
            system['input_support_definition'] = report['methods'][ANCHOR]['system']['input_support_definition']
            system['input_distance_definition'] = report['methods'][ANCHOR]['system']['input_distance_definition']
        system['all_predicted_groups'] = [{k: v for k, v in g.items() if k != 'metrics'}
                                          for g in system['all_predicted_groups']]
        selected = np.full(t, -1, np.int64)
        for row, gid in enumerate(artifact['aligned_group_indices'][:t]):
            if gid >= 0:
                selected[row] = new_groups[gid][0]
        arrays[name+'_selected_anchor_node_index'] = selected
        arrays.update({name+'_'+key: value for key, value in artifact.items()})
        emitted, emitted_valid = np.full(native.shape, np.nan), np.zeros(native.shape, bool)
        for gid, members in enumerate(new_groups):
            emitted[members], emitted_valid[members] = artifact['group_states'][gid], artifact['group_valid'][gid]
        # All new fragment comparisons describe the actually emitted group
        # states; the baseline's historical fallback view remains separately.
        metric = _metrics(emitted, fragment['target'], fragment['target_valid'], emitted_valid, fragment_regions)
        report['methods'][name] = dict(system=system, fragment=metric, emitted_fragment=metric,
            fixed_target_strata=_metrics(artifact['aligned_prediction'][:t], truth, valid,
                                        artifact['aligned_prediction_valid'][:t], fixed_regions),
            duplicate_outputs={attr: dict(trajectory_count=int(artifact['aligned_prediction_valid'][t:, :, list(cols)].all(-1).any(-1).sum()),
                point_count=int(artifact['aligned_prediction_valid'][t:, :, list(cols)].all(-1).sum()))
                for attr, cols in ATTRIBUTES.items()})
    artifacts = {name: _artifacts(arrays, name) for name in METHODS}
    groupings = dict(original=old_groups, refined=new_groups, singleton=[[i] for i in range(n)])
    method_grouping = {BASE: 'original', ANCHOR: 'original', NEW_ANCHOR: 'refined',
                       NEW_MEAN: 'refined', INDEPENDENT: 'singleton'}
    metas = {name: _group_metadata(groups, nodes, support, parents, old_groups, native)
             for name, groups in groupings.items()}
    report['grouping'] = {name: _group_population(groups, nodes, valid, metas[name])
                          for name, groups in groupings.items()}
    selected_meta, selected_regions = {}, {}
    for name in METHODS:
        group_name = method_grouping[name]
        meta = selected_meta[name] = _selected_metadata(artifacts[name], metas[group_name], t, q)
        selected_regions[name] = _group_regions(meta)
        for key, value in meta.items():
            arrays[name+'_selected_'+key] = value
    for group_name, meta in metas.items():
        for key, value in meta.items():
            arrays[group_name+'_'+key] = value
    arrays['group_class_names'] = np.asarray(CLASSES)
    report['pairwise'] = {}
    for a, b in combinations(sorted(METHODS), 2):
        sa, sb = arrays[a+'_selected_anchor_node_index'], arrays[b+'_selected_anchor_node_index']
        both = (sa >= 0) & (sb >= 0)
        same, changed = both & (sa == sb), both & (sa != sb)
        regions = dict(fixed_regions,
            same_representative=np.broadcast_to(same[:, None], (t, q)),
            changed_representative=np.broadcast_to(changed[:, None], (t, q)))
        # Method-specific strata prevent treating changed pure-group populations
        # as equivalent cohorts. Both names are explicit in the saved report.
        if method_grouping[a] == method_grouping[b]:
            # Same-partition fusion contrasts need purity/size strata; across
            # partitions the four-arm common cohort below controls both groups.
            regions.update({key: value for key, value in selected_regions[a].items() if key != 'all'})
        pair = _pair(a, b, artifacts, truth, valid, regions, arrays)
        pair.update(same_representative_target_count=int(same.sum()), changed_representative_target_count=int(changed.sum()))
        report['pairwise'][a+'__vs__'+b] = pair
    # Four arms on one population, plus an unchanged-anchor restriction, are
    # needed to distinguish better fusion from simply removing all helpers.
    four = (ANCHOR, BASE, NEW_ANCHOR, NEW_MEAN)
    common = valid.copy()
    for name in four:
        common &= artifacts[name]['aligned_prediction_valid'][:t]
    sa, sb = arrays[ANCHOR+'_selected_anchor_node_index'], arrays[NEW_ANCHOR+'_selected_anchor_node_index']
    same = np.broadcast_to(((sa >= 0) & (sa == sb))[:, None], (t, q))
    old_meta, new_meta = selected_meta[ANCHOR], selected_meta[NEW_ANCHOR]
    both_multi = same & (old_meta['group_size'] > 1)[:, None] & (new_meta['group_size'] > 1)[:, None]
    both_helpers = old_meta['helper_active'] & new_meta['helper_active']
    regions = dict(all=np.ones((t, q), bool), same_anchor=same,
        changed_anchor=np.broadcast_to(((sa >= 0) & (sb >= 0) & (sa != sb))[:, None], (t, q)),
        same_anchor_both_multi=both_multi,
        same_anchor_both_multi_old_helper=both_multi & old_meta['helper_active'],
        same_anchor_both_multi_new_helper=both_multi & new_meta['helper_active'],
        same_anchor_both_multi_both_helpers=both_multi & both_helpers,
        pure_multi_same_anchor_both_helpers=both_multi & both_helpers &
            (old_meta['group_class'] == 2)[:, None] & (new_meta['group_class'] == 2)[:, None],
        old_mixed_to_new_pure_multi_same_anchor_both_helpers=both_multi & both_helpers &
            (old_meta['group_class'] == 3)[:, None] & (new_meta['group_class'] == 2)[:, None])
    arrays['joint_four_arm_valid'] = common
    arrays.update({'joint_region_'+name: value for name, value in regions.items()})
    joint = report['joint_four_arm'] = dict(methods={name: _metrics(
        artifacts[name]['aligned_prediction'][:t], truth, valid, common, regions) for name in four})
    joint['fusion_effect'] = {key: _fusion_effect(joint['methods'][a], joint['methods'][b])
                             for key, a, b in (('original', ANCHOR, BASE), ('refined', NEW_ANCHOR, NEW_MEAN))}
    effective_common = common & old_meta['effective_helper_valid'] & new_meta['effective_helper_valid']
    arrays['joint_effective_helpers_valid'] = effective_common
    effective = report['joint_effective_helpers'] = dict(methods={name: _metrics(
        artifacts[name]['aligned_prediction'][:t], truth, valid, effective_common, regions) for name in four})
    effective['fusion_effect'] = {key: _fusion_effect(effective['methods'][a], effective['methods'][b])
        for key, a, b in (('original', ANCHOR, BASE), ('refined', NEW_ANCHOR, NEW_MEAN))}
    report['helper_definition'] = ('helper_active means at least one non-anchor member has support and positive mean_supported weight; '
        'it is not proof of benefit or counterfactual influence. joint_effective_helpers additionally excludes angular '
        'zero-resultant fallback for longitude/heading; position requires both coordinates. All masks use observable states/support only.')
    # All-output view intentionally repeats truth for duplicate representative
    # identities and must never be pooled into the unique-target main table.
    report['all_group_fusion'] = {}
    for label, a, b in (('original', ANCHOR, BASE), ('refined', NEW_ANCHOR, NEW_MEAN),
                        ('singleton', INDEPENDENT, INDEPENDENT)):
        aa, bb = artifacts[a], artifacts[b]
        rows = aa['group_anchor_target_indices']
        common_group = valid[rows] & aa['group_valid'] & bb['group_valid']
        regions = _group_regions(metas[label])
        am = _metrics(aa['group_states'], truth[rows], valid[rows], common_group, regions)
        bm = _metrics(bb['group_states'], truth[rows], valid[rows], common_group, regions)
        report['all_group_fusion'][label] = dict(anchor=am, mean_supported=bm,
            fusion_effect=_fusion_effect(am, bm), scope='all emitted groups including duplicate identities; not unique-target MAE')
    for new, old in ((NEW_ANCHOR, ANCHOR), (NEW_MEAN, BASE)):
        report['coverage_transitions'][new] = _transition(artifacts[old], artifacts[new], valid, arrays, new)
        if any(v['lost_target_count'] or v['lost_target_points']
               for v in report['coverage_transitions'][new].values()):
            raise ValueError('finite OLS refinement unexpectedly reduced target coverage')
        gain_regions = {}
        for attr in ATTRIBUTES:
            gain_regions['newly_covered_'+attr+'_points'] = arrays[new+'_gained_'+attr+'_points']
        report['methods'][new]['new_coverage_strata'] = _metrics(
            artifacts[new]['aligned_prediction'][:t], truth, valid,
            artifacts[new]['aligned_prediction_valid'][:t], gain_regions)
    old, new = report['grouping']['original'], report['grouping']['refined']
    dg, du, dd = (new[k]-old[k] for k in ('group_count', 'represented_identity_count', 'duplicate_group_count'))
    if min(dg, du, dd) < 0 or dg != du+dd:
        raise ValueError('refinement violates group/identity/duplicate conservation')
    removed_tp = old['co_membership']['tp']-new['co_membership']['tp']
    removed_fp = old['co_membership']['fp']-new['co_membership']['fp']
    if min(removed_tp, removed_fp) < 0 or new['co_membership']['fn'] < old['co_membership']['fn']:
        raise ValueError('refinement introduced a new co-membership pair')
    report['refinement_accounting'] = dict(delta_group_count=dg, delta_represented_identity_count=du,
        delta_duplicate_group_count=dd, removed_tp=removed_tp, removed_fp=removed_fp,
        attributes={attr: dict(report['coverage_transitions'][NEW_ANCHOR][attr],
            recovered_targets_per_added_group=report['coverage_transitions'][NEW_ANCHOR][attr]['rescued_target_count']/dg if dg else None,
            added_duplicates_per_added_group=dd/dg if dg else None) for attr in ATTRIBUTES})
    return report, arrays


def _fusion_effect(anchor, mean):
    return {region: {attr: dict(mean_minus_anchor_mae=(mean[region][attr]['mae']-anchor[region][attr]['mae'])
        if mean[region][attr]['mae'] is not None else None,
        common_point_count=mean[region][attr]['intersection_count']) for attr in ATTRIBUTES} for region in anchor}


def _sum_counts(items):
    result = Counter()
    for item in items:
        result.update(item)
    return dict(result)


def _aggregate_effect(rows, key, anchor, mean):
    result = _fusion_effect(anchor, mean)
    for region, attrs in result.items():
        for attr, out in attrs.items():
            values = [r[key][region][attr]['mean_minus_anchor_mae'] for r in rows]
            finite = [v for v in values if v is not None]
            out.update(scene_mean_worse_count=sum(v > 1e-9 for v in finite),
                scene_mean_better_count=sum(v < -1e-9 for v in finite), eligible_scene_count=len(finite),
                scene_macro_mean_minus_anchor_mae=float(np.mean(finite)) if finite else None)
    return result


def aggregate_refinement(rows):
    """Pool exactly supplied scenes; also usable for predeclared scene subsets."""
    if not rows:
        raise ValueError('cannot aggregate empty refinement scene population')
    result = dict(scene_count=len(rows), methods={}, grouping={}, pairwise={},
                  coverage_transitions={}, all_group_fusion={}, joint_four_arm={})
    for name in METHODS:
        items = [r['methods'][name] for r in rows]
        output = result['methods'][name] = {key: pool_metrics([r[key] for r in items])
            for key in ('fragment', 'emitted_fragment', 'fixed_target_strata')}
        if name in (NEW_ANCHOR, NEW_MEAN):
            output['new_coverage_strata'] = pool_metrics([r['new_coverage_strata'] for r in items])
        output['system'] = {key: pool_metrics([r['system'][key] for r in items]) for key in (
            'full_target_metrics', 'input_support_metrics', 'input_distance_metrics')}
        output['population'] = {key: sum(r['system'][key] for r in items) for key in (
            'target_identity_count', 'predicted_group_count', 'duplicate_prediction_group_count',
            'anchor_matched_target_count', 'reference_unavailable_target_count')}
        output['duplicate_outputs'] = {attr: _sum_counts([r['duplicate_outputs'][attr] for r in items]) for attr in ATTRIBUTES}
    for label in rows[0]['grouping']:
        items = [r['grouping'][label] for r in rows]
        out = result['grouping'][label] = {key: sum(r[key] for r in items) for key in (
            'group_count', 'represented_identity_count', 'duplicate_group_count', 'pair_count')}
        for key in ('group_class_counts', 'group_size_counts', 'fusion_support_audit'):
            out[key] = _sum_counts([r[key] for r in items])
        out['co_membership'] = _ratios({key: sum(r['co_membership'][key] for r in items) for key in ('tp', 'fp', 'fn', 'tn')})
        out['co_membership']['eligible_scene_count'] = sum(r['pair_count'] > 0 for r in items)
        for key in ('precision', 'recall', 'f1'):
            values = [r['co_membership'][key] for r in items
                      if r['pair_count'] > 0 and r['co_membership'][key] is not None]
            out['co_membership']['scene_macro_'+key] = float(np.mean(values)) if values else None
            out['co_membership']['scene_macro_'+key+'_defined_scene_count'] = len(values)
        out['capacity'] = {}
        for attr in ATTRIBUTES:
            cap = out['capacity'][attr] = {key: sum(r['capacity'][attr][key] for r in items) for key in (
                'available_target_trajectory_count', 'group_capacity_target_count', 'group_count_deficit')}
            cap['trajectory_coverage_upper_bound'] = cap['group_capacity_target_count']/cap['available_target_trajectory_count'] if cap['available_target_trajectory_count'] else None
    for name in rows[0]['coverage_transitions']:
        result['coverage_transitions'][name] = {attr: _sum_counts([r['coverage_transitions'][name][attr] for r in rows]) for attr in ATTRIBUTES}
    for pair, template in rows[0]['pairwise'].items():
        items = [r['pairwise'][pair] for r in rows]
        out = result['pairwise'][pair] = dict(methods={name: pool_metrics([r['methods'][name] for r in items]) for name in template['methods']},
            **{key: sum(r[key] for r in items) for key in ('same_representative_target_count', 'changed_representative_target_count')})
        out['error_change_second_minus_first'] = {region: {attr: _sum_counts([
            r['error_change_second_minus_first'][region][attr] for r in items]) for attr in ATTRIBUTES}
            for region in template['error_change_second_minus_first']}
        # Direction is explicit in method names; count scene consistency on the
        # same eligible common points, never on separately covered populations.
        a, b = template['methods']
        out['scene_error_change'] = {}
        for region in template['methods'][a]:
            out['scene_error_change'][region] = {}
            for attr in ATTRIBUTES:
                values = [r['methods'][b][region][attr]['mae']-r['methods'][a][region][attr]['mae']
                    for r in items if r['methods'][a][region][attr]['mae'] is not None]
                out['scene_error_change'][region][attr] = dict(eligible_scene_count=len(values),
                    second_worse_scene_count=sum(v > 1e-9 for v in values),
                    second_better_scene_count=sum(v < -1e-9 for v in values))
    for joint_key in ('joint_four_arm', 'joint_effective_helpers'):
        joint_items = [r[joint_key] for r in rows]
        joint = result[joint_key] = {}
        joint['methods'] = {name: pool_metrics([r['methods'][name] for r in joint_items]) for name in joint_items[0]['methods']}
        joint['fusion_effect'] = {}
        for label, a, b in (('original', ANCHOR, BASE), ('refined', NEW_ANCHOR, NEW_MEAN)):
            joint['fusion_effect'][label] = _aggregate_effect(
                [dict(effect=r['fusion_effect'][label]) for r in joint_items], 'effect', joint['methods'][a], joint['methods'][b])
    for label in rows[0]['all_group_fusion']:
        items = [r['all_group_fusion'][label] for r in rows]
        out = result['all_group_fusion'][label] = {name: pool_metrics([r[name] for r in items]) for name in ('anchor', 'mean_supported')}
        out['fusion_effect'] = _aggregate_effect(items, 'fusion_effect', out['anchor'], out['mean_supported'])
        out['scope'] = items[0]['scope']
    account = result['refinement_accounting'] = {key: sum(r['refinement_accounting'][key] for r in rows) for key in (
        'delta_group_count', 'delta_represented_identity_count', 'delta_duplicate_group_count', 'removed_tp', 'removed_fp')}
    dg, dd = account['delta_group_count'], account['delta_duplicate_group_count']
    account['attributes'] = {attr: dict(result['coverage_transitions'][NEW_ANCHOR][attr],
        recovered_targets_per_added_group=result['coverage_transitions'][NEW_ANCHOR][attr]['rescued_target_count']/dg if dg else None,
        added_duplicates_per_added_group=dd/dg if dg else None) for attr in ATTRIBUTES}
    result['helper_definition'] = rows[0]['helper_definition']
    return result
