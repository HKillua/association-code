"""128: same-information group geometry and leave-one-fragment-out diagnostics.

Features/candidates are frozen before identity/label arrays are opened. No new
partition, physical output, learned model or reference field is produced/read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from tools.analysis import static_partition_position_control as old

IDEA = 'IDEA-ASSOCIATION-GEOMETRY-SIGNAL-20261029'
RADIUS = 6371000.
TWO_ATOL_M, TWO_RTOL = 1e-6, 1e-10
EUCLIDEAN_ATOL_M2, EUCLIDEAN_RTOL = 1e-4, 1e-10
PHYSICAL_KEYS = ('native_states', 'support', 'sources', 'node_ids')
FEATURE_NAMES = ('candidate_size', 'source_count', 'intra_source_pairs', 'cross_source_pairs',
    'score_sum', 'score_mean', 'score_min', 'score_max', 'score_median', 'negative_margin_fraction',
    'pair_shared_median_m', 'pair_shared_p90_m', 'pair_shared_rms_m', 'pair_shared_query_count', 'pair_shared_query_fraction',
    'pair_full_median_m', 'pair_full_p90_m', 'pair_full_rms_m',
    'loo_median_m', 'loo_p90_m', 'loo_rms_m', 'loo_member_query_count', 'loo_supported_member_fraction')
SCORERS = dict(SCORE_SUM=('score_sum', 1.), SCORE_MEAN=('score_mean', 1.),
    SCORE_MIN=('score_min', 1.), SCORE_MEDIAN=('score_median', 1.),
    PAIR_SHARED_MEDIAN=('pair_shared_median_m', -1.), PAIR_SHARED_RMS=('pair_shared_rms_m', -1.),
    PAIR_FULL_MEDIAN=('pair_full_median_m', -1.), PAIR_FULL_RMS=('pair_full_rms_m', -1.),
    LOO_MEDIAN=('loo_median_m', -1.), LOO_RMS=('loo_rms_m', -1.))


def distance(a, b):
    """Great-circle metres; symmetric arithmetic, no production evaluator."""
    a, b = np.deg2rad(np.asarray(a, np.float64)), np.deg2rad(np.asarray(b, np.float64))
    delta = b-a
    h = np.sin(delta[..., 0]/2.)**2 + np.cos(a[..., 0])*np.cos(b[..., 0])*np.sin(delta[..., 1]/2.)**2
    return 2.*RADIUS*np.arcsin(np.sqrt(np.clip(h, 0., 1.)))


def candidates_from_groups(groups, pairs, scores, n, threshold):
    """All original groups plus two score-ranked single-node extensions."""
    matrix = np.full((n, n), np.nan)
    matrix[pairs[:, 0], pairs[:, 1]] = scores-threshold
    matrix[pairs[:, 1], pairs[:, 0]] = scores-threshold
    result, original = {}, set()
    for members in groups:
        key = tuple(sorted(members))
        result.setdefault(key, None)
        original.add(key)
    proposed = 0
    for members in groups:
        if len(members) < 2:
            continue
        outside = sorted(set(range(n))-set(members))
        ranking = sorted(outside, key=lambda node: (-float(matrix[members, node].mean()), node))
        for node in ranking[:2]:
            proposed += 1
            result.setdefault(tuple(sorted([*members, node])), None)
    candidates = list(result)
    return candidates, np.asarray([key in original for key in candidates], bool), proposed


def position_mean(positions, support):
    count = support.sum(0)
    weights = support/np.maximum(count, 1)
    lat = (positions[..., 0]*weights).sum(0)
    angle = np.deg2rad(positions[..., 1])
    x, y = (np.cos(angle)*weights).sum(0), (np.sin(angle)*weights).sum(0)
    lon = np.rad2deg(np.arctan2(y, x))
    degenerate = (np.abs(x)+np.abs(y)) < 1e-12
    lon[degenerate] = positions[0, degenerate, 1]
    value = np.stack([lat, lon], axis=-1)
    one = count == 1
    if one.any():
        index = support.argmax(0)
        value[one] = positions[index[one], np.flatnonzero(one)]
    return value, count > 0


def three_stats(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return [np.nan]*3
    return [float(np.median(values)), float(np.quantile(values, .9)), float(np.sqrt(np.mean(values*values)))]


def euclidean_reconstruction(positions, support):
    """Only the stated squared-error identity; no physical model assumption."""
    checked, maximum = 0, 0.
    for q in range(positions.shape[1]):
        p = positions[support[:, q], q]
        m = len(p)
        if m < 3:
            continue
        latitude = float(p[:, 0].mean())
        delta = p-p[0]
        delta[:, 1] = (delta[:, 1]+180.) % 360.-180.
        x = RADIUS*np.pi/180.*delta*np.array([1., np.cos(np.deg2rad(latitude))])
        prediction = (x.sum(0)-x)/(m-1)
        direct = float(np.mean(np.sum((x-prediction)**2, axis=-1)))
        a, b = np.triu_indices(m, 1)
        reconstructed = float(np.sum((x[a]-x[b])**2)/(m-1)**2)
        difference = abs(direct-reconstructed)
        maximum = max(maximum, difference)
        if not np.isclose(direct, reconstructed, rtol=EUCLIDEAN_RTOL, atol=EUCLIDEAN_ATOL_M2):
            raise ValueError('prespecified Euclidean pair reconstruction tolerance exceeded')
        checked += 1
    return checked, maximum


def construct_features(states, support, sources, groups, pairs, scores, threshold, *, pair_geometry_builder=None):
    """Identity/reference-free API and full pair geometry for the same candidates."""
    states, support, sources = np.asarray(states, np.float64), np.asarray(support), np.asarray(sources)
    pairs, scores = np.asarray(pairs), np.asarray(scores, np.float64)
    n = len(states)
    if (states.shape != (n, 64, 4) or not np.isfinite(states).all()
            or (np.abs(states[..., 0]) > 90).any() or (np.abs(states[..., 1]) > 180).any()
            or support.shape != (n, 64) or support.dtype != bool or sources.shape != (n,)
            or sources.dtype.kind not in 'US' or any(not str(value) for value in sources)
            or pairs.shape != (n*(n-1)//2, 2) or pairs.dtype.kind not in 'iu'
            or scores.shape != (len(pairs),) or not np.isfinite(scores).all()
            or not np.isfinite(threshold)):
        raise ValueError('complete finite Q64 geometry/support and unordered pair scores required')
    if not np.array_equal(pairs, np.stack(np.triu_indices(n, 1), axis=-1)):
        raise ValueError('complete original lexicographic i<j pair population required')
    canonical = old.groups_from_membership(
        old.partition.membership_from_clusters(groups, n), n)
    if canonical != groups:
        raise ValueError('original canonical group/member order required')
    candidates, original, proposed = candidates_from_groups(groups, pairs, scores, n, threshold)
    pair_geometry = (distance(states[pairs[:, 0], :, :2], states[pairs[:, 1], :, :2])
        if pair_geometry_builder is None else pair_geometry_builder(states, pairs))
    if pair_geometry_builder is not None and (pair_geometry.shape != (len(pairs), 64) or not np.isfinite(pair_geometry).all()):
        raise ValueError('complete finite same-query pair geometry required')
    pair_support = support[pairs[:, 0]] & support[pairs[:, 1]]
    pair_index = np.full((n, n), -1, np.int64)
    pair_index[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs))
    pair_index[pairs[:, 1], pairs[:, 0]] = np.arange(len(pairs))
    features = np.full((len(candidates), len(FEATURE_NAMES)), np.nan)
    pair_offsets, pair_members = [0], []
    two_checks, two_maximum, euclidean_checks, euclidean_maximum = 0, 0., 0, 0.
    for cid, candidate in enumerate(candidates):
        members = np.asarray(candidate, np.int64)
        m = len(members)
        ia, ib = np.triu_indices(m, 1)
        internal = pair_index[members[ia], members[ib]]
        pair_members.extend(internal.tolist())
        pair_offsets.append(len(pair_members))
        same = sources[members[ia]] == sources[members[ib]]
        row = dict(candidate_size=m, source_count=len(set(sources[members].tolist())),
                   intra_source_pairs=int(same.sum()), cross_source_pairs=int((~same).sum()))
        margin = scores[internal]-threshold
        if len(margin):
            row.update(score_sum=float(margin.sum()), score_mean=float(margin.mean()),
                score_min=float(margin.min()), score_max=float(margin.max()),
                score_median=float(np.median(margin)), negative_margin_fraction=float((margin < 0).mean()))
        shared = pair_geometry[internal][pair_support[internal]]
        full = pair_geometry[internal].reshape(-1)
        for prefix, values in [('pair_shared', shared), ('pair_full', full)]:
            row.update(zip((prefix+'_median_m', prefix+'_p90_m', prefix+'_rms_m'), three_stats(values)))
        row['pair_shared_query_count'] = len(shared)
        row['pair_shared_query_fraction'] = len(shared)/(len(internal)*64) if len(internal) else 0.
        residuals, supported_members = [], 0
        if m >= 2:
            for local, node in enumerate(members):
                others = np.delete(members, local)
                if len(others) == 1:
                    prediction, has = states[others[0], :, :2], support[others[0]]
                else:
                    prediction, has = position_mean(states[others, :, :2], support[others])
                use = support[node] & has
                value = distance(states[node, use, :2], prediction[use])
                residuals.extend(value.tolist())
                supported_members += int(use.any())
                if m == 2 and use.any():
                    expected = pair_geometry[internal[0], use]
                    two_checks += len(value)
                    two_maximum = max(two_maximum, float(np.max(np.abs(value-expected))))
                    if not np.allclose(value, expected, rtol=TWO_RTOL, atol=TWO_ATOL_M):
                        raise ValueError('prespecified two-member LOO/pair tolerance exceeded')
            count, difference = euclidean_reconstruction(states[members, :, :2], support[members])
            euclidean_checks += count
            euclidean_maximum = max(euclidean_maximum, difference)
        row.update(zip(('loo_median_m', 'loo_p90_m', 'loo_rms_m'), three_stats(residuals)))
        row.update(loo_member_query_count=len(residuals), loo_supported_member_fraction=supported_members/m)
        features[cid] = [row.get(key, np.nan) for key in FEATURE_NAMES]
    offsets = np.cumsum([0]+[len(candidate) for candidate in candidates])
    private = dict(features=features, feature_names=np.asarray(FEATURE_NAMES),
        candidate_offsets=offsets, candidate_node_indices=np.asarray([node for c in candidates for node in c], np.int64),
        candidate_is_original=original, candidate_pair_offsets=np.asarray(pair_offsets),
        candidate_pair_indices=np.asarray(pair_members, np.int64), pairs=pairs,
        pair_distance_m=pair_geometry, pair_joint_support=pair_support, pair_scores=scores,
        threshold=np.asarray(threshold))
    audit = dict(original_group_count=len(groups), proposed_extension_count=proposed,
        candidate_count=len(candidates), exact_duplicate_candidates_removed=len(groups)+proposed-len(candidates),
        singleton_candidate_count=int(sum(len(c) == 1 for c in candidates)),
        two_member_pair_query_checks=two_checks, maximum_two_member_difference_m=two_maximum,
        euclidean_query_checks=euclidean_checks, maximum_euclidean_difference_m2=euclidean_maximum,
        candidate_node_coverage_count=len({node for candidate in candidates for node in candidate}),
        feature_identity_or_reference_used=False)
    return private, audit


def auc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, np.float64)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind='stable')
    y, x = labels[order], scores[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(x) != 0)+1, len(x)]
    below_negative, value = 0, 0.
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        pos = int(y[left:right].sum())
        neg = right-left-pos
        value += pos*(below_negative+.5*neg)
        below_negative += neg
    return value/(positive*negative)


def evaluate_features(private, identities):
    identity = np.asarray(identities)
    offsets, members = private['candidate_offsets'], private['candidate_node_indices']
    n = 0 if not len(members) else int(members.max())+1
    if identity.shape != (n,) or identity.dtype.kind not in 'US':
        raise ValueError('one identity string per frozen original node required after feature freeze')
    pure, known = [], []
    for left, right in zip(offsets[:-1], offsets[1:]):
        ids = identity[members[left:right]]
        known.append(bool(np.all(ids != '')))
        pure.append(len(set(ids.tolist())) == 1)
    evidence = dict(pure=np.asarray(pure, bool), known=np.asarray(known, bool))
    for key in ('candidate_size', 'source_count', 'pair_shared_query_count', 'pair_shared_query_fraction', 'loo_member_query_count'):
        evidence[key] = private['features'][:, FEATURE_NAMES.index(key)]
    for name, (key, direction) in SCORERS.items():
        evidence[name] = private['features'][:, FEATURE_NAMES.index(key)]*direction
    return evidence


def summarize(evidence):
    pure, known = evidence['pure'], evidence['known']
    size, source = evidence['candidate_size'], evidence['source_count']
    masks = dict(all=np.ones(len(pure), bool), original=evidence.get('original', np.ones(len(pure), bool)),
        size1=size == 1, size2=size == 2, size3=size == 3, size4plus=size >= 4,
        single_source=source == 1, cross_source=source > 1,
        no_shared_pair_support=evidence['pair_shared_query_count'] == 0,
        has_shared_pair_support=evidence['pair_shared_query_count'] > 0,
        shared_fraction_0_to_half=(evidence['pair_shared_query_fraction'] > 0) & (evidence['pair_shared_query_fraction'] <= .5),
        shared_fraction_over_half=evidence['pair_shared_query_fraction'] > .5,
        common_all_scorers=np.logical_and.reduce([np.isfinite(evidence[name]) for name in SCORERS]))
    out = {}
    for label, mask in masks.items():
        out[label] = dict(candidate_count=int(mask.sum()), known_label_count=int((mask & known).sum()),
            unknown_label_count=int((mask & ~known).sum()), pure_count=int((mask & known & pure).sum()),
            mixed_count=int((mask & known & ~pure).sum()), scorers={})
        for name in SCORERS:
            use = mask & known & np.isfinite(evidence[name])
            out[label]['scorers'][name] = dict(defined_count=int(use.sum()),
                undefined_count=int((mask & ~np.isfinite(evidence[name])).sum()),
                pure_defined_count=int((use & pure).sum()), mixed_defined_count=int((use & ~pure).sum()),
                auroc=auc(pure[use], evidence[name][use]))
    return out


def aggregate(rows, evidence):
    combined = {key: np.concatenate([value[key] for value in evidence]) for key in evidence[0]}
    return dict(scene_ids=[row['scene_id'] for row in rows], scene_count=len(rows),
        node_count=sum(row['node_count'] for row in rows),
        counts={key: sum(row['feature_audit'][key] for row in rows) for key in
            ('original_group_count', 'proposed_extension_count', 'candidate_count',
             'exact_duplicate_candidates_removed', 'singleton_candidate_count',
             'two_member_pair_query_checks', 'euclidean_query_checks', 'candidate_node_coverage_count')},
        maximum_two_member_difference_m=max(row['feature_audit']['maximum_two_member_difference_m'] for row in rows),
        maximum_euclidean_difference_m2=max(row['feature_audit']['maximum_euclidean_difference_m2'] for row in rows),
        candidate_population=summarize(combined),
        scene_equal_auroc={name: dict(defined_scenes=sum(row['candidate_population']['all']['scorers'][name]['auroc'] is not None for row in rows),
            mean=float(np.mean([row['candidate_population']['all']['scorers'][name]['auroc'] for row in rows
                if row['candidate_population']['all']['scorers'][name]['auroc'] is not None])) if any(
                row['candidate_population']['all']['scorers'][name]['auroc'] is not None for row in rows) else None)
            for name in SCORERS})


def run(args):
    source, partition, output = (Path(getattr(args, key)).resolve() for key in ('source_system', 'partition_run', 'output'))
    for protected in (source.parent, partition):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('new output must not overlap frozen input')
    if not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), source_system=str(source), partition_run=str(partition),
        completed_scenes=0, models_constructed=0, model_forwards=0, optimizer_updates=0, gpu_used=False,
        partition_changed=False, physical_predictions_changed=False, scores_recomputed=False,
        reference_array_fields_loaded=False, identity_array_fields_loaded=False,
        label_arrays_loaded_only_after_feature_freeze=True, test_opened=False,
        metadata_binding_uses_existing_evaluation_summaries=True,
        two_member_tolerance=dict(atol_metres=TWO_ATOL_M, rtol=TWO_RTOL),
        euclidean_tolerance=dict(atol_m2=EUCLIDEAN_ATOL_M2, rtol=EUCLIDEAN_RTOL))
    def checkpoint(**extra):
        state.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes(),
            advisory_exceeded=time.monotonic()-started > 600, **extra)
        old.frozen.write(output/'run.json', state)
        if state['peak_rss_bytes'] > old.frozen.RAM_BUDGET:
            raise MemoryError('4GiB resident-memory budget exceeded')
    try:
        checkpoint()
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([old.frozen.MAX_AS]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
            if hasattr(os, 'sched_getaffinity'):
                os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cpu_threads])
        meta = old.metadata(source, partition, old.frozen.read(args.manifest))
        checkpoint(scene_ids=meta['ids'], total_scenes=len(meta['ids']))
        (output/'scenes').mkdir()
        (output/'private').mkdir()
        rows, all_evidence = [], []
        with np.load(meta['association']/'final-scores.npz', allow_pickle=False) as scores:
            old._same(scores['scene_ids'], meta['all_ids'], '056 full score scene order')
            for sid in meta['ids']:
                with np.load(source/'predictions'/f'states-{sid}.npz', allow_pickle=False) as saved:
                    physical = {key: saved[key].copy() for key in PHYSICAL_KEYS}
                for key in ('node_ids', 'sources'):
                    old._same(physical[key], scores[f'{sid}_{key}'], '056 '+key)
                pairs, values = scores[f'{sid}_pairs'], scores[f'{sid}_scores']
                membership = np.asarray(old.frozen.read(partition/'private'/f'scene-{sid}.json.gz')['memberships']['WS'])
                groups = old.groups_from_membership(membership, len(physical['node_ids']))
                features, audit = construct_features(physical['native_states'], physical['support'], physical['sources'],
                    groups, pairs, values, old.frozen.THRESHOLD)
                # This artifact is durable before opening identity/label fields.
                np.savez_compressed(output/'private'/f'features-{sid}.npz', **features)
                with np.load(source/'predictions'/f'states-{sid}.npz', allow_pickle=False) as saved:
                    identities = saved['identities'].copy()
                state['identity_array_fields_loaded'] = True
                labels = np.asarray(scores[f'{sid}_labels'])
                known_pair = (identities[pairs[:, 0]] != '') & (identities[pairs[:, 1]] != '')
                old._same((identities[pairs[:, 0]] == identities[pairs[:, 1]])[known_pair], labels[known_pair],
                          '056 post-freeze identity/pair label binding')
                evidence = evaluate_features(features, identities)
                evidence['original'] = features['candidate_is_original']
                np.savez_compressed(output/'private'/f'evaluation-{sid}.npz', **evidence)
                row = dict(scene_id=sid, component_id=str(meta['components'][sid]), node_count=len(identities),
                    feature_audit=audit, candidate_population=summarize(evidence),
                    feature_file_written_before_identity_read=True, reference_array_fields_loaded=False)
                old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                rows.append(row); all_evidence.append(evidence)
                checkpoint(completed_scenes=len(rows), last_scene_id=sid)
        components = sorted({row['component_id'] for row in rows})
        def subset(indices):
            return aggregate([rows[i] for i in indices], [all_evidence[i] for i in indices]) if indices else None
        largest = max(components, key=lambda cid: sum(row['component_id'] == cid for row in rows))
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **aggregate(rows, all_evidence),
            components={cid: subset([i for i, row in enumerate(rows) if row['component_id'] == cid]) for cid in components},
            without_largest_component=subset([i for i, row in enumerate(rows) if row['component_id'] != largest]),
            no_expansion=True, model_forwards=0, optimizer_updates=0,
            limitations=['repeated V32 development, shared identities, not independent testing',
                'candidate AUROC/coverage is not new partition F1 or physical fusion accuracy',
                'LOO geometry is not calibrated likelihood or proof of conditional state information',
                'full pair geometry control preserves information; no matched learned heads were trained'])
        old.frozen.write(output/'scene-summary.json.gz', rows)
        checkpoint(phase='finalizing')
        result.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'])
        old.frozen.write(output/'summary.json', result)
        checkpoint(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
        return result
    except BaseException as error:
        checkpoint(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            exit_code=1, error=f'{type(error).__name__}: {error}', finished_at=datetime.now(timezone.utc).isoformat())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-system', 'partition-run', 'manifest', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException:
        traceback.print_exc(); return 1
    print(f'COMPLETED: {args.output}/summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
