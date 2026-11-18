"""MTAD C151P: paired association-only position smoothing on the C148 fit graph."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time

import numpy as np

from association.data.archive import MTADArchive
from association.data.legacy_dataset import validate_archive
from association.evaluation.association import _aggregate, _counts
from association.evaluation.native_motion import _short_delta, predict_native_positions
from tools.analysis import association_endpoint_qualification as c148
from tools.analysis import offline_baseline_core as c145
from tools.analysis.offline_baseline_assignment import _components
from tools.analysis.offline_baseline_assignment_run import edge_array, identity_weights, partition
from tools.analysis.offline_baseline_restart import digest, write


PARENT = 'LEVEL1-OFFLINE-ENDPOINT-QUALIFICATION-C148-20261116'
PARENT_FREEZE = '5506485e9dfcfabb9d7f1a19c5bd8e502c6db0f848ea1234e5306fd2b8759c9e'
PARENT_SUMMARY = '56b6c550a5ecf729bc058a5d8fd2a45e9ffe0caadbd76e1403d05ec5570dd902'
ARMS = ('RAW', 'SMOOTH')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_hash(path, expected):
    require(digest(path) == expected, f'archive digest mismatch: {path}')


def smooth_positions(times, original):
    """One simultaneous time-aware three-point average, confined to short intervals."""
    step = np.diff(times)
    require(len(times) >= 2 and original.shape == (len(times), 2) and np.all(step > 0)
            and np.isfinite(times).all() and np.isfinite(original).all(), 'invalid native track')
    result = original.copy()
    middle = np.flatnonzero((step[:-1] <= 60.) & (step[1:] <= 60.)) + 1
    a, b = original[middle - 1], original[middle + 1]
    fraction = step[middle - 1] / (step[middle - 1] + step[middle])
    expected_lat = a[:, 0] + fraction * (b[:, 0] - a[:, 0])
    expected_lon = a[:, 1] + fraction * _short_delta(b[:, 1] - a[:, 1])
    result[middle, 0] += .5 * (expected_lat - result[middle, 0])
    result[middle, 1] += .5 * _short_delta(expected_lon - result[middle, 1])
    result[middle, 1] = (result[middle, 1] + 180.) % 360. - 180.
    return result, len(middle)


def chain_positions(scene, chains):
    query, times, positions, offsets = scene['native']
    filtered = scene['states'][..., :2].copy()
    eligible = 0
    for node, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        t = times[lo:hi]
        smoothed, n = smooth_positions(t, positions[lo:hi])
        eligible += n
        if n:
            motion = predict_native_positions(smoothed[None], t[None], np.ones((1, len(t)), bool), query)
            support = scene['support'][node]
            np.testing.assert_array_equal(motion['regions']['inside_support'][0], support)
            filtered[node, support] = motion['predictions']['native_hold'][0, support]
    coverage, raw = c145._chain_queries(chains, scene['support'], scene['states'])
    filtered_support, new = c145._chain_queries(chains, scene['support'], filtered)
    np.testing.assert_array_equal(coverage, filtered_support)
    return coverage, raw, new, eligible


def score_existing_pairs(pairs, chain_support, positions):
    costs = np.empty(len(pairs), dtype=float)
    # Different joint Q counts cannot share a rectangular DTW batch.
    for start in range(0, len(pairs), 512):
        left, right = pairs[start:start+512].T
        common = chain_support[left] & chain_support[right]
        length = common.sum(axis=1)
        require((length >= c145.MIN_COMMON_QUERIES).all(), 'unqualified pair in frozen graph')
        for n in np.unique(length):
            take = np.flatnonzero(length == n)
            joint = common[take]
            a = positions[left[take]][joint].reshape(-1, n, 2)
            b = positions[right[take]][joint].reshape(-1, n, 2)
            costs[start+take] = c145._dtw_path_mean(a, b)
    require(np.isfinite(costs).all() and (costs >= 0.).all(), 'invalid frozen-pair DTW cost')
    return costs


def matched_links(sources, pairs, raw_cost, new_cost, old_links):
    """Keep the old graph, block cardinalities and old matching on exact objective ties."""
    graph = {tuple(edge): float(d) for edge, d in zip(pairs, raw_cost, strict=True)
             if d <= c145.ACCEPT_METRES}
    costs = dict(zip(map(tuple, pairs), new_cost, strict=True))
    baseline = set(map(tuple, old_links))
    require(baseline <= graph.keys(), 'invalid archived links')
    chosen = []
    for _, left, right, edges in _components(tuple(sources), graph):
        keys = [c148.pair(i, j) for i, j, _ in edges]
        old = baseline.intersection(keys)
        integers, _, _ = c148.integer_scores([costs[k] for k in keys], (), ())
        weights = dict(zip(keys, integers, strict=True))
        proposed = c148.fixed_matching(left, right, edges, weights, len(old))
        old_sum = sum(weights[k] for k in old)
        new_sum = sum(weights[k] for k in proposed)
        require(len(proposed) == len(old) and new_sum <= old_sum, 'fixed-quota objective mismatch')
        chosen.extend(proposed if new_sum < old_sum else old)
    require(len(chosen) == len(baseline), 'matching cardinality changed')
    return edge_array(sorted(chosen))


def freeze_scene(scene, parent, path):
    for key in ('support', 'nodes', 'sources', 'query'):
        np.testing.assert_array_equal(scene[key], parent[key])
    chains = np.split(parent['chain_nodes'], parent['chain_offsets'][1:-1]) if len(scene['nodes']) else []
    coverage, raw, filtered, eligible = chain_positions(scene, chains)
    np.testing.assert_array_equal(coverage, parent['chain_support'])
    pairs = parent['candidate_pairs']
    raw_scores = score_existing_pairs(pairs, coverage, raw)
    np.testing.assert_array_equal(raw_scores, parent['distance_m'])
    smooth_scores = score_existing_pairs(pairs, coverage, filtered)
    original_links = parent['RAW_FIXED_links']
    new_links = matched_links(parent['chain_sources'], pairs, raw_scores, smooth_scores, original_links)
    raw_members = partition(chains, original_links, len(scene['nodes']))
    np.testing.assert_array_equal(raw_members, parent['RAW_FIXED_membership'])
    keys = ('nodes', 'sources', 'query', 'support', 'chain_nodes', 'chain_offsets', 'chain_sources', 'candidate_pairs')
    saved = {k: parent[k] for k in keys}
    np.savez_compressed(path, **saved, RAW_cost=raw_scores, SMOOTH_cost=smooth_scores,
                        RAW_membership=raw_members, SMOOTH_membership=partition(chains, new_links, len(scene['nodes'])),
                        RAW_links=original_links, SMOOTH_links=new_links)
    return dict(native_reports=len(scene['values']), eligible_interior_reports=eligible,
                geometric_chain_pairs=len(pairs), original_accepted_graph=int((raw_scores <= c145.ACCEPT_METRES).sum()),
                score_changes=int((np.abs(raw_scores-smooth_scores) > 1e-9).sum()))


def evaluate(saved, identities):
    sources, support = saved['sources'], saved['support']
    left, right = np.triu_indices(len(sources), 1)
    actual = identities[left] == identities[right]
    cross = sources[left] != sources[right]
    cross_matrix = sources[:, None] != sources
    same_ship = identities[:, None] == identities
    comparisons, supply = {}, {}
    changed = not np.array_equal(saved['RAW_membership'], saved['SMOOTH_membership'])
    for arm in ARMS:
        group = saved[f'{arm}_membership']
        linked = group[:, None] == group
        decisions = linked[left, right]
        metrics = {}
        for scope, mask in [('MSTSA', cross), ('ITSA', ~cross)]:
            metrics[scope] = dict(pair_count=int(mask.sum()), positive_count=int(actual[mask].sum()),
                                  **_counts(actual[mask], decisions[mask]))
        _, anchor = np.unique(group, return_index=True)
        representatives = len(set(identities[anchor]))
        comparisons[arm] = dict(**metrics, represented=representatives, duplicates=len(anchor)-representatives)
        partners = linked & cross_matrix
        supply[arm] = {key: (mask.astype(np.int32) @ support.astype(np.int32)) > 0 for key, mask in
                       [('any', partners), ('same_ship', partners & same_ship), ('other_ship', partners & ~same_ship)]}
    require(comparisons['RAW']['ITSA'] == comparisons['SMOOTH']['ITSA'], 'same-source output changed')
    counts = {}
    for key in supply['RAW']:
        old, new = supply['RAW'][key], supply['SMOOTH'][key]
        for scope, mask in [('all', np.ones_like(support)), ('missing', ~support)]:
            prefix = f'{key}_{scope}'
            counts.update({prefix+'_before': int((old & mask).sum()), prefix+'_after': int((new & mask).sum()),
                           prefix+'_lost': int((old & ~new & mask).sum()), prefix+'_restored': int((~old & new & mask).sum())})
    pairs = saved['candidate_pairs']
    weights = identity_weights(saved, identities) if len(identities) else {}
    positives = np.array([weights.get(tuple(k), 0) for k in pairs], dtype=np.int64)
    sizes = np.diff(saved['chain_offsets'])
    negatives = sizes[pairs[:, 0]] * sizes[pairs[:, 1]] - positives
    require(np.all(negatives >= 0), 'mixed-chain pair weights invalid')
    evidence = dict(positive=positives, negative=negatives, **{a: saved[f'{a}_cost'] for a in ARMS})
    return dict(arms=comparisons, supply=counts, changed=changed), evidence


def ranking(cost, positive, negative):
    order = np.argsort(cost, kind='stable')
    d = cost[order]
    ends = np.flatnonzero(np.r_[d[1:] != d[:-1], True]) if len(d) else np.array([], int)
    tp = np.cumsum(positive[order], dtype=np.int64)[ends]
    fp = np.cumsum(negative[order], dtype=np.int64)[ends]
    pos, neg = int(positive.sum()), int(negative.sum())
    tpr = tp / pos if pos else np.zeros(len(tp))
    fpr = fp / neg if neg else np.zeros(len(fp))
    precision = tp / (tp + fp)
    def point(i):
        ntrue, nfalse = (int(tp[i]), int(fp[i])) if i >= 0 else (0, 0)
        return dict(max_distance_m=float(d[ends[i]]) if i >= 0 else None, tp=ntrue, fp=nfalse,
                    true_retained=ntrue/pos if pos else None, false_rejected=1.-nfalse/neg if neg else None,
                    precision=ntrue/(ntrue+nfalse) if ntrue+nfalse else None)
    curve = dict(distance_m=d[ends], true_positive=tp, false_positive=fp)
    summary = dict(candidate_positive=pos, candidate_negative=neg,
                   roc_auc=float(np.sum(np.diff(np.r_[0., fpr]) * (tpr + np.r_[0., tpr][:-1]) / 2)) if pos and neg else None,
                   average_precision=float(np.sum(np.diff(np.r_[0., tpr]) * precision)) if pos else None,
                   at_5000=point(np.searchsorted(d[ends], c145.ACCEPT_METRES, side='right')-1),
                   at_recall={str(x): point(np.searchsorted(tpr, x)) if pos else None for x in (.95, .99)})
    return summary, curve


def summarize(rows, evidence, scope, curves):
    output = dict(scene_count=len(rows), changed_components=sum(r['changed'] for r in rows))
    output['supply'] = {k: sum(r['supply'][k] for r in rows) for k in rows[0]['supply']}
    output['input_counts'] = {k: sum(r['input_counts'][k] for r in rows) for k in rows[0]['input_counts']}
    for arm in ARMS:
        a = [r['arms'][arm] for r in rows]
        result = {rel: _aggregate([r[rel] for r in a]) for rel in ('ITSA', 'MSTSA')}
        result.update({k: sum(r[k] for r in a) for k in ('represented', 'duplicates')})
        rank, curve = ranking(*(np.concatenate([ev[key] for ev in evidence]) for key in (arm, 'positive', 'negative')))
        total = result['MSTSA']
        rank['outside_pool_positive'] = total['tp'] + total['fn'] - rank['candidate_positive']
        rank['outside_pool_negative'] = total['tn'] + total['fp'] - rank['candidate_negative']
        require(min(rank['outside_pool_positive'], rank['outside_pool_negative']) >= 0, 'pool exceeds scene population')
        result['ranking'] = rank
        curves.update({f'{scope}_{arm}_{key}': values for key, values in curve.items()})
        output[arm] = result
    raw, new = output['RAW'], output['SMOOTH']
    support = output['supply']
    output['meets_per_scope_conditions'] = bool(new['MSTSA']['fp'] < raw['MSTSA']['fp']
        and new['MSTSA']['fn'] < raw['MSTSA']['fn'] and support['same_ship_missing_lost'] == 0
        and new['represented'] >= raw['represented'] and support['any_all_after'] >= support['any_all_before'])
    return output


def run(args):
    start = time.monotonic()
    runs, output = args.runs.resolve(), args.output.resolve()
    output.mkdir()
    private = output / 'private'
    private.mkdir(mode=0o700)
    state = dict(idea='IDEA-OFFLINE-C151P-LOCAL-POSITION', plan_version='1.0', status='RUNNING',
                 started=c148.utc_now(), code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 predictions=0, scored=0, trained=False, other_splits_opened=False)
    def checkpoint(**fields):
        state.update(fields)
        write(output/'run.json', dict(**state, elapsed_seconds=time.monotonic()-start))

    try:
        checkpoint(phase='bind')
        manifest, archives, sampled, sizes, provenance = c148.bind_population(runs, args.manifest.resolve())
        parent = runs / (PARENT+'-attempt-1')
        verify_hash(parent/'freeze.json', PARENT_FREEZE)
        frozen = c148.read_json(parent/'freeze.json')
        original = c148.read_json(parent/'run.json')
        launch = c148.read_json(runs/(PARENT+'-launch-attempt-1')/'status.json')
        require(original['status'] == launch['status'] == 'COMPLETED' and original['exit_code'] == launch['returncode'] == 0
                and original['code'] == frozen['code'] and original['scored'] == len(sampled) == 76, 'invalid C148 parent')
        require(list(sampled.values()) == frozen['scene_ids'] and frozen['components'] == sampled
                and all(frozen[k] == v for k,v in provenance.items()), 'C148 population drift')
        largest = min(sizes, key=lambda c: (-sizes[c], c))
        require(largest == frozen['largest_fit_component'], 'largest fit component differs')
        hashes, inputs = {}, {}
        data = runs / c148.DATA / 'private'
        with MTADArchive(archives) as archive:
            validate_archive(archive, manifest)
            for sid in sampled.values():
                motion = data/'motion/T_core'/f'scene-{sid}.npz'
                old = parent/'private'/f'{sid}.npz'
                verify_hash(old, frozen['prediction_sha256'][sid])
                verify_hash(motion, frozen['motion_sha256'][sid])
                scene = c148.build_scene(archive, motion, sid)
                path = private/f'{sid}.npz'
                with np.load(old, allow_pickle=False) as saved:
                    inputs[sid] = freeze_scene(scene, saved, path)
                hashes[sid] = digest(path)
                checkpoint(phase='freeze', predictions=len(hashes))
            require([(r['scene_id'],r['kind']) for r in archive.members_read] ==
                    [(sid,'measurements') for sid in sampled.values()], 'unexpected archive access')
        write(output/'freeze.json', dict(source_freeze_sha256=PARENT_FREEZE, prediction_sha256=hashes,
              components=sampled, input_counts=inputs, all_predictions_before_identity=True, measurements_read=len(hashes)))
        hashes = c148.read_json(output/'freeze.json')['prediction_sha256']
        verify_hash(parent/'summary.json', PARENT_SUMMARY)
        prior_results = c148.read_json(parent/'summary.json')
        rows, weights, eval_hashes = [], [], {}
        for component, sid in sampled.items():
            path = private/f'{sid}.npz'
            verify_hash(path, hashes[sid])
            truth = data/'evaluation/T_core'/f'scene-{sid}.npz'
            verify_hash(truth, prior_results['evaluation_sha256'][sid])
            eval_hashes[sid] = prior_results['evaluation_sha256'][sid]
            with np.load(path, allow_pickle=False) as saved:
                identities = c148.read_identity(truth, saved['nodes'], saved['query'])
                row, values = evaluate(saved, identities)
            row.update(scene_id=sid, component=component, input_counts=inputs[sid])
            rows.append(row)
            weights.append(values)
            checkpoint(phase='evaluate', scored=len(rows))
        indices = [i for i,r in enumerate(rows) if r['component'] != largest]
        curves = {}
        full = summarize(rows, weights, 'full', curves)
        reduced = summarize([rows[i] for i in indices], [weights[i] for i in indices], 'reduced', curves)
        for scope, result in [('full',full),('without_largest_fit_component',reduced)]:
            prior = prior_results[scope]
            for rel in ('MSTSA','ITSA'):
                for k in ('tp','fp','fn'):
                    require(result['RAW'][rel][k] == prior[f'RAW_FIXED_{rel.lower()}_{k}'], 'C148 control changed')
            require(result['supply']['same_ship_missing_before'] == prior['same_ship_missing_support_before'],
                    'C148 missing-slot coverage changed')
        np.savez_compressed(output/'curves.npz', **curves)
        write(output/'scenes.json.gz', rows)
        write(output/'summary.json', dict(full=full, without_largest=reduced, largest_fit_component=largest,
              evaluation_sha256=eval_hashes, qualifies=(full['meets_per_scope_conditions'] and
              reduced['meets_per_scope_conditions'] and reduced['changed_components'] >= 2)))
        checkpoint(phase='done',status='COMPLETED',finished=c148.utc_now(),exit_code=0)
    except BaseException as exc:
        checkpoint(phase='failed',status='FAILED',finished=c148.utc_now(),exit_code=1,error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('runs', 'manifest', 'output'):
        parser.add_argument('--'+arg, required=True, type=Path)
    parser.add_argument('--time-policy', required=True, choices=['advisory'])
    run(parser.parse_args())
