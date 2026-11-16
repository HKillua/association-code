"""C148 fit pilot: which fixed-quota decisions can log transport potentials change?"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from association.data.archive import MTADArchive
from association.data.legacy_dataset import validate_archive
from association.data.motion_evidence import scene_motion_evidence
from association.data.physical_units import KNOT_TO_MPS
from tools.analysis import offline_baseline_core as c145
from tools.analysis.held_fusion_population import _canonical_tracks
from tools.analysis.offline_association_tail import scene_split
from tools.analysis.offline_baseline_assignment import _components, _minimum_cost
from tools.analysis.offline_baseline_assignment_run import edge_array, identity_weights, partition
from tools.analysis.offline_baseline_restart import digest, write


IDEA = 'IDEA-OFFLINE-ENDPOINT-QUALIFICATION-C148-20261116'
DATA = 'LEVEL1-FUSION-PEER-FULL-DATA-20261030-151-attempt-1'
PREP = 'LEVEL1-OFFLINE-ASSOCIATION-TAIL-PREP-20261112-attempt-1'
TEMPERATURE_METRES = 500.0
DUSTBIN_SCORE_METRES = -c145.ACCEPT_METRES
ITERATIONS = 20
ARMS = ('RAW_FIXED', 'POTENTIAL_FIXED')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def pair(i, j):
    return (min(i, j), max(i, j))


def endpoints(links):
    return {node for edge in links for node in edge}


def fixed_matching(left, right, edges, costs, quota):
    shift = max(0, -min(costs.values()))
    result, _, _ = _minimum_cost(left, right,
                                [(i, j, costs[pair(i, j)] + shift) for i, j, _ in edges],
                                exact_count=quota, stop_increment=0)
    return result


def integer_scores(distances, row_terms, column_terms):
    """Put the separately evaluated binary floats on a common exact scale."""
    values = [float(v).as_integer_ratio() for v in (*distances, *row_terms, *column_terms)]
    denominator = max(d for _, d in values)
    integers = [numerator * (denominator // divisor) for numerator, divisor in values]
    first, second = len(distances), len(distances) + len(row_terms)
    return integers[:first], integers[first:second], integers[second:]


def dustbin_potentials(left, right, edges):
    """Return additive cost terms, without assigning mass to unavailable real edges."""
    rows = {node: i for i, node in enumerate(left)}
    columns = {node: j for j, node in enumerate(right)}
    n, m = len(left), len(right)
    logits = np.full((n + 1, m + 1), DUSTBIN_SCORE_METRES / TEMPERATURE_METRES)
    logits[:n, :m] = -np.inf
    for i, j, distance in edges:
        logits[rows[i], columns[j]] = -distance / TEMPERATURE_METRES
    log_rows = np.log(np.r_[np.ones(n), m] / (n + m))
    log_columns = np.log(np.r_[np.ones(m), n] / (n + m))
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    for _ in range(ITERATIONS):
        u = log_rows - np.logaddexp.reduce(logits + v, axis=1)
        v = log_columns - np.logaddexp.reduce(logits + u[:, None], axis=0)
    fitted = logits + u[:, None] + v
    residual = max(np.max(np.abs(np.exp(np.logaddexp.reduce(fitted, axis=1)) - np.exp(log_rows))),
                   np.max(np.abs(np.exp(np.logaddexp.reduce(fitted, axis=0)) - np.exp(log_columns))))
    if (not (np.isfinite(u).all() and np.isfinite(v).all() and np.isfinite(residual))
            or np.any(np.exp(fitted[np.isfinite(logits)]) == 0)):
        raise ValueError('nonfinite dustbin normalization or underflow on an allowed edge')
    return -TEMPERATURE_METRES * u[:n], -TEMPERATURE_METRES * v[:m], float(residual)


def compare_block(left, right, edges, quota):
    distances = [d for _, _, d in edges]
    keys = [pair(i, j) for i, j, _ in edges]
    eligible = 0 < quota and (quota < len(left) or quota < len(right))
    row_terms, col_terms, residual = (dustbin_potentials(left, right, edges) if eligible
                                     else (np.zeros(len(left)), np.zeros(len(right)), None))
    unit, row_units, col_units = integer_scores(distances, row_terms, col_terms)
    raw_costs = dict(zip(keys, unit, strict=True))
    baseline = fixed_matching(left, right, edges, raw_costs, quota)
    result = baseline
    if eligible:
        row_unit = dict(zip(left, row_units, strict=True))
        col_unit = dict(zip(right, col_units, strict=True))
        adjusted = {pair(i, j): raw_costs[pair(i, j)] + row_unit[i] + col_unit[j]
                    for i, j, _ in edges}
        proposed = fixed_matching(left, right, edges, adjusted, quota)
        gain = sum(adjusted[p] for p in baseline) - sum(adjusted[p] for p in proposed)
        if gain < 0 or (gain > 0 and endpoints(proposed) == endpoints(baseline)):
            raise ValueError('fixed objective optimality or endpoint cancellation failed')
        if gain > 0:
            result = proposed
    return baseline, result, eligible, row_terms, col_terms, residual


def read_json(path):
    return json.loads(path.read_text())


def bind_population(runs, manifest_path):
    data, prep = runs / DATA, runs / PREP
    manifest = read_json(manifest_path)
    data_run, prep_run = read_json(data / 'run.json'), read_json(prep / 'run.json')
    expected_roles = read_json(prep / 'frozen-split.json')
    splits = manifest['splits']
    components = manifest['components']
    if (data_run['status'] != 'COMPLETED' or data_run['exit_code'] != 0
            or prep_run['status'] != 'COMPLETED' or prep_run['test_opened']
            or data_run['test_opened'] or data_run['completed_T_scenes'] != 2100
            or data_run['state_speed_unit'] != 'mps' or data_run['measurement_speed_unit'] != 'kn'
            or manifest != read_json(Path(data_run['manifest']))
            or expected_roles != scene_split(splits['T_core'], components)):
        raise ValueError('151 or C92 population provenance mismatch')
    fit = expected_roles['fit']
    component_sizes = Counter(components[sid] for sid in fit)
    if len(fit) != 1748 or len(set(fit)) != len(fit) or len(component_sizes) != 76:
        raise ValueError('original C92 fit population changed')
    withheld = set(expected_roles['calibration'] + expected_roles['held'] + splits['V_select'] + splits['T_test'])
    if set(fit) & withheld:
        raise ValueError('fit roles overlap withheld scenes')
    sampled = {}
    for sid in fit:
        sampled.setdefault(components[sid], sid)

    # The training groups in this index are derived from truth, and are never used.
    index = read_json(data / 'private/index.json')['T_core']
    for component, sid in sampled.items():
        row = index[sid]
        if any(Path(row[key]) != data / 'private' / key / 'T_core' / f'scene-{sid}.npz'
               for key in ('motion', 'evaluation')) or row['component_id'] != component:
            raise ValueError('C151 fit path or component does not bind to the scene')
    provenance = {name: digest(path) for name, path in (
        ('manifest_sha256', manifest_path), ('fit_split_sha256', prep / 'frozen-split.json'),
        ('motion_index_sha256', data / 'private/index.json'))}
    return manifest, data_run['archive'], sampled, component_sizes, provenance


def build_scene(archive, motion_path, sid):
    keys = ('node_ids', 'sources', 'query_seconds', 'support', 'nearest_seconds', 'native_count', 'states')
    with np.load(motion_path, allow_pickle=False) as file:
        if str(file['state_speed_unit']) != 'mps':
            raise ValueError('C151 explicit cache unit must be m/s')
        nodes, sources, query, support, nearest, counts, states = (file[key] for key in keys)
    if (nodes.ndim != 1 or len(np.unique(nodes)) != len(nodes) or sources.shape != nodes.shape
            or not set(sources.tolist()) <= {'9001', '9002'} or query.shape != (64,)
            or not np.isfinite(query).all() or (np.diff(query) <= 0).any()
            or support.shape != (len(nodes), 64) or support.dtype != bool
            or nearest.shape != support.shape or counts.shape != nodes.shape or states.shape != (*support.shape, 4)
            or not np.isfinite(states).all()):
        raise ValueError('C151 measurement schema or scene roster differs')
    tracks = _canonical_tracks(archive._rows(sid, 'measurements'), sid, nodes, sources)
    if tracks:
        rebuilt = scene_motion_evidence(dict(tracks=tracks, node_ids=nodes, sources=sources, grid=query),
                                        motion_observation_policy='canonical_all')
        np.testing.assert_array_equal(rebuilt['support'], support)
        np.testing.assert_array_equal(rebuilt['nearest_seconds'], nearest)
        np.testing.assert_array_equal(rebuilt['native_count'], counts)
        rebuilt['states'][..., 2] *= KNOT_TO_MPS
        np.testing.assert_array_equal(rebuilt['states'], states)
    times = np.concatenate([t['canonical_times'] for t in tracks]) if tracks else np.empty(0)
    values = np.concatenate([t['canonical_values'] for t in tracks]) if tracks else np.empty((0, 4))
    native = query, times, values[:, :2], np.r_[0, np.cumsum(counts)].astype(np.int64)
    return dict(nodes=nodes, sources=sources, query=query, support=support, native=native,
                values=values, states=states)


def freeze_scene(scene, path):
    nodes, sources, support = (scene[k] for k in ('nodes', 'sources', 'support'))
    before, after, scores, _ = c145._itsa_candidates(sources, scene['native'], scene['values'])
    chains, itsa = c145._itsa_chains(len(nodes), before, after, scores)
    chain_sources = sources[[c[0] for c in chains]] if chains else sources.copy()
    chain_support, positions = c145._chain_queries(chains, support, scene['states'])
    candidates, temporal = c145._mstsa_candidates(chain_sources, chain_support, positions)
    _, greedy_chains = c145._mstsa_partition(chains, chain_sources, candidates, len(nodes), 2)
    lookup = {chain: i for i, chain in enumerate(chains)}
    greedy = {pair(lookup[a], lookup[b]) for a, b in greedy_chains}
    allowed = {(i, j): distance for i, j, distance, _ in candidates if distance <= c145.ACCEPT_METRES}
    if (any(not np.isfinite(d) or d < 0 for _, _, d, _ in candidates)
            or len(allowed) != sum(d <= c145.ACCEPT_METRES for _, _, d, _ in candidates)
            or not greedy <= allowed.keys() or len(endpoints(greedy)) != 2 * len(greedy)):
        raise ValueError('C145 greedy does not define per-block valid quotas')

    links = {name: [] for name in ARMS}
    blocks, priors = [], np.zeros(len(chains))
    for root, left, right, edges in _components(tuple(chain_sources), allowed):
        keys = [pair(i, j) for i, j, _ in edges]
        quota = len(greedy.intersection(keys))
        raw, amended, eligible, row_term, col_term, residual = compare_block(left, right, edges, quota)
        for name, chosen in zip(ARMS, (raw, amended), strict=True):
            if not set(chosen) <= allowed.keys() or len(chosen) != quota or len(endpoints(chosen)) != 2 * quota:
                raise ValueError('fixed arm violates frozen eligibility or quota')
            links[name].extend(chosen)
        priors[left], priors[right] = row_term, col_term
        unmatched = sorted((set(left) | set(right)) - endpoints(raw)) if eligible else []
        added_endpoints = sorted(endpoints(amended) - endpoints(raw))
        removed_endpoints = sorted(endpoints(raw) - endpoints(amended))
        blocks.append(dict(root=root, quota=quota, candidate_edges=len(edges), eligible=eligible,
                           possible_added_endpoints=len(unmatched),
                           possible_added_fragments=sum(len(chains[i]) for i in unmatched),
                           possible_added_chain_queries=int(chain_support[unmatched].sum()),
                           added_endpoints=added_endpoints, removed_endpoints=removed_endpoints,
                           added_chain_queries=int(chain_support[added_endpoints].sum()),
                           removed_chain_queries=int(chain_support[removed_endpoints].sum()),
                           available_original_pairs=sum(len(chains[i]) * len(chains[j]) for i, j, _ in edges),
                           max_marginal_residual=residual))
    saved = {key: scene[key] for key in ('nodes', 'sources', 'query', 'support')}
    saved.update(chain_nodes=np.asarray([i for c in chains for i in c], np.int64),
                 chain_offsets=np.r_[0, np.cumsum([len(c) for c in chains], dtype=np.int64)],
                 chain_sources=chain_sources, chain_support=chain_support, endpoint_costs=priors,
                 candidate_pairs=edge_array((i, j) for i, j, _, _ in candidates),
                 distance_m=np.asarray([d for _, _, d, _ in candidates], np.float64),
                 synchronous_m=np.asarray([d for _, _, _, d in candidates], np.float64),
                 greedy_links=edge_array(sorted(greedy)), itsa=edge_array(itsa))
    for name, chosen in links.items():
        saved[f'{name}_membership'] = partition(chains, chosen, len(nodes))
        saved[f'{name}_links'] = edge_array(sorted(chosen))
    np.savez_compressed(path, **saved)
    return dict(nodes=len(nodes), chains=len(chains), temporal=temporal, gated=len(allowed), blocks=blocks)


def read_identity(path, nodes, query):
    with np.load(path, allow_pickle=False) as cached:
        np.testing.assert_array_equal(cached['node_ids'], nodes)
        np.testing.assert_array_equal(cached['query_seconds'], query)
        identities = cached['identities']
    if identities.shape != nodes.shape or identities.dtype.kind != 'U' or np.any(identities == ''):
        raise ValueError('C151 evaluation identity roster differs')
    return identities


def score_scene(saved, identities, blocks):
    support, sources = saved['support'], saved['sources']
    rows, cols = np.triu_indices(len(sources), 1)
    same = sources[rows] == sources[cols]
    truth = identities[rows] == identities[cols]
    cross = sources[:, None] != sources[None, :]
    same_identity = identities[:, None] == identities[None, :]
    missing = ~support

    def association(membership):
        chosen = membership[rows] == membership[cols]
        return {axis: dict(tp=int((mask & chosen & truth).sum()), fp=int((mask & chosen & ~truth).sum()),
                           fn=int((mask & ~chosen & truth).sum()))
                for axis, mask in (('itsa', same), ('mstsa', ~same))}

    def supplies(membership):
        paired = (membership[:, None] == membership[None, :]) & cross
        masks = (paired, paired & same_identity, paired & ~same_identity)
        return tuple((mask.astype(np.int64) @ support.astype(np.int64)) > 0 for mask in masks)

    base, changed = (saved[f'{name}_membership'] for name in ARMS)
    counts = Counter()
    for name, membership in zip(ARMS, (base, changed), strict=True):
        for scope, scores in association(membership).items():
            counts.update({f'{name}_{scope}_{key}': value for key, value in scores.items()})
        _, first = np.unique(membership, return_index=True)
        counts[f'{name}_represented'] = len(set(identities[first]))
        counts[f'{name}_duplicates'] = len(first) - counts[f'{name}_represented']
    if any(counts[f'{ARMS[0]}_itsa_{k}'] != counts[f'{ARMS[1]}_itsa_{k}'] for k in ('tp', 'fp', 'fn')):
        raise ValueError('same-source final pair decisions changed')
    for scope, old, new in zip(('any', 'same_ship', 'other_ship'), supplies(base), supplies(changed), strict=True):
        for region, mask in (('all', np.ones_like(missing)), ('missing', missing)):
            prefix = f'{scope}_{region}_support'
            counts.update({f'{prefix}_before': int((mask & old).sum()), f'{prefix}_after': int((mask & new).sum()),
                           f'{prefix}_lost': int((mask & old & ~new).sum()),
                           f'{prefix}_recovered': int((mask & ~old & new).sum())})

    base_links, changed_links = (set(map(tuple, saved[f'{name}_links'].tolist())) for name in ARMS)
    nodes_per_chain = np.diff(saved['chain_offsets'])
    weights = identity_weights(saved, identities) if len(identities) else {}
    for direction, pairs in (('added', changed_links - base_links), ('removed', base_links - changed_links)):
        truth_pairs = sum(weights.get(p, 0) for p in pairs)
        counts.update({f'{direction}_links': len(pairs), f'{direction}_true': truth_pairs,
                       f'{direction}_false': sum(int(nodes_per_chain[i] * nodes_per_chain[j])
                                                 for i, j in pairs) - truth_pairs})

    allowed = {tuple(p): float(d) for p, d in zip(saved['candidate_pairs'].tolist(), saved['distance_m'], strict=True)
               if d <= c145.ACCEPT_METRES}
    for spec, (_, left, right, edges) in zip(blocks, _components(tuple(saved['chain_sources']), allowed), strict=True):
        max_weight = max(weights.get(pair(i, j), 0) for i, j, _ in edges)
        cost = {pair(i, j): max_weight - weights.get(pair(i, j), 0) for i, j, _ in edges}
        oracle = fixed_matching(left, right, edges, cost, spec['quota'])
        counts['same_quota_true_pair_upper_bound'] += sum(weights.get(p, 0) for p in oracle)
    if (counts['same_quota_true_pair_upper_bound'] < max(counts[f'{name}_mstsa_tp'] for name in ARMS)
            or counts['added_true'] - counts['removed_true'] != counts[f'{ARMS[1]}_mstsa_tp'] - counts[f'{ARMS[0]}_mstsa_tp']
            or counts['added_false'] - counts['removed_false'] != counts[f'{ARMS[1]}_mstsa_fp'] - counts[f'{ARMS[0]}_mstsa_fp']):
        raise ValueError('same-quota or expanded changed-link accounting differs from actual groups')
    counts.update(blocks=len(blocks), eligible=sum(spec['eligible'] for spec in blocks),
                  changed=sum(bool(spec['added_endpoints']) for spec in blocks), query_points=int(support.size),
                  available_chain_pairs=len(allowed), missing_node_queries=int(missing.sum()))
    for field in ('possible_added_endpoints', 'possible_added_fragments', 'possible_added_chain_queries',
                  'added_chain_queries', 'removed_chain_queries', 'available_original_pairs'):
        counts[field] = sum(spec[field] for spec in blocks)
    counts['added_endpoints'] = sum(len(spec['added_endpoints']) for spec in blocks)
    counts['removed_endpoints'] = sum(len(spec['removed_endpoints']) for spec in blocks)
    return dict(counts)


def pool(rows):
    counts = Counter()
    for row in rows:
        counts.update(row['counts'])
    ratios = {}
    for name in ARMS:
        for scope in ('itsa', 'mstsa'):
            tp, fp, fn = (counts[f'{name}_{scope}_{key}'] for key in ('tp', 'fp', 'fn'))
            den = 2*tp + fp + fn
            ratios[f'{name}_{scope}_f1'] = (2*tp / den) if den else None
    return dict(**counts, **ratios, scenes=len(rows), changed_components=sum(row['counts']['changed'] > 0 for row in rows),
                eligible_block_share=counts['eligible'] / counts['blocks'] if counts['blocks'] else None)


def passes(row):
    return (all(row[f'{ARMS[1]}_mstsa_{key}'] < row[f'{ARMS[0]}_mstsa_{key}'] for key in ('fp', 'fn'))
            and row['same_ship_missing_support_lost'] == 0
            and row[f'{ARMS[1]}_represented'] >= row[f'{ARMS[0]}_represented']
            and row['any_all_support_after'] >= row['any_all_support_before'])


def run(args):
    start, started = time.monotonic(), utc_now()
    output, runs = args.output.resolve(), args.runs.resolve()
    output.mkdir()
    private = output / 'private'
    private.mkdir(mode=0o700)
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    state = dict(status='RUNNING', phase='bind', idea_id=IDEA, plan_version='1.0', code=code,
                 started=started, other_splits_opened=False, trained=False, predictions=0, scored=0)

    def checkpoint(**values):
        state.update(**values)
        write(output / 'run.json', dict(state, elapsed_seconds=time.monotonic() - start))

    checkpoint()
    try:
        manifest, paths, sampled, sizes, provenance = bind_population(runs, args.manifest.resolve())
        largest = min(sizes, key=lambda k: (-sizes[k], k))
        scene_ids = list(sampled.values())
        decisions, hashes, motion_hashes = {}, {}, {}
        data = runs / DATA / 'private'
        with MTADArchive(paths) as archive:
            validate_archive(archive, manifest)
            for count, sid in enumerate(scene_ids, 1):
                motion_path = data / 'motion/T_core' / f'scene-{sid}.npz'
                motion_hashes[sid] = digest(motion_path)
                scene = build_scene(archive, motion_path, sid)
                prediction = private / f'{sid}.npz'
                decisions[sid] = freeze_scene(scene, prediction)
                hashes[sid] = digest(prediction)
                checkpoint(phase='freeze', predictions=count)
            access = [(member['scene_id'], member['kind']) for member in archive.members_read]
        if access != [(sid, 'measurements') for sid in scene_ids]:
            raise ValueError('measurement-only archive scope changed')
        write(private / 'blocks.json.gz', decisions)
        write(output / 'freeze.json', dict(**provenance, code=code, population='C92 fit component-first',
              scene_ids=scene_ids, components=sampled, full_fit_scene_count=1748, largest_fit_component=largest,
              measurements_read=len(access), prediction_sha256=hashes, motion_sha256=motion_hashes,
              blocks_sha256=digest(private / 'blocks.json.gz'), temperature_m=TEMPERATURE_METRES,
              dustbin_m=DUSTBIN_SCORE_METRES, iterations=ITERATIONS, other_splits_opened=False,
              all_predictions_before_first_identity=True))
        rows, evaluation_hashes = [], {}
        for count, (component, sid) in enumerate(sampled.items(), 1):
            prediction = private / f'{sid}.npz'
            if digest(prediction) != hashes[sid]:
                raise ValueError('frozen prediction hash changed')
            evaluation = data / 'evaluation/T_core' / f'scene-{sid}.npz'
            evaluation_hashes[sid] = digest(evaluation)
            with np.load(prediction, allow_pickle=False) as saved:
                identities = read_identity(evaluation, saved['nodes'], saved['query'])
                row = score_scene(saved, identities, decisions[sid]['blocks'])
            residuals = [b['max_marginal_residual'] for b in decisions[sid]['blocks']
                         if b['max_marginal_residual'] is not None]
            rows.append(dict(scene_id=sid, component_id=component, fit_component_scene_count=sizes[component],
                             max_marginal_residual=max(residuals, default=None), counts=row))
            checkpoint(phase='evaluate', scored=count)
        overall = pool(rows)
        reduced = pool([row for row in rows if row['component_id'] != largest])
        write(output / 'scenes.json.gz', rows)
        write(output / 'summary.json', dict(full=overall, without_largest_fit_component=reduced,
            largest_fit_component=largest, evaluation_sha256=evaluation_hashes,
            qualifies_for_larger_fit_diagnostic=reduced['changed_components'] >= 2 and passes(overall) and passes(reduced)))
        checkpoint(status='COMPLETED', phase='completed', exit_code=0, finished=utc_now())
    except BaseException as error:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}', finished=utc_now())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--time-policy', choices=['advisory'], required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
