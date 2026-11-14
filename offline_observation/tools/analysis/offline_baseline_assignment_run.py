"""C146: replay the frozen C145 cross-source graph and compare assignment rules."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from association.data.physical_units import KNOT_TO_MPS
from association.evaluation.fusion_controls import ordinary_fusion
from tools.analysis import held_fusion_population as population
from tools.analysis import offline_baseline_core as c145
from tools.analysis.offline_baseline_assignment import assign, maximum_pair_weight
from tools.analysis.offline_baseline_restart import aggregate, digest, output_groups, score_outputs, write


C145_RUN = 'LEVEL1-OFFLINE-BASELINE-RESTART-20261114-attempt-1'
ARMS = ('GREEDY', 'GLOBAL_FIXED', 'EXPLICIT_NULL')
REGISTRY = ('nodes', 'query', 'support', 'sources')


def edge_array(edges):
    return np.asarray(list(edges), dtype=np.int64).reshape(-1, 2)


def partition(chains, links, node_count):
    """Emit canonical labels for a matching over the already frozen ITSA chains."""
    parent = list(range(len(chains)))
    used = set()
    for i, j in links:
        if i >= j or i in used or j in used:
            raise ValueError('invalid cross-chain matching')
        parent[j] = i
        used.update((i, j))
    membership = np.empty(node_count, np.int64)
    roots = {}
    for i, chain in enumerate(chains):
        membership[list(chain)] = roots.setdefault(parent[i], len(roots))
    return membership


def states_for(states, nearest, support, membership):
    groups = output_groups(membership, len(states))
    fusion = ordinary_fusion(states, nearest, support, groups)
    selected = fusion['predictions']['mean_supported'][[group[0] for group in groups]]
    selected[..., 2] *= KNOT_TO_MPS
    return selected


def read_motion(runs, sid, old):
    path = runs / population.PARENTS['mean'] / 'private/native-motion' / f'scene-{sid}.npz'
    validated = runs / population.PARENTS['full_data'] / 'private/motion' / population.SPLIT / f'scene-{sid}.npz'
    with np.load(path, allow_pickle=False) as raw, np.load(validated, allow_pickle=False) as checked:
        for key, original in zip(('node_ids', 'query_seconds', 'support', 'sources'), REGISTRY, strict=True):
            np.testing.assert_array_equal(raw[key], old[original])
            np.testing.assert_array_equal(checked[key], old[original])
        nearest, states = raw['nearest_seconds'], raw['states']
        np.testing.assert_array_equal(nearest, checked['nearest_seconds'])
        np.testing.assert_array_equal(raw['native_count'], checked['native_count'])
        converted = states.copy()
        converted[..., 2] *= KNOT_TO_MPS
        np.testing.assert_array_equal(converted, checked['states'])
        if str(checked['state_speed_unit']) != 'mps':
            raise ValueError('C151 speed unit must be m/s')
    return states, nearest


def freeze_scene(runs, sid, old_path, expected, path):
    with np.load(old_path, allow_pickle=False) as old:
        if set(old['sources'].tolist()) != {'9001', '9002'}:
            raise ValueError('the C145 roster requires its two declared sources')
        states, nearest = read_motion(runs, sid, old)
        saved = {key: old[key] for key in REGISTRY}
        itsa = old['END_DTW_ITSA']
        node_count = len(saved['nodes'])
        chains, rebuilt = c145._itsa_chains(node_count, itsa[:, 0], itsa[:, 1], np.zeros(len(itsa)))
        if set(map(tuple, itsa.tolist())) != set(rebuilt) or sorted(n for c in chains for n in c) != list(range(node_count)):
            raise ValueError('stored directed C145 decisions do not define the original chains')
        chain_sources = saved['sources'][[chain[0] for chain in chains]]
        if any(np.any(saved['sources'][list(chain)] != source)
               for chain, source in zip(chains, chain_sources, strict=True)):
            raise ValueError('C145 same-source chain contains different sources')
        coverage, positions = c145._chain_queries(chains, saved['support'], states)
        candidates, time_qualified = c145._mstsa_candidates(chain_sources, coverage, positions)
        greedy_membership, greedy = c145._mstsa_partition(chains, chain_sources, candidates, node_count, 2)
        np.testing.assert_array_equal(greedy_membership, old['END_DTW_membership'])
        np.testing.assert_array_equal(edge_array((i, j) for a, b in greedy for i in a for j in b),
                                      old['END_DTW_MSTSA'])
        keys = {chain: i for i, chain in enumerate(chains)}
        greedy_links = [(keys[a], keys[b]) for a, b in greedy]
        pure_candidates = [(i, j, dtw) for i, j, dtw, _ in candidates]
        new, components = assign(chain_sources, pure_candidates, greedy_links)
        expected_fields = dict(itsa_accepted=len(itsa), same_source_chains=len(chains),
                               mstsa_time_qualified=time_qualified, mstsa_candidates=len(candidates),
                               mstsa_under_accept_limit=sum(d <= c145.ACCEPT_METRES for _, _, d in pure_candidates),
                               mstsa_accepted=len(greedy_links), output_groups=len(np.unique(greedy_membership)),
                               singleton_groups=int((np.bincount(greedy_membership) == 1).sum()))
        if any(expected[key] != value for key, value in expected_fields.items()):
            raise ValueError('C145 same-source or cross-source stage counts differ')
        for name, links in {'GREEDY': greedy_links, **new}.items():
            membership = partition(chains, links, node_count)
            outputs = states_for(states, nearest, saved['support'], membership)
            saved[f'{name}_membership'] = membership
            saved[f'{name}_states'] = outputs
            saved[f'{name}_ITSA'] = itsa
            saved[f'{name}_MSTSA'] = edge_array((a, b) for i, j in links for a in chains[i] for b in chains[j])
            saved[f'{name}_links'] = edge_array(links)
            if name == 'GREEDY':
                np.testing.assert_array_equal(membership, greedy_membership)
                np.testing.assert_array_equal(outputs, old['END_DTW_states'])
        saved.update(chain_nodes=np.concatenate(chains), chain_offsets=np.r_[0, np.cumsum([len(c) for c in chains])],
                     chain_sources=chain_sources, chain_support=coverage,
                     candidate_pairs=edge_array((i, j) for i, j, _ in pure_candidates),
                     distance_m=np.asarray([d for _, _, d in pure_candidates], dtype=np.float64),
                     synchronous_m=np.asarray([diagonal for _, _, _, diagonal in candidates], dtype=np.float64))
    np.savez_compressed(path, **saved)
    return dict(**expected_fields, candidate_components=components)


def identity_weights(saved, identities):
    chains = np.split(saved['chain_nodes'], saved['chain_offsets'][1:-1])
    chain_part = {}
    for i, nodes in enumerate(chains):
        for identity, count in Counter(identities[nodes].tolist()).items():
            chain_part.setdefault(identity, []).append((i, count))
    sources = saved['chain_sources']
    weights = Counter()
    for carriers in chain_part.values():
        for index, (i, n) in enumerate(carriers):
            for j, m in carriers[index + 1:]:
                if sources[i] != sources[j]:
                    weights[i, j] += n * m
    return weights


def capacity_and_changes(saved, identities, scores):
    weights = identity_weights(saved, identities)
    support = saved['chain_support']
    geometric = {tuple(edge): float(d) for edge, d in
                 zip(saved['candidate_pairs'].tolist(), saved['distance_m'], strict=True)}
    temporal = {edge for edge in weights if np.count_nonzero(support[edge[0]] & support[edge[1]]) >= 2}
    above = {edge for edge in weights if edge in geometric}
    allowed = {edge for edge in above if geometric[edge] <= c145.ACCEPT_METRES}
    if not allowed <= above <= temporal:
        raise ValueError('C145 stage reachability is not nested')
    stages = dict(all_true=sum(weights.values()),
                  temporal_true=sum(weights[edge] for edge in temporal),
                  geometry_true=sum(weights[edge] for edge in above),
                  dtw_true=sum(weights[edge] for edge in allowed),
                  capacity_true=maximum_pair_weight(saved['chain_sources'], [(i, j, weights[i, j]) for i, j in allowed]))
    losses = dict(time=stages['all_true']-stages['temporal_true'],
                  geometry=stages['temporal_true']-stages['geometry_true'],
                  dtw=stages['geometry_true']-stages['dtw_true'],
                  one_to_one=stages['dtw_true']-stages['capacity_true'])
    nodes_per_chain = np.diff(saved['chain_offsets'])
    links = {name: set(map(tuple, saved[f'{name}_links'].tolist())) for name in ARMS}
    effects = {}
    for name in ARMS:
        row = scores[name]['association']['MSTSA']
        actual = sum(weights[pair] for pair in links[name])
        gap = stages['capacity_true'] - actual
        if min(gap, *losses.values()) < 0 or row['tp'] != actual or sum(losses.values()) + gap != row['fn']:
            raise ValueError('cross-source native-pair identity accounting differs from scored closure')
        effects[name] = dict(actual_true=actual, residual_capacity_gap=gap, links=len(links[name]),
                             false_positive=row['fp'], false_negative=row['fn'])
    changes = {}
    for old, new in zip(ARMS, ARMS[1:]):
        differences = {}
        for label, pairs in (('added', links[new] - links[old]), ('removed', links[old] - links[new])):
            tp = sum(weights[pair] for pair in pairs)
            all_pairs = sum(int(nodes_per_chain[i] * nodes_per_chain[j]) for i, j in pairs)
            differences[label] = dict(chain_links=len(pairs), true_pairs=tp, false_pairs=all_pairs-tp)
        changes[new] = differences
    return dict(stages=stages, gate_losses=losses, arms=effects, changes=changes)


def merge_counts(records):
    first = records[0]
    return {key: (merge_counts([row[key] for row in records]) if isinstance(value, dict)
                  else sum(row[key] for row in records)) for key, value in first.items()}


def report(rows, *, paper=True):
    return dict(arms=aggregate([r['metrics'] for r in rows], paper=paper, arms=ARMS),
                decomposition=merge_counts([r['decomposition'] for r in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--time-policy', choices=['advisory'], required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    wall = time.monotonic()
    runs, output = args.runs.resolve(), args.output.resolve()
    old = runs / C145_RUN
    old_status = json.loads((old/'run.json').read_text())
    old_freeze = json.loads((old/'freeze.json').read_text())
    old_rows = population.read(old/'scenes.json.gz')
    ids, components, _ = population.load_full_population(runs)
    if (old_status['status'] != 'COMPLETED' or old_status['exit_code'] != 0 or old_freeze['test_opened']
            or old_freeze['scene_ids'] != ids or old_freeze['components'] != components
            or [row['scene_id'] for row in old_rows] != ids
            or any(row['component_id'] != components[row['scene_id']] for row in old_rows)):
        raise ValueError('completed C145 V300 and original component order required')
    output.mkdir()
    private = output/'private'
    private.mkdir(mode=0o700)
    errors = private/'errors'
    errors.mkdir()
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    write(output/'run.json', dict(status='PREDICTING', code=code, started=started.isoformat(), scenes=len(ids)))
    hashes, decisions = {}, {}
    for index, (sid, old_row) in enumerate(zip(ids, old_rows, strict=True), 1):
        old_path = old/'private'/f'scene-{sid}.npz'
        if digest(old_path) != old_freeze['sha256'][sid]:
            raise ValueError('C145 frozen input file has changed')
        path = private/f'scene-{sid}.npz'
        decisions[sid] = freeze_scene(runs, sid, old_path, old_row['candidate_diagnostics']['END_DTW'], path)
        hashes[sid] = digest(path)
        if index % 25 == 0:
            print(f'prediction {index}/{len(ids)}', flush=True)
    write(output/'freeze.json', dict(code=code, source_c145=str(old), source_code=old_freeze['code'],
          scene_ids=ids, components=components, split=population.SPLIT, sha256=hashes, cache_only=True, test_opened=False))
    write(output/'run.json', dict(status='SCORING', code=code, started=started.isoformat(), scenes=len(ids)))
    rows, per_scene = [], []
    for index, (sid, old_row) in enumerate(zip(ids, old_rows, strict=True), 1):
        path = private/f'scene-{sid}.npz'
        if digest(path) != hashes[sid]:
            raise ValueError('frozen C146 prediction changed before evaluation')
        with np.load(path, allow_pickle=False) as saved:
            reference = population.read_evaluation(runs, sid, saved['nodes'], saved['query'], scene_ids=ids)
            metrics = score_outputs(saved, reference, sid, components[sid], ARMS)
            decomposition = capacity_and_changes(saved, reference['identities'], metrics)
        record = dict(metrics=metrics, decomposition=decomposition)
        scene = report([record], paper=False)
        replay = scene['arms']['GREEDY']
        original = old_row['arms']['END_DTW']
        if (replay['fusion']['own'] != original['fusion']['own'] or replay['association'] != original['association']
                or any(replay[k] != original[k] for k in ('groups', 'duplicates', 'unmatched_targets'))):
            raise ValueError('C145 historical greedy scoring failed to replay')
        for name in ARMS[1:]:
            if any(scene['arms'][name]['association'][key] != replay['association'][key]
                   for key in ('ITSA', 'ITSA_direct')):
                raise ValueError('a cross-source assignment changed same-source edges')
        rows.append(record)
        per_scene.append(dict(scene_id=sid, component_id=components[sid], **scene, assignment=decisions[sid]))
        np.savez_compressed(errors/f'scene-{sid}.npz',
                            **{f'{arm}_{scope}': metrics[arm][scope] for arm in ARMS for scope in ('own', 'common')})
        if index % 25 == 0:
            print(f'evaluation {index}/{len(ids)}', flush=True)
    sizes = Counter(components.values())
    largest = min(sizes, key=lambda key: (-sizes[key], key))
    by_component = {key: [row for sid, row in zip(ids, rows, strict=True) if components[sid] == key]
                    for key in sorted(sizes)}
    write(output/'scenes.json.gz', per_scene)
    write(output/'summary.json', dict(**report(rows), by_component={
              key: report(r, paper=False) for key, r in by_component.items()}, component_sizes=dict(sizes),
          without_largest=report([row for sid, row in zip(ids, rows, strict=True) if components[sid] != largest]),
          excluded_largest_component=largest, scene_count=len(rows), updates=0, test_opened=False))
    write(output/'run.json', dict(status='COMPLETED', exit_code=0, code=code, scenes=len(rows), split=population.SPLIT,
          started=started.isoformat(), ended=datetime.now(timezone.utc).isoformat(),
          elapsed_seconds=time.monotonic()-wall, time_policy=args.time_policy, updates=0))


if __name__ == '__main__':
    main()
