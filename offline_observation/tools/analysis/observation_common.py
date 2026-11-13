"""Shared immutable cache/evaluation plumbing for observation-only pilots.

No policy or reference-dependent selection lives here. Method lists and split
are explicit: historical full artifacts exist on V, not on T.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from association.evaluation.group_prediction import emit_group_outputs
from tools.analysis import source_risk_pilot as source

old, qc = source.old, source.qc
load_npz = source.load_npz
physical = source.physical
evaluation = source.evaluation


def metadata(native, value, partition, source_risk, manifest):
    native, value, partition, source_risk = map(Path, (native, value, partition, source_risk))
    tids, vids = source.metadata(native, value, partition, manifest)
    state = old.frozen.read(source_risk/'run.json')
    if (state.get('status') != 'COMPLETED' or state.get('exit_code') != 0
            or state.get('training_scene_ids') != tids or state.get('scene_ids') != vids
            or state.get('completed_training_scene_count') != 32
            or state.get('completed_scene_count') != 32):
        raise ValueError('133 completed original T32/V32 required')
    for key, expected in (('native_run', native), ('value_run', value), ('partition_run', partition)):
        if Path(state[key]).resolve() != expected.resolve():
            raise ValueError('133 original input lineage differs: '+key)
    return tids, vids


def membership(value, partition, split, sid):
    if split == 'T_core':
        return load_npz(Path(value)/'private'/f'T-features-{sid}.npz', ('membership',))['membership']
    if split == 'V_select':
        return np.asarray(old.frozen.read(Path(partition)/'private'/f'scene-{sid}.json.gz')['memberships']['WS'], np.int64)
    raise ValueError('only fixed T_core/V_select permitted')


def load_artifacts(path, method_names, template='WS_MEAN'):
    with np.load(path, allow_pickle=False) as stored:
        prefix = template+'_'
        fields = [k[len(prefix):] for k in stored.files if k.startswith(prefix)]
        if not fields or 'group_states' not in fields:
            raise ValueError('complete artifact template missing')
        return {name: {field: stored[name+'_'+field] for field in fields} for name in method_names}


def save_artifacts(path, artifacts):
    np.savez_compressed(path, **{name+'_'+key: value for name, artifact in artifacts.items()
                                for key, value in artifact.items()})


def evaluate_predictions(motion, groups, pack, ev, predictions, retained, *, method_names, split,
                         prediction_speed_unit=None):
    names = tuple(method_names)
    if not names or len(set(names)) != len(names) or set(predictions) != set(names):
        raise ValueError('exact unique declared predictions required')
    if split not in ('T_core', 'V_select'):
        raise ValueError('only fixed T_core/V_select permitted')
    if not set(retained).issubset(names):
        raise ValueError('retained controls outside declared population')
    fragment = dict(query_seconds=motion['query_seconds'], identities=ev['identities'],
                    support=motion['support'], nearest_seconds=motion['nearest_seconds'])
    regions = old.fixed_regions(fragment, ev['target_ids'])
    group_counts = np.asarray([pack['available'][a:b].sum(0)
                              for a, b in zip(pack['group_offsets'][:-1], pack['group_offsets'][1:])])
    counts = np.full((len(ev['target_ids']), motion['states'].shape[1]), -1, np.int64)
    registry = {s: i for i, s in enumerate(ev['target_ids'].tolist())}
    claimed = set()
    for gid, group in enumerate(groups):
        identity = ev['identities'][group[0]]
        if identity in registry and identity not in claimed:
            counts[registry[identity]] = group_counts[gid]
            claimed.add(identity)
    regions.update(source_support_zero=counts == 0, source_support_one=counts == 1,
                   source_support_multi=counts >= 2)
    artifacts, methods, arrays = {}, {}, {}
    for name in names:
        out = emit_group_outputs(predictions[name], groups, node_count=len(motion['states']))
        if prediction_speed_unit is None:
            report = old.evaluate_full_scene(out, node_identities=ev['identities'],
                target_identities=ev['target_ids'], target_states=ev['target_states'],
                target_valid=ev['target_valid'], node_observed=motion['support'], split=split,
                grouping_kind='predicted', correspondence_policy='anchor_identity_first',
                return_artifact=True,
                node_input_support=dict(support=motion['support'], nearest_seconds=motion['nearest_seconds']))
            artifact = report.pop('artifact')
        else:
            from association.evaluation.fusion_benchmark import evaluate_fusion_methods
            reports, saved = evaluate_fusion_methods({name: out}, prediction_speed_units={name: prediction_speed_unit},
                node_identities=ev['identities'], target_ids=ev['target_ids'], target_states=ev['target_states'],
                target_valid=ev['target_valid'], support=motion['support'], nearest_seconds=motion['nearest_seconds'],
                target_speed_unit='mps', split=split)
            report, artifact = reports[name], saved[name]
        if name in retained:
            if set(artifact) != set(retained[name]):
                raise ValueError('original artifact schema differs')
            for key in artifact:
                old._same(artifact[key], retained[name][key], 'original exact '+name+'/'+key)
        if artifacts:
            for key in artifact:
                if key not in ('group_states', 'aligned_prediction'):
                    old._same(artifact[key], artifacts[names[0]][key], 'fixed population '+name+'/'+key)
        methods[name], arrays[name] = qc.describe(report, artifact, fragment, regions)
        artifacts[name] = artifact
    pairs, pair_arrays = {}, {}
    for base, candidate in combinations(names, 2):
        key = candidate+'_vs_'+base
        pairs[key], pair_arrays[key], _ = old.compare_pair(artifacts[base], artifacts[candidate], regions)
        pairs[key].update(baseline_method=base, candidate_method=candidate,
                          side_labels=dict(A0=base, candidate=candidate))
    return dict(methods=methods, pairwise=pairs), dict(methods=arrays, pairs=pair_arrays), artifacts


def component_bootstrap(rows, arrays, primary):
    components = sorted({row['component_id'] for row in rows})
    sums, counts = [], []
    for component in components:
        delta = np.concatenate([a['pairs'][primary]['all']['delta'] for row, a in zip(rows, arrays)
                                if row['component_id'] == component])
        sums.append(float(delta.sum()))
        counts.append(len(delta))
    sums, counts = np.asarray(sums), np.asarray(counts)
    draws = np.random.default_rng(20).integers(0, len(components), size=(2000, len(components)))
    denominator = counts[draws].sum(1)
    if not np.all(denominator > 0):
        raise ValueError('bootstrap resample without evaluable points')
    deltas = sums[draws].sum(1)/denominator
    return dict(comparison=primary, components=components, component_count=len(components),
        resamples=2000, seed=20, estimator='common-position-point-weighted MAE difference; candidate minus baseline',
        point_estimate_m=float(sums.sum()/counts.sum()), percentile95_m=np.quantile(deltas, [.025, .975]).tolist(),
        independent_test=False)


def summarize(rows, arrays, *, method_names, primary_pairs):
    if not rows or len(rows) != len(arrays):
        raise ValueError('matching nonempty scene records required')
    components = sorted({row['component_id'] for row in rows})
    def aggregate(indices):
        return qc.aggregate([rows[i] for i in indices], [arrays[i] for i in indices], method_names=method_names)
    largest = max(components, key=lambda c: sum(row['component_id'] == c for row in rows))
    return dict(all_components=aggregate(range(len(rows))),
        by_component={c: aggregate([i for i, row in enumerate(rows) if row['component_id'] == c]) for c in components},
        without_largest_component=aggregate([i for i, row in enumerate(rows) if row['component_id'] != largest]),
        largest_component_id=largest,
        primary_bootstrap={pair: component_bootstrap(rows, arrays, pair) for pair in primary_pairs})
