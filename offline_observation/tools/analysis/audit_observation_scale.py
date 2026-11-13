"""Independent saved-observation selection/position audit for experiment 137.

Uses NumPy and independent auditing helpers only. It never calls the production
CV predictor, held-out splitter, selector, readout or evaluator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from tools.analysis import audit_observation_common as shared

core = shared.core
RHOS = tuple(10.**k for k in range(-10, -1))
IMPORTED = ('WS_MEAN', 'WS_NEAREST_SUPPORTED', 'WS_LOW_SOURCE_SUPPORTED',
    'WS_FIXED_LOW_SOURCE_SUPPORTED', 'WS_QUADRATIC_VALUE_GATE', 'WS_SOURCE_EQUAL',
    'WS_SHARED_Q_SOFT', 'WS_SHARED_Q_HARD', 'WS_SHARED_LINEAR_SOFT', 'WS_SHARED_SHUFFLED_SOFT')
HISTORICAL = ('WS_CV_FORWARD_MEAN', 'WS_CV_FORWARD_LOW', 'WS_CV_RTS_MEAN', 'WS_CV_RTS_LOW')
NEW = ('WS_REFERENCE_SCALE_FORWARD_EQUAL', 'WS_REFERENCE_SCALE_RTS_EQUAL',
       'WS_OBS_SCALE_FORWARD_EQUAL', 'WS_OBS_SCALE_RTS_EQUAL')
T_METHODS, V_METHODS = ('WS_SOURCE_EQUAL',)+NEW, IMPORTED+HISTORICAL+NEW
PRIMARY = (NEW[3]+'_vs_WS_SOURCE_EQUAL', NEW[3]+'_vs_'+NEW[1])
IDEA = 'IDEA-OBSERVATION-SCALE-20261030'


def load(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def distances(a, b):
    a, b = np.broadcast_arrays(a, b)
    def xyz(value):
        lat, lon = np.deg2rad(value[..., 0]), np.deg2rad(value[..., 1])
        return np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), axis=-1)
    x, y = xyz(a), xyz(b)
    return 6371000.*np.arctan2(np.linalg.norm(np.cross(x, y), axis=-1), np.sum(x*y, axis=-1))


def inspect_heldout(saved, native, checks):
    checks.array(saved['node_ids'], native['node_ids'], 'heldout native node order')
    checks.array(saved['query_seconds'], native['query_seconds'], 'heldout query seconds')
    n = len(native['node_ids'])
    offsets = saved['heldout_offsets']
    raw_offsets = native['event_offsets']
    checks.exact(offsets.shape == (n+1,) and offsets[0] == 0 and np.all(np.diff(offsets) >= 0),
                 'heldout complete offsets')
    size = int(offsets[-1])
    checks.exact(saved['mse_by_rho'].shape == (n, 9) and saved['eligible'].shape == (n,),
                 'heldout error dimensions')
    for key in ('rts_positions', 'forward_positions'):
        checks.exact(saved[key].shape == (9, size, 2) and np.isfinite(saved[key]).all(),
                     'heldout complete finite '+key)
    errors = np.zeros((n, 9))
    eligible = np.zeros(n, bool)
    retained_total = heldout_total = 0
    for node in range(n):
        raw = slice(int(raw_offsets[node]), int(raw_offsets[node+1]))
        times = native['event_times_seconds'][raw]
        values = native['event_values'][raw, :2]
        length = len(times)
        index = np.arange(length)
        held = index[(index > 0) & (index < length-1) & (index % 3 == 1)]
        part = slice(int(offsets[node]), int(offsets[node+1]))
        checks.exact(length >= 2 and np.all(np.diff(times) > 0)
                     and native['event_valid'][raw, :2].all(), 'complete valid native sequence')
        checks.array(saved['heldout_indices'][part], held, 'predeclared interior heldout pattern')
        checks.array(saved['heldout_times'][part], times[held], 'heldout actual native seconds')
        checks.array(saved['heldout_latlon'][part], values[held], 'heldout actual observed positions')
        checks.exact(int(saved['retained_count'][node]) == length-len(held), 'retained count')
        eligible[node] = len(held) > 0
        if eligible[node]:
            error = distances(saved['rts_positions'][:, part], values[held][None])
            errors[node] = np.mean(error**2, axis=1)
            shared.close(saved['mse_by_rho'][node], errors[node], checks,
                         'independent node heldout MSE m2', atol=1e-3)
        else:
            checks.exact(np.isfinite(saved['mse_by_rho'][node]).all(), 'finite unscored placeholder')
        retained_total += length-len(held)
        heldout_total += len(held)
    checks.array(saved['eligible'], eligible, 'eligible mask')
    return (np.mean(errors[eligible], axis=0) if eligible.any() else None), dict(
        fragments=n, eligible_fragments=int(eligible.sum()), heldout_events=heldout_total,
        retained_events=retained_total, scored_node_rho_calls=9*int(eligible.sum()))


def source_equal_positions(motion, positions, membership, checks):
    """Independently split equal source mass over supported raw nodes."""
    support, sources = motion['support'], motion['sources']
    n, q = support.shape
    checks.exact(positions.shape == (n, q, 2), 'complete native state positions')
    group_ids = np.unique(membership)
    checks.array(group_ids, np.arange(len(group_ids)), 'consecutive memberships')
    output, weights = np.empty((len(group_ids), q, 2)), np.zeros((n, q))
    for gid in group_ids:
        members = np.flatnonzero(membership == gid)
        by_source = {}
        for node in members:
            by_source.setdefault(str(sources[node]), []).append(int(node))
        available = {source: support[nodes].any(0) for source, nodes in by_source.items()}
        count = np.sum(list(available.values()), axis=0)
        for source, nodes in by_source.items():
            local_count = support[nodes].sum(0)
            weights[nodes] = np.divide(support[nodes], count*np.maximum(local_count, 1),
                out=np.zeros((len(nodes), q)), where=count > 0)
        # Fixed fallback rule uses this model's first-node state, including
        # outside-support CV endpoint extrapolation; speed/heading are checked
        # separately against original Mean in the final artifact.
        out = positions[members[0]].copy()
        mass = weights[members]
        active = count > 0
        out[active, 0] = np.sum(mass*positions[members, :, 0], axis=0)[active]
        angles = np.deg2rad(positions[members, :, 1])
        sine, cosine = np.sum(mass*np.sin(angles), axis=0), np.sum(mass*np.cos(angles), axis=0)
        use = active & (np.hypot(sine, cosine) > 1e-12)
        out[use, 1] = ((np.rad2deg(np.arctan2(sine, cosine))+180.) % 360.-180.)[use]
        single = (mass > 0).sum(0) == 1
        selected = np.argmax(mass, axis=0)
        out[single] = positions[members[selected[single]], np.flatnonzero(single)]
        output[gid] = out
    return output, weights


def source_masses(motion, membership, saved, checks):
    mapping = np.empty(len(membership), np.int64)
    counts, offsets = [], [0]
    for gid in np.unique(membership):
        by_source = {}
        for node in np.flatnonzero(membership == gid):
            by_source.setdefault(str(motion['sources'][node]), []).append(int(node))
        for nodes in by_source.values():
            mapping[nodes] = len(counts)
            counts.append(motion['support'][nodes].sum(0))
        offsets.append(len(counts))
    counts = np.asarray(counts)
    available = counts > 0
    checks.array(saved['node_to_source_row'], mapping, 'physical source mapping')
    checks.array(saved['supported_counts'], counts, 'physical source support counts')
    checks.array(saved['available'], available, 'physical available sources')
    checks.array(saved['group_offsets'], np.asarray(offsets), 'source group offsets')
    weights = np.zeros(counts.shape)
    for a, b in zip(offsets[:-1], offsets[1:]):
        total = available[a:b].sum(0)
        weights[a:b] = np.divide(available[a:b], total, out=np.zeros_like(weights[a:b]), where=total > 0)
    return weights


def inspect_run(run_path, native_path, risk_path, manifest_path, checks):
    run_path, native_path, risk_path = map(Path, (run_path, native_path, risk_path))
    run, summary = (core.read(run_path/name) for name in ('run.json', 'summary.json'))
    manifest = core.read(manifest_path)
    old_risk = core.read(risk_path/'run.json')
    native_run = core.read(native_path/'run.json')
    checks.exact(run.get('status') == summary.get('status') == 'COMPLETED'
                 and run.get('exit_code') == 0, 'science completed exit0')
    checks.exact(run['idea_id'] == summary['idea_id'] == IDEA
                 and run['plan_version'] == summary['plan_version'] == '1.0', 'fixed Idea and plan')
    checks.exact(old_risk['status'] == native_run['status'] == 'COMPLETED'
                 and old_risk['exit_code'] == native_run['exit_code'] == 0, 'completed frozen inputs')
    tids, vids = old_risk['training_scene_ids'], old_risk['scene_ids']
    checks.exact(tids == manifest['splits']['T_core'][:32]
                 and len(vids) == 32 and set(vids).issubset(manifest['splits']['V_select'])
                 and not set(tids) & set(vids), 'original fixed development population')
    checks.exact(run['training_scene_ids'] == tids and run['scene_ids'] == vids
                 and tuple(run['T_methods']) == T_METHODS and tuple(run['methods']) == V_METHODS,
                 'declared T5 V18 arms and scene order')
    for key, expected in (('native_run', native_path), ('source_risk_run', risk_path)):
        checks.exact(Path(run[key]).resolve() == expected.resolve(), 'frozen '+key+' path')
    for key in ('teacher_forward_calls', 'neural_models_constructed', 'optimizer_updates',
                'target_reference_fit_count', 'analytic_head_fits'):
        checks.value(0, run[key], key, 'no new neural or target-reference fit')
    checks.exact(run['test_opened'] is False and run['archive_read'] is False
                 and run['selection_frozen_before_reference_control'] is True,
                 'information access declarations')
    selection = core.read(run_path/'scale-selection.json')
    checks.exact(selection['idea_id'] == IDEA and selection['training_scene_ids'] == tids
                 and selection['reference_used'] is False, 'observation selection declaration')
    shared.close(selection['rhos'], RHOS, checks, 'fixed nine rhos')
    scene_objectives, counts = [], []
    for sid, public in zip(tids, selection['per_scene']):
        checks.exact(public['scene_id'] == sid, 'selection scene order')
        native = load(native_path/'private/native-events/T_core'/f'scene-{sid}.npz')
        saved = load(run_path/'private'/f'heldout-T-{sid}.npz')
        objective, count = inspect_heldout(saved, native, checks)
        counts.append(count)
        checks.value(count['fragments'], public['fragment_count'], 'fragment_count', 'selection population')
        checks.value(count['eligible_fragments'], public['eligible_fragment_count'],
                     'eligible_fragment_count', 'selection population')
        if objective is None:
            checks.exact(public['objective_by_rho'] is None, 'unscored scene remains null')
        else:
            shared.close(public['objective_by_rho'], objective, checks, 'scene-equal node criterion', atol=1e-3)
            scene_objectives.append(objective)
    checks.exact(len(selection['per_scene']) == len(tids) and len(scene_objectives) > 0,
                 'all selection scenes present')
    objective = np.mean(scene_objectives, axis=0)
    shared.close(selection['objective_by_rho'], objective, checks, 'scene-equal global criterion', atol=1e-3)
    # The exact minimizer is checked on the originally saved per-node MSE too;
    # independent spherical geometry may differ by harmless roundoff at a tie.
    exact_objectives = np.mean([np.asarray(row['objective_by_rho']) for row in selection['per_scene']
                               if row['objective_by_rho'] is not None], axis=0)
    chosen = int(np.argmin(exact_objectives))
    checks.exact(int(selection['selected_index']) == chosen
                 and float(selection['selected_rho']) == RHOS[chosen]
                 and selection['tie_count'] == int((exact_objectives == exact_objectives[chosen]).sum()),
                 'minimum criterion and exact lower-rho tie policy')
    observation_rho = RHOS[chosen]
    reference_rho = float(core.read(native_path/'scale-selection.json')['selected_rho'])
    unique = list(dict.fromkeys((reference_rho, observation_rho)))
    checks.exact(reference_rho in RHOS, 'original reference control rho')
    checks.exact(run['unique_recovery_rhos'] == unique
                 and run['observation_selected_rho'] == observation_rho
                 and run['reference_selected_rho'] == reference_rho
                 and run['recovery_alias'] == (observation_rho == reference_rho), 'unique recovery scales')
    checks.exact(summary['unique_recovery_rhos'] == unique
                 and summary['observation_selected_rho'] == observation_rho
                 and summary['reference_selected_rho'] == reference_rho
                 and summary['recovery_alias'] == (observation_rho == reference_rho), 'public scale binding')
    checks.exact(tuple(summary['primary_comparisons']) == PRIMARY, 'complete public primary estimands')
    fit_selection = sum(c['scored_node_rho_calls'] for c in counts)
    checks.value(fit_selection, run['selection_filter_smoother_fits'], 'selection_filter_smoother_fits', 'real declared fit count')
    checks.value(sum(c['heldout_events'] for c in counts), run['selection_heldout_event_count'],
                 'selection_heldout_event_count', 'selection population')
    checks.value(sum(c['retained_events'] for c in counts), run['selection_retained_event_count'],
                 'selection_retained_event_count', 'selection population')
    blocks, recovery_nodes = [], 0
    for split, ids, names, row_file, block in (
        ('T_core', tids, T_METHODS, 'T-scene-summary.json.gz', summary['training_summary']),
        ('V_select', vids, V_METHODS, 'scene-summary.json.gz', summary)):
        rows = core.read(run_path/row_file)
        for row in rows:
            checks.exact(row['component_id'] == str(manifest['components'][row['scene_id']]),
                         split+' manifest component')
        _, metric_scope = shared.inspect_metric_block(run_path, split, rows, block, names, checks,
            expected_scene_ids=ids, expected_component_count=12 if split == 'T_core' else 9,
            expected_primary_pairs=PRIMARY, native_evaluation_root=native_path/'private/native-evaluation')
        for sid in ids:
            motion = load(native_path/'private/native-motion'/split/f'scene-{sid}.npz')
            saved = load(run_path/'private'/f'native-state-{split}-{sid}.npz')
            artifacts = core.artifacts(run_path/'private'/f'states-{split}-{sid}.npz', names, template=names[0])
            for key in ('node_ids', 'query_seconds', 'support'):
                checks.array(saved[key], motion[key], split+' native-state '+key)
            shared.close(saved['unique_rhos'], unique, checks, 'saved unique scales')
            checks.exact(int(saved['reference_rho_index']) == unique.index(reference_rho)
                         and int(saved['obs_rho_index']) == unique.index(observation_rho), 'saved scale indices')
            membership = saved['membership']
            checks.array(membership, artifacts[names[0]]['node_group_indices'], 'private/artifact membership')
            equal_source_masses = source_masses(motion, membership, saved, checks)
            expected_base, equal_weights = source_equal_positions(motion, motion['states'][..., :2], membership, checks)
            shared.close(artifacts['WS_SOURCE_EQUAL']['group_states'][..., :2], expected_base, checks,
                         'original source equal raw-node positions', atol=1e-9)
            for label, rho in (('REFERENCE', reference_rho), ('OBS', observation_rho)):
                for direction, suffix in (('forward', 'FORWARD'), ('rts', 'RTS')):
                    name = f'WS_{label}_SCALE_{suffix}_EQUAL'
                    pos = saved[f'rho_{unique.index(rho)}__{direction}_positions']
                    expected, node_weights = source_equal_positions(motion, pos, membership, checks)
                    shared.close(saved[name+'__node_weights'], node_weights, checks, name+' source-equal node masses')
                    shared.close(saved[name+'__source_weights'], equal_source_masses, checks,
                                 name+' independently equal source masses')
                    checks.array(saved[name+'__group_states'], artifacts[name]['group_states'],
                                 name+' private/artifact states binding')
                    shared.close(artifacts[name]['group_states'][..., :2], expected, checks,
                                 name+' independent physical readout', atol=1e-9)
                    checks.array(artifacts[name]['group_states'][..., 2:],
                                 artifacts['WS_SOURCE_EQUAL']['group_states'][..., 2:], name+' original speed heading')
                    shared.close(node_weights, equal_weights, checks, name+' contribution unchanged')
            if split == 'V_select':
                old_arrays = core.artifacts(risk_path/'private'/f'states-{sid}.npz', IMPORTED)
                historical = core.artifacts(native_path/'private'/f'states-{sid}.npz', HISTORICAL)
                for name, old in {**old_arrays, **historical}.items():
                    checks.exact(set(artifacts[name]) == set(old), 'old artifact fields')
                    for key in old:
                        checks.array(artifacts[name][key], old[key], 'exact imported '+name+'/'+key)
            recovery_nodes += len(motion['node_ids'])
        blocks.append(metric_scope)
    expected_recovery_fits = recovery_nodes*len(unique)
    checks.value(expected_recovery_fits, run['recovery_filter_smoother_fits'],
                 'recovery_filter_smoother_fits', 'deduplicated recovery cost')
    checks.value(fit_selection+expected_recovery_fits, run['analytic_filter_smoother_fits'],
                 'analytic_filter_smoother_fits', 'total analytic calls')
    for key in ('selection_filter_smoother_fits', 'recovery_filter_smoother_fits', 'analytic_filter_smoother_fits'):
        checks.value(run[key], summary[key], key, 'public fit count binding')
    checks.exact(run['completed_selection_scene_count'] == run['completed_training_scene_count']
                 == run['completed_scene_count'] == 32, 'fixed endpoint')
    return dict(metric_blocks=blocks, independent_objective_by_rho=objective.tolist(),
        selected_rho=observation_rho, reference_rho=reference_rho,
        selection_fit_count=fit_selection, recovery_fit_count=expected_recovery_fits,
        selection_population=counts, recovery_alias=len(unique) == 1,
        exclusions=['Does not re-execute the CV/RTS kernel or certify physical reference/noise correctness',
                    'Chronological isolation additionally rests on code review and focused deletion tests',
                    'No independent speed/heading error or paper-coordinate metric certification',
                    'No real K>=3, cross-domain, online or peer superiority certification'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'native-run', 'source-risk-run', 'manifest', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('new independent audit output required')
    began = time.monotonic()
    checks = core.Checks()
    result = dict(status='RUNNING', idea_id=IDEA, started_at=datetime.now(timezone.utc).isoformat())
    exit_code = 0
    try:
        scope = inspect_run(args.run, args.native_run, args.source_risk_run, args.manifest, checks)
        result.update(status='COMPLETED', exit_code=0, scope=scope)
    except BaseException as error:
        exit_code = 1
        result.update(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
    result.update(elapsed_seconds=time.monotonic()-began,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        exact_checks=checks.exact_checks, numeric_checks=checks.numeric_checks,
        null_checks=checks.null_checks, maximum_absolute_difference=checks.maximum_absolute_difference,
        finished_at=datetime.now(timezone.utc).isoformat())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('status', 'exit_code', 'elapsed_seconds')}))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
