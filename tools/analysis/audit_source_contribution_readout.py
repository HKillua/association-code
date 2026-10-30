"""Independent 140 audit: chord responses, thirteen ridge fits and three readouts.

All imported project computations are earlier independent NumPy auditors.
The numerical surrogate certificate does not certify physical conditional risk.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[setting] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from tools.analysis import audit_source_selection_evidence as prior

common, original, physical, core = prior.common, prior.original, prior.physical, prior.core
load, close, tree_equal = prior.load, prior.close, prior.tree_equal
IDEA = 'IDEA-SOURCE-CONTRIBUTION-READOUT-20261030'
NEW = ('WS_CHORD_RISK_HARD', 'WS_CHORD_RISK_INVERSE', 'WS_CHORD_RISK_GEOMETRY')
OLD_T = ('WS_SOURCE_EQUAL', 'WS_SHARED_Q_SOFT', 'WS_SHARED_Q_HARD')
T_METHODS, V_METHODS = OLD_T+NEW, original.METHODS+NEW
PRIMARY = (NEW[0]+'_vs_WS_SHARED_Q_HARD', NEW[1]+'_vs_'+NEW[0],
           NEW[2]+'_vs_'+NEW[0], NEW[2]+'_vs_'+NEW[1])
MODEL_FIELDS = ('head', 'target_kind', 'mean', 'std', 'beta', 'training_label_count',
                'training_weight_sum', 'no_labels_fallback', 'ridge_lambda')
RADIUS_M, FLOOR = 6371000., 1e-6
RAM, MAX_AS = 6*1024**3, 12*1024**3


def chord(errors):
    errors = np.asarray(errors, float)
    if not np.isfinite(errors).all() or np.any((errors < 0) | (errors > np.pi*RADIUS_M)):
        raise ValueError('invalid original radial labels')
    return np.square(2.*(RADIUS_M/1000.)*np.sin(.5*errors/RADIUS_M))


def fit(training):
    x = np.concatenate([r['features'][r['labeled']] for r in training]) if training else np.empty((0, 10))
    y = np.concatenate([chord(r['errors_m'][r['labeled']]) for r in training]) if training else np.empty(0)
    weight = np.concatenate([r['sample_weight'][r['labeled']] for r in training]) if training else np.empty(0)
    total = float(weight.sum()); norm = weight/total if len(y) else weight
    mean = (norm[:, None]*x).sum(0) if len(y) else np.zeros(10)
    std = np.sqrt((norm[:, None]*(x-mean)**2).sum(0)) if len(y) else np.ones(10)
    std[std < 1e-6] = 1.
    design = original.design(x, mean, std, 'QUADRATIC')
    beta = (np.linalg.solve((design*norm[:, None]).T@design+np.diag(np.r_[0., np.ones(65)]),
                           (design*norm[:, None]).T@y) if len(y) else np.zeros(66))
    return dict(head='QUADRATIC', target_kind='squared_chord_km2', mean=mean, std=std, beta=beta,
        training_label_count=len(y), training_weight_sum=total, no_labels_fallback=not len(y), ridge_lambda=1.)


def inspect_fit(payload, fold, training, old_payload, checks):
    model = fit(training)
    for key in MODEL_FIELDS:
        stored = payload[fold+'__'+key]
        if key in ('head', 'target_kind'):
            checks.exact(str(stored) == model[key], fold+' '+key)
        elif key in ('mean', 'std', 'beta'):
            close(stored, model[key], checks, fold+' independent '+key, 1e-8)
        else:
            checks.value(stored.item(), model[key], key, fold)
    for key in ('mean', 'std', 'training_label_count', 'training_weight_sum', 'no_labels_fallback'):
        close(payload[fold+'__'+key], old_payload[fold+'__QUADRATIC__'+key], checks,
              fold+' unchanged original133 '+key, 1e-8)
    return model


def inspect_training(run_path, paths, manifest, ids, checks):
    rows = []
    for sid in ids:
        row = load(run_path/'private'/f'training-{sid}.npz')
        old = load(paths['source_risk_run']/'private'/f'training-{sid}.npz')
        checks.exact(set(row) == (set(old)-{'y'}) | {'y_chord_km2'}, 'exact new response schema')
        for key in set(old)-{'y'}:
            checks.array(row[key], old[key], 'unchanged133 training '+key)
        motion = load(paths['native_run']/'private/native-motion/T_core'/f'scene-{sid}.npz')
        ev = load(paths['native_run']/'private/native-evaluation/T_core'/f'scene-{sid}.npz')
        member = load(paths['value_run']/'private'/f'T-features-{sid}.npz')['membership']
        checks.array(row['membership'], member, 'original125 T partition')
        for key in ('node_ids', 'query_seconds'):
            checks.array(ev[key], motion[key], 'T original evaluation '+key)
        groups = original.inspect_pack(row, motion, member, checks, '140 T source pack')
        label_audit = original.inspect_labels(dict(row, y=old['y']), groups, ev, checks)
        checks.exact(np.isnan(row['y_chord_km2'][~row['labeled']]).all(), 'unlabeled chord outcome unavailable')
        close(row['y_chord_km2'][row['labeled']], chord(row['errors_m'][row['labeled']]), checks,
              'exact radial to chord label', 1e-10)
        component = str(manifest['components'][sid])
        checks.exact(str(row['component_id']) == component, 'original T component')
        rows.append(dict(row, scene_id=sid, component_id=component, audit=label_audit))
    components = sorted({row['component_id'] for row in rows})
    checks.exact(len(rows) == 32 and len(components) == 12, 'T32 twelve components')
    folds = ['leave_'+c for c in components]+['full']
    payload = load(run_path/'private/models.npz'); before = load(paths['source_risk_run']/'private/models.npz')
    checks.exact(set(payload) == {f+'__'+k for f in folds for k in MODEL_FIELDS}, 'exact thirteen model schema')
    predictions, full = {}, None
    for fold in folds:
        heldout = None if fold == 'full' else fold[len('leave_'):]
        selected = [r for r in rows if r['component_id'] != heldout]
        model = inspect_fit(payload, fold, selected, before, checks)
        if heldout is None:
            full = model
        else:
            for row in (r for r in rows if r['component_id'] == heldout):
                sid = row['scene_id']; saved = load(run_path/'private'/f'oof-{sid}.npz')
                score = original.predict(model, row['features'], 'QUADRATIC')
                checks.exact(set(saved) == {'risk_raw'}, 'only declared OOF risk')
                close(saved['risk_raw'], score, checks, 'independent OOF risk scores', 1e-8)
                predictions[sid] = score
    entries = core.read(run_path/'T-fold-index.json')
    checks.exact([r['fold'] for r in entries] == folds, 'complete ordered thirteen folds')
    for entry in entries:
        heldout = entry['heldout_component']; selected = [r for r in rows if r['component_id'] != heldout]
        expected = dict(fold='full' if heldout is None else 'leave_'+heldout, heldout_component=heldout,
            training_components=sorted({r['component_id'] for r in selected}), training_scene_ids=[r['scene_id'] for r in selected],
            training_label_count=sum(int(r['labeled'].sum()) for r in selected),
            training_weight_sum=float(sum(r['sample_weight'].sum() for r in selected)))
        for key in ('fold', 'heldout_component', 'training_components', 'training_scene_ids'):
            checks.exact(entry[key] == expected[key], 'fold exclusion '+key)
        for key in ('training_label_count', 'training_weight_sum'):
            checks.value(entry[key], expected[key], key, 'fold accounting')
    return rows, predictions, full


def policies(risk, available, offsets, states):
    """Independent analytic K<=2 solution; real K>=3 is outside this audit scope."""
    available, risk = np.asarray(available), np.asarray(risk, float)
    shape = (len(offsets)-1, risk.shape[1])
    result = {name: dict(source_weights=np.zeros_like(risk), floor_hit=np.zeros_like(available),
        objective_km2=np.full(shape, np.nan), duality_gap_km2=np.full(shape, np.nan),
        solver_iterations=np.zeros(shape, np.int64), solver_converged=np.zeros(shape, bool),
        solver_defined=np.zeros(shape, bool)) for name in NEW}
    angles = np.deg2rad(states[..., :2])
    unit = np.stack((np.cos(angles[..., 0])*np.cos(angles[..., 1]),
                     np.cos(angles[..., 0])*np.sin(angles[..., 1]), np.sin(angles[..., 0])), axis=-1)
    xyz = unit*(RADIUS_M/1000.)
    for gid, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        for qi in range(risk.shape[1]):
            rows = np.flatnonzero(available[a:b, qi])+a
            if len(rows) > 2:
                raise ValueError('independent140 real audit supports K<=2 only; no silent general-K certification')
            if not len(rows):
                continue
            r = risk[rows, qi]
            d = np.square(xyz[rows, qi, None, :]-xyz[None, rows, qi, :]).sum(-1)
            hard = (r == r.min()).astype(float); hard /= hard.sum()
            inv = 1./np.maximum(r, FLOOR); inv /= inv.sum()
            if len(rows) == 1 or d[0, 1] == 0:
                geom = hard.copy()
            else:
                alpha = np.clip((d[0, 1]+r[1]-r[0])/(2.*d[0, 1]), 0., 1.)
                geom = np.array([alpha, 1.-alpha])
            for name, weights in zip(NEW, (hard, inv, geom)):
                p = result[name]; p['source_weights'][rows, qi] = weights
                gradient = r-d@weights
                p['objective_km2'][gid, qi] = np.dot(r, weights)-.5*np.dot(weights, d@weights)
                p['duality_gap_km2'][gid, qi] = max(0., float(weights@gradient-gradient.min()))
                p['solver_defined'][gid, qi] = True
                if name == NEW[1]:
                    p['floor_hit'][rows, qi] = r < FLOOR
                if name == NEW[2]:
                    p['solver_converged'][gid, qi] = True
    return result


def stat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return dict(count=len(x), mean=float(x.mean()) if len(x) else None,
                max=float(x.max()) if len(x) else None, min=float(x.min()) if len(x) else None)


def contributions(pack, weights):
    maxima, entropy, l1, changes, slots = [], [], [], [], []
    for a, b in zip(pack['group_offsets'][:-1], pack['group_offsets'][1:]):
        active, w = pack['available'][a:b], weights[a:b]
        live = active.any(0)
        for qi in np.flatnonzero(live):
            ws = w[:, qi]; equal = active[:, qi]/active[:, qi].sum()
            maxima.append(float(ws.max())); entropy.append(float(-np.sum(ws*np.log(np.maximum(ws, 1e-300)))))
            l1.append(float(np.abs(ws-equal).sum()))
        if live.any():
            slots.extend(w[:, live].mean(1).tolist())
        changes.extend(np.abs(np.diff(w, axis=1)).sum(0)[live[:-1] & live[1:]].tolist())
    return dict(supported_group_queries=len(maxima), mean_max_source_mass=float(np.mean(maxima)) if maxima else None,
        mean_entropy=float(np.mean(entropy)) if entropy else None, mean_l1_from_source_equal=float(np.mean(l1)) if l1 else None,
        adjacent_supported_query_pairs=len(changes), mean_adjacent_l1_change=float(np.mean(changes)) if changes else None,
        source_group_slot_mean_mass_quantiles=np.quantile(slots, [0,.25,.5,.75,1]).tolist() if slots else [])


def policy_report(pack, raw, computed):
    mask = pack['available']; values = raw[mask]
    result = dict(available_source_queries=int(mask.sum()), raw_negative_count=int((values < 0).sum()),
        raw_negative_values_km2=stat(values[values < 0]), clipped_zero_count=int((values <= 0).sum()), methods={})
    for name, p in computed.items():
        defined = p['solver_defined']; objective = p['objective_km2'][defined]
        result['methods'][name] = dict(floor_hit_count=int(p['floor_hit'].sum()),
            floor_denominator_source_queries=int(mask.sum()), defined_group_queries=int(defined.sum()),
            negative_objective_count=int((objective < 0).sum()), objective_km2=stat(objective),
            duality_gap_km2=stat(p['duality_gap_km2'][defined]), solver_iterations=stat(p['solver_iterations'][defined]),
            solver_not_converged_count=int((defined & ~p['solver_converged']).sum()) if name == NEW[2] else None,
            contributions=contributions(pack, p['source_weights']))
    return result


def endpoint_report(pack, labels, score):
    weight = original.policy_weights(score, pack['available'], pack['group_offsets'], 'hard')
    cost, regret, credit = [], [], []
    for a, b in zip(pack['group_offsets'][:-1], pack['group_offsets'][1:]):
        for qi in np.flatnonzero(labels['labeled'][a:b].any(0)):
            use = labels['labeled'][a:b, qi]
            error = np.where(use, labels['errors_m'][a:b, qi], np.inf)
            w = weight[a:b, qi]; value = float(w@np.where(use, error, 0.))
            cost.append(value); regret.append(value-float(error.min())); credit.append(float(w@(error == error.min())))
    regret = np.asarray(regret)
    return dict(query_count=len(cost), expected_endpoint_error_m=stat(cost), selection_regret_m=stat(regret),
        selection_credit=stat(credit), regret_gt_1000m_count=int((regret > 1000).sum()),
        regret_gt_1000m_sum=float(regret[regret > 1000].sum()))


def inspect_policy(run_path, paths, split, row, predictions, full, checks):
    sid = row['scene_id']; names = T_METHODS if split == 'T_core' else V_METHODS
    pack = load(run_path/'private'/f'policy-{split}-{sid}.npz')
    saved = core.artifacts(run_path/'private'/f'states-{split}-{sid}.npz', names, template=names[0])
    motion = load(paths['native_run']/'private/native-motion'/split/f'scene-{sid}.npz')
    membership = (load(paths['value_run']/'private'/f'T-features-{sid}.npz')['membership'] if split == 'T_core'
        else np.asarray(core.read(paths['partition_run']/'private'/f'scene-{sid}.json.gz')['memberships']['WS']))
    checks.array(pack['membership'], membership, 'original grouping')
    groups = original.inspect_pack(pack, motion, membership, checks, split+' source pack')
    old_pack = load(paths['source_risk_run']/'private'/
                    (f'training-{sid}.npz' if split == 'T_core' else f'policy-{sid}.npz'))
    for key in ('node_ids', 'query_seconds', 'membership', 'group_offsets', 'node_to_source_row', 'source_group_indices',
                'supported_counts', 'source_member_count', 'available', 'source_states', 'features'):
        checks.array(pack[key], old_pack[key], 'immutable original133 '+key)
    old_q = (load(paths['source_risk_run']/'private'/f'oof-{sid}.npz')['QUADRATIC__scores'] if split == 'T_core'
        else old_pack['WS_SHARED_Q_HARD__scores'])
    checks.array(pack['old_Q_scores'], old_q, 'original Q score exact')
    raw = predictions[sid] if split == 'T_core' else original.predict(full, pack['features'], 'QUADRATIC')
    close(pack['risk_raw'], raw, checks, 'complete raw risk scores', 1e-8)
    clipped = np.maximum(pack['risk_raw'], 0.)
    checks.array(pack['risk_clipped'], clipped, 'one common exact nonnegative clip')
    # After score verification, use saved floating scores for exact tie/sign counts.
    # Small independent normal-equation rounding must not redefine a saved tie.
    computed = policies(clipped, pack['available'], pack['group_offsets'], pack['source_states'])
    mean, _ = physical.mean_and_recovery(motion, {'groups': groups})
    offset, available, mapping, counts = (pack[k] for k in
        ('group_offsets', 'available', 'node_to_source_row', 'supported_counts'))
    ng = np.stack([available[a:b].sum(0) for a, b in zip(offset[:-1], offset[1:])])
    checks.exact(row['source_support_scope'] == dict(zero=int((ng == 0).sum()), one=int((ng == 1).sum()),
        at_least_two=int((ng >= 2).sum())) and row['node_count'] == len(motion['node_ids']), 'complete source scope')
    for name, p in computed.items():
        for key, value in p.items():
            close(pack[name+'__'+key], value, checks, name+'/'+key, 1e-9)
        sw = p['source_weights']; nw = sw[mapping]*motion['support']/np.maximum(counts[mapping], 1)
        close(pack[name+'__node_weights'], nw, checks, name+' node weights', 1e-12)
        restored, _ = physical.mean_and_recovery(motion, {'groups': groups}, nw)
        for gid, members in enumerate(groups):
            absent = ~motion['support'][members].any(0)
            restored[gid, absent] = mean[gid, absent]
        restored[..., 2:] = mean[..., 2:]
        close(pack[name+'__group_states'], restored, checks, name+' node physical output', 1e-8)
        close(saved[name]['group_states'], restored, checks, name+' saved physical output', 1e-8)
        checks.array(saved[name]['group_states'][..., 2:], saved['WS_SOURCE_EQUAL']['group_states'][..., 2:], 'original speed heading exact')
    if split == 'V_select':
        prior.bind_original_artifacts(saved, paths['source_risk_run']/'private'/f'states-{sid}.npz', original.METHODS, checks, split)
    else:
        for name, kind in zip(OLD_T, ('equal', 'soft', 'hard')):
            sw = original.policy_weights(old_q, available, offset, kind)
            nw = sw[mapping]*motion['support']/np.maximum(counts[mapping], 1)
            restored, _ = physical.mean_and_recovery(motion, {'groups': groups}, nw)
            for gid, members in enumerate(groups):
                absent = ~motion['support'][members].any(0); restored[gid, absent] = mean[gid, absent]
            restored[..., 2:] = mean[..., 2:]
            for key, value in dict(source_weights=sw, node_weights=nw, group_states=restored).items():
                close(pack[name+'__'+key], value, checks, 'T old score new recovery '+name+'/'+key, 1e-8)
            close(saved[name]['group_states'], restored, checks, 'T old score artifact '+name, 1e-8)
    tree_equal(policy_report(pack, pack['risk_raw'], computed), row['risk_policy'], checks, 'policy diagnostic')
    ev = load(paths['native_run']/'private/native-evaluation'/split/f'scene-{sid}.npz')
    for key in ('node_ids', 'query_seconds'):
        checks.array(ev[key], motion[key], 'original evaluation '+key)
    labels = load(run_path/'private'/f'diagnostic-{split}-{sid}.npz')
    checks.exact(set(labels) == {'labeled', 'sample_weight', 'errors_m'}, 'diagnostic only labels')
    log = np.full(labels['errors_m'].shape, np.nan)
    log[labels['labeled']] = np.log1p(labels['errors_m'][labels['labeled']]/1000.)
    audit = original.inspect_labels(dict(pack, **labels, y=log), groups, ev, checks)
    if split == 'T_core':
        for key in labels:
            checks.array(labels[key], old_pack[key], 'original133 T label exact '+key)
    mask = labels['labeled']; weight = labels['sample_weight'][mask]
    target = chord(labels['errors_m'][mask]); rs = pack['risk_raw'][mask]
    raw_residual, clipped_residual = rs-target, np.maximum(rs, 0.)-target
    expected = dict(label_audit=audit, labeled_source_queries=int(mask.sum()), raw_negative_count=int((rs < 0).sum()),
        inverse_floor_hit_labeled_count=int((np.maximum(rs, 0.) < FLOOR).sum()),
        raw_squared_risk_residual_km4=stat(raw_residual**2), clipped_squared_risk_residual_km4=stat(clipped_residual**2),
        training_weight_sum=float(weight.sum()), weighted_raw_squared_residual_sum_km4=float(weight@(raw_residual**2)),
        weighted_clipped_squared_residual_sum_km4=float(weight@(clipped_residual**2)),
        old_Q_endpoint=endpoint_report(pack, labels, old_q), chord_risk_endpoint=endpoint_report(pack, labels, pack['risk_clipped']))
    tree_equal(expected, row['reference_diagnostic'], checks, 'reference diagnostic')


def audit_run(run_path, paths, manifest_path, checks):
    state, summary, manifest = (core.read(p) for p in (run_path/'run.json', run_path/'summary.json', manifest_path))
    checks.exact(state['status'] == summary['status'] == 'COMPLETED' and state['exit_code'] == 0, 'completed140')
    checks.exact(state['idea_id'] == summary['idea_id'] == IDEA and state['plan_version'] == summary['plan_version'] == '1.0', 'fixed140 contract')
    for key, path in paths.items():
        checks.exact(Path(state[key]).resolve() == path.resolve(), 'CLI lineage '+key)
    checks.exact(Path(state['manifest']).resolve() == manifest_path.resolve(), 'CLI original manifest')
    paths = dict(paths, value_run=Path(state['value_run']), partition_run=Path(state['partition_run']))
    for path in paths.values():
        s = core.read(path/'run.json')
        checks.exact(s['status'] == 'COMPLETED' and s['exit_code'] == 0, 'completed original input')
    source = core.read(paths['source_risk_run']/'run.json')
    for key in ('native_run', 'value_run', 'partition_run'):
        checks.exact(Path(source[key]).resolve() == paths[key].resolve(), 'original133 lineage '+key)
    tids, vids = state['training_scene_ids'], state['scene_ids']
    checks.exact(tids == source['training_scene_ids'] == manifest['splits']['T_core'][:32]
        and vids == source['scene_ids'] and set(vids).issubset(manifest['splits']['V_select'])
        and not set(tids) & set(vids), 'fixed original T/V populations')
    checks.exact(tuple(state['T_methods']) == T_METHODS and tuple(state['V_methods']) == V_METHODS
        and tuple(state['primary_pairs']) == PRIMARY, 'complete method and primary declarations')
    checks.exact(state['analytic_head_fits'] == summary['analytic_head_fits'] == 13
        and state['ridge_system_solves']+state['no_label_fallback_heads'] == 13, 'thirteen fits accounting')
    for key in ('teacher_forward_calls', 'neural_models_constructed', 'optimizer_updates', 'native_event_files_read', 'grid_oracle_calls'):
        checks.exact(state[key] == 0, 'declared zero '+key)
    for key in ('archive_read', 'csv_read', 'test_opened', 'gpu_used'):
        checks.exact(state[key] is False, 'declared false '+key)
    checks.exact(state['full_T_models_frozen_before_V'] is True and state['time_policy'] == 'advisory'
        and 1 <= state['cpu_threads'] <= 4, 'declared temporal and resource bounds')
    training, predictions, full = inspect_training(run_path, paths, manifest, tids, checks)
    scopes = {}
    for split, ids, names, nc, rows_path, sm in (
        ('T_core', tids, T_METHODS, 12, 'T-scene-summary.json.gz', core.read(run_path/'T-summary.json')),
        ('V_select', vids, V_METHODS, 9, 'scene-summary.json.gz', summary)):
        rows = core.read(run_path/rows_path)
        for row in rows:
            checks.exact(row['split'] == split and row['component_id'] == str(manifest['components'][row['scene_id']]), 'scene split/component')
            inspect_policy(run_path, paths, split, row, predictions, full, checks)
            checks.exact(row['original133_exact_replays_verified'] == (split == 'V_select')
                and row['fixed_population_invariants_verified'] and row['predictions_frozen_before_reference'], 'declared per-scene lineage')
            checks.exact(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024) <= RAM, 'bounded audit RSS')
        expected = [{k: r[k] for k in ('scene_id', 'component_id', 'risk_policy', 'reference_diagnostic')} for r in rows]
        checks.exact(sm['risk_diagnostics_by_scene'] == expected, 'complete pooled diagnostic scene binding')
        _, scopes[split] = common.inspect_metric_block(run_path, split, rows, sm, names, checks,
            expected_scene_ids=ids, expected_component_count=nc, expected_primary_pairs=PRIMARY,
            native_evaluation_root=paths['native_run']/'private/native-evaluation')
        if split == 'T_core':
            checks.exact(summary['training_summary'] == sm, 'final nested T summary exact')
    checks.exact(state['completed_training_cache_count'] == state['completed_training_scene_count'] == state['completed_scene_count'] == 32, 'complete fixed endpoint')
    return dict(metric_scope=scopes, audited_model_count=13,
        labeled_source_query_count=sum(int(r['labeled'].sum()) for r in training),
        T_components=12, V_components=9, T_old_score_recovered_controls=3, original_V_controls=10,
        actual_source_count_scope='K<=2; a K>=3 input is rejected, not silently certified')


def audit(run_path, native_path, source_path, manifest_path, output):
    started, before, checks = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF), core.Checks()
    run_path, manifest_path, output = map(lambda p: Path(p).resolve(), (run_path, manifest_path, output))
    paths = dict(native_run=Path(native_path).resolve(), source_risk_run=Path(source_path).resolve())
    if any(output == p or p in output.parents for p in (*paths.values(), manifest_path)):
        raise ValueError('audit output cannot alter original inputs')
    report = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
        scope='Independent140 source packs/labels,13 chord ridge fits/OOF/full,clip,K<=2 hard/inverse/geometry objective and gap,node recovery,old controls,main populations/metrics/bootstrap',
        not_audited=['physical reference correctness or true conditional moments/calibrated sensor health',
            'upstream teacher or original127 OLS correctness', 'general K>=3 solver beyond separate software tests',
            'runtime file-access order beyond code inspection and saved-array equality',
            'speed/heading metrics beyond original-output equality',
            'additional support-conditioned pair strata beyond common all/observed/missing metrics',
            'causal mechanism, independent deployment or credible peer superiority'],
        time_policy='advisory1800 one fixed endpoint no retries', gpu_used=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream)
    try:
        if sys.platform.startswith('linux'):
            if hasattr(os, 'sched_getaffinity'):
                os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([MAX_AS]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        report.update(audit_run(run_path, paths, manifest_path, checks), status='COMPLETED', exit_code=0)
    except BaseException as error:
        report.update(status='FAILED', exit_code=1, error_type=type(error).__name__,
            error=str(error) if isinstance(error, ValueError) else 'audit incomplete; preserve original inputs')
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(exact_checks=checks.exact_checks, numeric_checks=checks.numeric_checks, null_checks=checks.null_checks,
            maximum_absolute_difference=checks.maximum_absolute_difference, elapsed_seconds=time.monotonic()-started,
            advisory_exceeded=time.monotonic()-started > 1800,
            user_cpu_seconds=usage.ru_utime-before.ru_utime, system_cpu_seconds=usage.ru_stime-before.ru_stime,
            peak_rss_bytes=int(usage.ru_maxrss*(1 if sys.platform == 'darwin' else 1024)),
            finished_at=datetime.now(timezone.utc).isoformat())
        with output.open('w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'native-run', 'source-risk-run', 'manifest', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('audit signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    result = audit(args.run, args.native_run, args.source_risk_run, args.manifest, args.output)
    print(result['status']+': '+str(args.output))
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
