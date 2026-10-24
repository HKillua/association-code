"""085: existing PTFM output/log qualification; no checkpoint or forward access.

The all-output view deliberately repeats an anchor target's reference for its
duplicate groups. It is a diagnostic loss population, never a new system score.
"""
from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.ptfm_speed_diagnostics import SOURCE_METHODS, load_scene as replay_scene
from association.evaluation.support_gate_objectives import _artifact, _read
from association.evaluation.system_attribution import ATTRIBUTES, _same, assert_metrics, pool_metrics
from association.utils.runtime import read_json

ENDPOINTS = ('initial', 'final')
SCOPES = ('all_output_groups', 'unique_selected_groups')
TASKS = ('position', 'speed', 'heading')
MODES = ('clean', 'clean_plus_wrong', 'wrong_only')
EPS = 1e-8


def bind_source(source, config, manifest):
    """Metadata-only binding. Model files are named for provenance, never read."""
    source = Path(source).resolve()
    run, summary, fit = [read_json(source / p) for p in ('run.json', 'summary.json', 'fit/run.json')]
    norm = read_json(source / 'fit/normalizer.json')
    steps, batch = config['training']['updates'], config['training']['batch_size']
    ids = manifest['splits']['V_select']
    components = {sid: str(manifest['components'][sid]) for sid in ids}
    for item in (run, summary, fit):
        if (item.get('status') != 'COMPLETED' or item.get('configuration') != config
                or item.get('optimizer_updates') != steps or item.get('test_opened') is not False):
            raise ValueError('085 requires completed fixed source configuration/updates')
    if (run.get('completed_endpoints') != list(ENDPOINTS) or run.get('validation_selection') is not False
            or run.get('old066_acceptance') != 'REJECTED' or fit.get('final_checkpoint') != 'final.pt'
            or fit.get('training_scene_ids') != manifest['splits']['T_core']
            or fit.get('normalizer_sampler_replay_exact') is not True
            or norm != fit.get('normalizer') or norm != summary.get('normalizer')
            or norm.get('split') != 'T_core' or norm.get('truth_used') is not False
            or norm.get('missing_helper_rows_counted') is not False
            or type(norm.get('count')) is not int or norm['count'] < 1
            or norm.get('training_updates') != steps or norm.get('sampled_anchors') != steps * batch):
        raise ValueError('085 fixed endpoint/normalizer/training population binding differs')
    mean, scale = np.asarray(norm['mean']), np.asarray(norm['scale'])
    if (mean.shape != (4,) or scale.shape != (4,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('085 finite native-unit normalizer required')
    source056 = Path(summary['source_system']).resolve()
    if Path(run['source_system']).resolve() != source056:
        raise ValueError('085 source056 metadata differs')
    ends = {}
    for label, step, filename in (('initial', 0, 'step-0.pt'), ('final', steps, 'final.pt')):
        end, status = [read_json(source / label / p) for p in ('summary.json', 'run.json')]
        provenance = dict(path=str(source / 'fit' / filename), step=step, actual_updates=step,
            configured_final_updates=steps, routing_semantics=config['routing_semantics'])
        for record in (end, status):
            if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                    or record.get('initial') is not (label == 'initial') or record.get('device') != 'cpu'
                    or record.get('optimizer_updates') != 0 or record.get('test_opened') is not False
                    or record.get('groups_recomputed') is not False or record.get('scores_recomputed') is not False
                    or record.get('threshold_reselected') is not False or record.get('scene_ids') != ids
                    or record.get('checkpoint') != provenance or Path(record['source_system']).resolve() != source056):
                raise ValueError('085 frozen CPU endpoint metadata differs/' + label)
        if (end.get('scene_count') != len(ids) or status.get('completed_scenes') != len(ids)
                or status.get('total_scenes') != len(ids) or end.get('method_names') != list(SOURCE_METHODS)
                or summary.get('endpoints', {}).get(label) != provenance):
            raise ValueError('085 complete endpoint/provenance population differs')
        actual = {str(sid): str(c) for c, group in end['components'].items() for sid in group['scene_ids']}
        if actual != components or end.get('component_count') != len(set(components.values())):
            raise ValueError('085 component registry differs')
        inventory = {f'states-{sid}.npz' for sid in ids}
        for folder in ('predictions', 'system-predictions'):
            if {p.name for p in (source / label / folder).glob('*.npz')} != inventory:
                raise ValueError('085 full V NPZ inventory differs')
        with gzip.open(source / label / 'scene-summary.json.gz', 'rt', encoding='utf-8') as handle:
            rows = json.load(handle)
        if [r['scene_id'] for r in rows] != ids:
            raise ValueError('085 original scene summary population/order differs')
        ends[label] = dict(summary=end, rows=rows)
    return dict(path=source, run=run, summary=summary, fit=fit, normalizer=norm,
        scene_ids=ids, components=components, endpoints=ends)


def group_objectives(states, truth, valid, weights, scale):
    """Exact equations of models.ptfm.objectives, per group with task masks.

    No-supervision groups contribute zero total, as during training. Per-task
    undefined quantities are NaN internally and excluded from reported means.
    """
    states, truth, weights, scale = [np.asarray(x, np.float64) for x in (states, truth, weights, scale)]
    valid = np.asarray(valid)
    if (states.ndim != 3 or states.shape[-1] != 4 or truth.shape != states.shape
            or valid.shape != states.shape or valid.dtype != bool
            or weights.shape != states.shape[:2] + (3,) or scale.shape != (4,)
            or not np.isfinite(states).all() or not np.isfinite(truth[valid]).all()
            or not np.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any()
            or not np.allclose(weights.sum(-1), 1., rtol=0, atol=1e-6)
            or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('085 invalid native states/task weights/masks/scale')
    safe = np.where(valid, truth, 0.)
    dlat = states[..., 0] - safe[..., 0]
    dlon = (states[..., 1] - safe[..., 1] + 180.) % 360. - 180.
    errors = ((dlat / scale[0]) ** 2 + (dlon / scale[1]) ** 2,
        ((states[..., 2] - safe[..., 2]) / scale[2]) ** 2,
        np.abs((states[..., 3] - safe[..., 3] + 180.) % 360. - 180.) / scale[3])
    masks = (valid[..., :2].all(-1), valid[..., 2], valid[..., 3])
    physical = attribute_errors(states, truth)
    count = np.stack([m.sum(-1) for m in masks], -1)
    result = dict(count=count)
    for name in ('loss', 'sigma', 'coefficient', 'weighted_loss', 'contribution', 'physical_mae', 'physical_mse'):
        result[name] = np.full(count.shape, np.nan)
    for task, (error, mask) in enumerate(zip(errors, masks)):
        known = count[:, task] > 0
        denominator = np.maximum(count[:, task], 1)
        loss = np.where(mask, error, 0.).sum(-1) / denominator
        sigma = np.where(mask, -np.log(weights[..., task] + EPS), 0.).sum(-1) / denominator
        coefficient = np.exp(-sigma)
        values = dict(loss=loss, sigma=sigma, coefficient=coefficient, weighted_loss=coefficient * loss,
            contribution=coefficient * loss + sigma)
        for key, error_values in (('physical_mae', physical[TASKS[task]]),
                                  ('physical_mse', physical[TASKS[task]] ** 2)):
            values[key] = np.where(mask, error_values, 0.).sum(-1) / denominator
        for key, value in values.items():
            result[key][known, task] = value[known]
    result['total'] = np.nansum(result['contribution'], axis=-1)
    if any(not np.isfinite(result[k][count > 0]).all() for k in ('loss', 'sigma', 'coefficient', 'contribution')):
        raise ValueError('085 nonfinite valid objective')
    return result


def summarize_objectives(records):
    result = {}
    for scope in SCOPES:
        select = np.concatenate([r['selected'] if scope == 'unique_selected_groups'
            else np.ones(len(r['selected']), bool) for r in records])
        arrays = {key: np.concatenate([r[key] for r in records])[select]
                  for key in ('count', 'loss', 'sigma', 'coefficient', 'weighted_loss', 'contribution',
                              'physical_mae', 'physical_mse', 'total')}
        supervised = arrays['count'].any(-1)
        row = dict(group_count=int(select.sum()), supervised_group_count=int(supervised.sum()),
            unsupervised_group_count=int((~supervised).sum()),
            total_objective_group_mean=float(arrays['total'].mean()) if len(arrays['total']) else None,
            total_objective_group_sum=float(arrays['total'].sum()), tasks={})
        for i, task in enumerate(TASKS):
            count = arrays['count'][:, i]; known = count > 0; total = int(count.sum())
            entry = dict(valid_group_count=int(known.sum()), valid_point_count=total)
            for key in ('loss', 'sigma', 'coefficient', 'weighted_loss', 'contribution', 'physical_mae', 'physical_mse'):
                entry[key + '_group_mean'] = float(arrays[key][known, i].mean()) if known.any() else None
            for key in ('loss', 'physical_mae', 'physical_mse'):
                entry[key + '_point_mean'] = float((arrays[key][known, i] * count[known]).sum() / total) if total else None
            entry['physical_rmse_point'] = float(np.sqrt(entry['physical_mse_point_mean'])) if total else None
            row['tasks'][task] = entry
        result[scope] = row
    return result


def endpoint_comparison(initial, final):
    """Group-paired changes; masks and the selected-group set must stay fixed."""
    for a, b in zip(initial, final):
        _same(a['count'], b['count'], '085 endpoint task masks/counts')
        _same(a['selected'], b['selected'], '085 endpoint selected groups')
    changes = {}
    for scope in SCOPES:
        select = np.concatenate([r['selected'] if scope == 'unique_selected_groups'
            else np.ones(len(r['selected']), bool) for r in initial])
        counts = np.concatenate([r['count'] for r in initial])[select]
        changes[scope] = {}
        for i, task in enumerate(TASKS):
            known = counts[:, i] > 0
            values = {}
            for key in ('loss', 'physical_mae', 'physical_mse', 'contribution'):
                a, b = [np.concatenate([r[key] for r in records])[select, i][known] for records in (initial, final)]
                delta = b - a
                values[key + '_group_mean_change'] = float(delta.mean()) if len(delta) else None
            a, b = [np.concatenate([r['loss'] for r in records])[select, i][known] for records in (initial, final)]
            p, q = [np.concatenate([r['physical_mae'] for r in records])[select, i][known] for records in (initial, final)]
            # Exact signs describe numeric changes, not scientific significance.
            values['group_sign_table'] = {f'objective_{x}_physical_{y}': int(((np.sign(b-a) == sx) & (np.sign(q-p) == sy)).sum())
                for x, sx in (('lower', -1), ('equal', 0), ('higher', 1))
                for y, sy in (('lower', -1), ('equal', 0), ('higher', 1))}
            changes[scope][task] = values
    return changes


def audit_training_log(bound, check=None):
    """Logged batches are trends, not fixed-T endpoint or per-sample losses."""
    config, fit = bound['fit']['configuration'], bound['fit']
    steps, size = config['training']['updates'], config['training']['batch_size']
    rows = []; scenes, anchors = Counter(), Counter(); counts = np.zeros(3, np.int64)
    with (bound['path'] / 'fit/training.jsonl').open() as handle:
        for step, line in enumerate(handle, 1):
            if check: check()
            row = json.loads(line)
            if (row.get('step') != step or row.get('mode') != MODES[(step-1) % 3]
                    or len(row.get('batch', [])) != size):
                raise ValueError('085 actual training log step/mode/batch differs')
            for key in ('loss', 'l2', 'gradient_norm', 'gradient_clip_scale', 'task_losses', 'task_sigmas'):
                value = np.asarray(row.get(key), np.float64)
                if not np.isfinite(value).all() or value.shape != ((3,) if key.startswith('task_') else ()):
                    raise ValueError('085 invalid logged objective/gradient statistic/' + key)
            clip = config['training']['gradient_clip']
            if (row['gradient_norm'] < 0 or row['l2'] < 0 or np.any(np.asarray(row['task_losses']) < 0)
                    or type(row.get('gradient_clipped')) is not bool
                    or row['gradient_clipped'] != (row['gradient_norm'] > clip)
                    or not np.isclose(row['gradient_clip_scale'], min(1., clip/(row['gradient_norm']+1e-6)), rtol=1e-12, atol=1e-12)):
                raise ValueError('085 logged clipping/loss semantics differ')
            for sample in row['batch']:
                c = sample.get('task_valid_counts')
                if (sample.get('scene_id') not in fit['training_scene_ids'] or not isinstance(c, list) or len(c) != 3
                        or any(type(v) is not int or v < 0 or v > config['data']['length'] for v in c)
                        or sample.get('valid_position_points') != c[0]):
                    raise ValueError('085 actual training exposure malformed')
                counts += c; scenes[sample['scene_id']] += 1; anchors[sample['node_id']] += 1
            rows.append(row)
    expected = dict(successful_updates=steps, sampled_anchors=steps*size, valid_position_points=int(counts[0]),
                    task_valid_counts=counts.tolist(), scenes=dict(scenes), anchors=dict(anchors))
    if len(rows) != steps or any(fit['exposure'].get(k) != v for k, v in expected.items()):
        raise ValueError('085 actual successful training exposure differs')
    if sum(bool(r['gradient_clipped']) for r in rows) != fit['gradient_clipping']['clipped_updates']:
        raise ValueError('085 clipped update count differs')

    def trend(part):
        return dict(update_count=len(part), first_step=part[0]['step'], last_step=part[-1]['step'],
            loss_mean=float(np.mean([r['loss'] for r in part])), l2_mean=float(np.mean([r['l2'] for r in part])),
            task_losses_mean=np.mean([r['task_losses'] for r in part], 0).tolist(),
            task_sigmas_mean=np.mean([r['task_sigmas'] for r in part], 0).tolist(),
            gradient_norm_median=float(np.median([r['gradient_norm'] for r in part])),
            clipped_updates=sum(bool(r['gradient_clipped']) for r in part))
    return dict(actual_successful_exposure_verified=True, exposure=expected,
        by_mode={mode: trend([r for r in rows if r['mode'] == mode]) for mode in MODES if any(r['mode'] == mode for r in rows)},
        fixed_100_step_windows=[trend(rows[i:i+100]) for i in range(0, len(rows), 100)],
        fixed_T_endpoint_comparison='UNKNOWN: no fixed-T endpoint predictions in these caches',
        per_sample_weighted_task_terms='UNIDENTIFIABLE: logs store batch means, not joint per-sample terms'), rows


def analyze(direct, residual, check=None):
    """Validated metadata in, compact summary and reproducible per-scene rows out."""
    if (direct['scene_ids'] != residual['scene_ids'] or direct['components'] != residual['components']
            or direct['normalizer'] != residual['normalizer']
            or direct['summary']['source_system'] != residual['summary']['source_system']
            or any(direct['fit']['configuration'][k] != residual['fit']['configuration'][k]
                   for k in ('training', 'data', 'train_splits', 'motion_observation_policy'))):
        raise ValueError('085 direct/residual frozen population or normalizer differs')
    results, output_rows = {}, []
    prior_inputs = {}
    training_rows = {}
    for name, bound in (('direct074C', direct), ('residual076B', residual)):
        training, training_rows[name] = audit_training_log(bound, check)
        records, source_replay = {}, {}
        for label in ENDPOINTS:
            cached = bound['endpoints'][label]; records[label] = []; replay = []
            for index, sid in enumerate(bound['scene_ids']):
                if check: check()
                endpoint = bound['path'] / label
                checked = replay_scene(endpoint, sid)  # all six methods, partitions, masks and physical errors
                original = cached['rows'][index]
                if (original['component_id'] != bound['components'][sid]
                        or original['fragment_count'] != checked['fragment_count']):
                    raise ValueError('085 source per-scene metadata differs')
                for method in SOURCE_METHODS:
                    if original['methods'][method]['population'] != checked['population']:
                        raise ValueError('085 source per-scene population differs')
                    assert_metrics(checked['replay'][method]['full_target_metrics'],
                        original['methods'][method]['full_target_metrics'], '085 source scene/' + sid + '/' + method)
                fragment = _read(endpoint / 'predictions' / f'states-{sid}.npz')
                system = _read(endpoint / 'system-predictions' / f'states-{sid}.npz')
                base = _artifact(system, 'anchor'); model = _artifact(system, 'ptfm_adapted')
                # Bind ALL original inputs, ordinary outputs and correspondence across both endpoints/methods.
                fixed = {k: fragment[k] for k in ('node_ids', 'identities', 'target_identity_ids', 'native_states',
                    'support', 'nearest_seconds', 'exact_observation', 'query_seconds',
                    'learned_assoc_group_offsets', 'learned_assoc_group_node_indices')}
                fixed.update({'system_' + k: v for k, v in system.items() if not k.startswith('ptfm_adapted_')})
                if sid not in prior_inputs: prior_inputs[sid] = fixed
                else:
                    if set(fixed) != set(prior_inputs[sid]): raise ValueError('085 fixed cache schema differs')
                    for key in fixed: _same(fixed[key], prior_inputs[sid][key], '085 original inputs/ordinary/' + key)
                anchors = base['group_anchor_target_indices']; t = len(base['target_ids'])
                truth = base['aligned_truth'][:t][anchors]
                valid = base['aligned_truth_valid'][:t][anchors] & base['group_valid']
                record = group_objectives(model['group_states'], truth, valid, fragment['group_task_weights'],
                                          bound['normalizer']['scale'])
                selected = base['aligned_group_indices'][:t]; used = np.zeros(len(anchors), bool)
                used[selected[selected >= 0]] = True
                record.update(selected=used, scene_id=sid, component_id=bound['components'][sid])
                if name == 'residual076B' and label == 'initial':
                    _same(model['group_states'][..., :2], _artifact(system, 'fixed_prior095')['group_states'][..., :2],
                          '085 residual initial prior095')
                records[label].append(record); replay.append(checked)
                output_rows.append(dict(method=name, endpoint=label, scene_id=sid, component_id=record['component_id'],
                    population=checked['population'], objectives=summarize_objectives([record])))
            pooled = {}
            for method in SOURCE_METHODS:
                pooled[method] = pool_metrics([r['replay'][method]['full_target_metrics'] for r in replay])
                assert_metrics(pooled[method], cached['summary']['methods'][method]['full_target_metrics'], '085 pooled/' + method)
                population = {k: sum(r['population'][k] for r in replay) for k in replay[0]['population']}
                if population != cached['summary']['methods'][method]['population']:
                    raise ValueError('085 source pooled population differs')
                for c, component in cached['summary']['components'].items():
                    selected = [r for r in replay if bound['components'][r['scene_id']] == str(c)]
                    assert_metrics(pool_metrics([r['replay'][method]['full_target_metrics'] for r in selected]),
                        component['methods'][method]['full_target_metrics'], '085 source component/' + str(c))
            if sum(r['fragment_count'] for r in replay) != cached['summary']['fragment_count']:
                raise ValueError('085 source fragment population differs')
            source_replay[label] = pooled['ptfm_adapted']
            # Selected-group physical metrics must also close against the native source scorer.
            diagnostic = summarize_objectives(records[label])['unique_selected_groups']['tasks']
            for task, value in diagnostic.items():
                expected = pooled['ptfm_adapted']['all'][task]
                if value['valid_point_count'] != expected['intersection_count']:
                    raise ValueError('085 selected objective population differs')
                for key, expected_key in (('physical_mae_point_mean', 'mae'), ('physical_mse_point_mean', 'mse')):
                    a, b = value[key], expected[expected_key]
                    if (a is None or b is None) and a is not b or a is not None and b is not None and not np.isclose(a, b, rtol=1e-10, atol=1e-9):
                        raise ValueError('085 selected objective physical replay differs')
        results[name] = dict(training_log=training, endpoints={label: summarize_objectives(records[label]) for label in ENDPOINTS},
            final_minus_initial=endpoint_comparison(records['initial'], records['final']), source_physical_replay=source_replay,
            components={c: {label: summarize_objectives([r for r in records[label] if r['component_id'] == c])
                for label in ENDPOINTS} for c in sorted(set(bound['components'].values()))})
    for a, b in zip(training_rows['direct074C'], training_rows['residual076B']):
        if any(a[k] != b[k] for k in ('step', 'mode', 'batch')):
            raise ValueError('085 direct/residual actual exposure pairing differs')
    return dict(status='COMPLETED', scene_ids=direct['scene_ids'], scene_count=len(direct['scene_ids']),
        methods=results, source_physical_population_replayed=True, actual_training_batches_paired=True,
        fixed_T_endpoint_comparison='UNKNOWN: requires inputs and new frozen endpoint forwards, outside085',
        V_l2='UNIDENTIFIABLE: weights never loaded; V objective excludes LSTM regularizer',
        decision='Current PTFM recipes remain CLOSED. No training, correction or native-task qualification authorized by this audit.',
        limitations=['V reused, shared identities; group diagnostics are not independent training repeats.',
            'All-output loss maps each anchor identity to its existing reference, including duplicates; not formal unique-target system scoring.',
            'Unique selected-group loss is group-equal; point-equal physical metrics are separately reported.',
            'Effective task coefficients are geometric mean weights on each task mask, not helper reliability.',
            'Training logs contain batch means on changing samples; no exp(mean sigma)*mean loss reconstruction is valid.',
            'Objective/metric disagreement is descriptive, not proof of the cause of training failure.',
            'This neither constructs the original PTFM dual-source window task nor certifies author reproduction.']), output_rows
