"""069 read-only audit of the successful 068 optimizer-update log.

This module deliberately imports neither Torch nor model/training code. The
logged batch loss is a mean of per-anchor mean log errors; it cannot recover
linear position MAE, per-query gradients or gate saturation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

MODES = ('clean', 'clean_plus_wrong', 'wrong_only')
_SCOPE = 'support-gate-v1'
_ROUTING = 'supported-mean-position-gate-v1'
_LOSS = 'mean_valid_position_log1p_haversine_m_div_1000_all_queries_no_auxiliary'
_COUNTS = ('eligible_query_points', 'eligible_valid_position_points', 'valid_position_points')


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be a finite nonnegative number')
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(name + ' must be a finite nonnegative number')
    return number


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _read_json(value):
    def reject_constant(token):
        raise ValueError('nonfinite JSON token: ' + token)
    return json.loads(value, object_pairs_hook=_object, parse_constant=reject_constant)


def _same_counts(actual, expected, name='exposure'):
    """Do not let bool/int equality hide a malformed successful-update audit."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(name + ' keys disagree with recomputed exposure')
        for key in expected:
            _same_counts(actual[key], expected[key], name + '.' + key)
    elif type(actual) is not int or actual != expected:
        raise ValueError(name + ' disagrees with recomputed exposure')


def _stats(values):
    if not values:
        return dict(count=0, mean=None, minimum=None, p05=None, p50=None,
                    p95=None, p99=None, maximum=None)
    array = np.asarray(values, dtype=np.float64)
    q = np.quantile(array, [.05, .50, .95, .99], method='linear')
    return dict(count=len(values), mean=float(array.mean()), minimum=float(array.min()),
                p05=float(q[0]), p50=float(q[1]), p95=float(q[2]), p99=float(q[3]),
                maximum=float(array.max()))


def _window(rows, clip):
    output = {}
    for name in ('all', *MODES):
        selected = rows if name == 'all' else [row for row in rows if row['mode'] == name]
        norms = [row['gradient_norm'] for row in selected]
        output[name] = dict(log_loss=_stats([row['loss'] for row in selected]),
            preclip_gradient_norm=_stats(norms),
            preclip_norm_greater_than_configured_clip_count=sum(value > clip for value in norms),
            nonzero_gradient_count=sum(value > 0 for value in norms))
    return output


def audit_training(fit_dir: Path, expected_updates=1000) -> dict:
    """Verify complete fit/run.json and training.jsonl without loading weights.

    expected_updates is overridable for synthetic tests. The 069 executable must
    bind the production source to completed 068 and require exactly 1,000 steps.
    No compressed aggregate or checkpoint substitutes for the actual JSONL.
    """
    expected_updates = _integer(expected_updates, 'expected_updates', minimum=1)
    fit_dir = Path(fit_dir)
    run = _read_json((fit_dir/'run.json').read_text())
    if (not isinstance(run, dict) or run.get('status') != 'COMPLETED'
            or run.get('test_opened') is not False or run.get('scope') != _SCOPE
            or run.get('routing_semantics') != _ROUTING or run.get('loss_semantics') != _LOSS
            or run.get('motion_observation_policy') != 'canonical_all'
            or run.get('final_checkpoint') != 'final.pt'):
        raise ValueError('training source must be completed 068 with frozen routing/loss and closed test')
    if _integer(run.get('optimizer_updates'), 'optimizer_updates') != expected_updates:
        raise ValueError('actual optimizer updates disagree with expected updates')
    config = run.get('configuration', {})
    training = config.get('training', {})
    if (config.get('scope') != _SCOPE or config.get('routing_semantics') != _ROUTING
            or config.get('motion_observation_policy') != 'canonical_all'
            or config.get('train_splits') != ['T_core']
            or config.get('model') != dict(initial_gate=.95)
            or config.get('data') != dict(length=64, max_observations=128,
                time_unit_seconds=20., reference_longitude_policy='shortest_arc')
            or training.get('seed') != 20261021 or training.get('learning_rate') != .001
            or training.get('gradient_clip') != 1.):
        raise ValueError('training source configuration is not the frozen 068 protocol')
    if (_integer(training.get('updates'), 'configured updates', minimum=1) != expected_updates
            or _integer(training.get('batch_size'), 'batch_size', minimum=1) != 8):
        raise ValueError('configured updates/batch8 disagree with frozen audit protocol')
    scenes = run.get('training_scene_ids')
    if (not isinstance(scenes, list) or not scenes
            or any(not isinstance(value, str) or not value for value in scenes)
            or len(set(scenes)) != len(scenes)):
        raise ValueError('training_scene_ids must be distinct nonempty T_core scene names')
    declared_scenes = set(scenes)
    exposure = dict(successful_updates=0, nonzero_gradient_updates=0, zero_eligible_updates=0,
        sampled_anchors=0, eligible_query_points=0, eligible_valid_position_points=0,
        valid_position_points=0, scenes={}, anchors={}, modes={mode: dict(updates=0, anchors=0,
            eligible_query_points=0, eligible_valid_position_points=0, valid_position_points=0,
            nonzero_gradient_updates=0, zero_eligible_updates=0) for mode in MODES})
    rows, node_bindings, anchor_bindings = [], {}, {}
    with (fit_dir/'training.jsonl').open() as handle:
        for expected_step, line in enumerate(handle, 1):
            if expected_step > expected_updates:
                raise ValueError('training log has updates beyond the fixed terminal step')
            row = _read_json(line)
            if not isinstance(row, dict) or set(row) != {'step', 'mode', 'loss', 'gradient_norm', 'batch'}:
                raise ValueError('training log row schema disagrees with frozen 068')
            if _integer(row['step'], 'step', minimum=1) != expected_step:
                raise ValueError('successful training steps must be exactly 1..updates in order')
            mode = MODES[(expected_step-1) % len(MODES)]
            if row['mode'] != mode:
                raise ValueError('training mode disagrees with the fixed three-mode cycle')
            loss = _finite(row['loss'], 'loss')
            gradient = _finite(row['gradient_norm'], 'gradient_norm')
            batch = row['batch']
            if not isinstance(batch, list) or len(batch) != 8:
                raise ValueError('each successful update must have exactly eight anchor draws')
            totals = dict.fromkeys(_COUNTS, 0)
            for member in batch:
                if (not isinstance(member, dict) or set(member) != {
                        'scene_id', 'anchor_index', 'node_id', 'members', *_COUNTS}):
                    raise ValueError('training batch schema disagrees with frozen 068')
                scene, node = member['scene_id'], member['node_id']
                if not isinstance(scene, str) or scene not in declared_scenes:
                    raise ValueError('training batch references a scene outside declared T_core')
                if not isinstance(node, str) or not node:
                    raise ValueError('node_id must be a nonempty fragment identifier')
                anchor = _integer(member['anchor_index'], 'anchor_index')
                members = member['members']
                if (not isinstance(members, list) or not members
                        or any(type(value) is not int or value < 0 for value in members)
                        or len(set(members)) != len(members) or members[0] != anchor):
                    raise ValueError('members must be distinct nonnegative indices with anchor first')
                binding = (scene, anchor)
                if (node in node_bindings and node_bindings[node] != binding
                        or binding in anchor_bindings and anchor_bindings[binding] != node):
                    raise ValueError('fragment node_id and scene/anchor binding changed in training log')
                node_bindings[node], anchor_bindings[binding] = binding, node
                counts = {key: _integer(member[key], key) for key in _COUNTS}
                eligible, overlap, valid = (counts[key] for key in _COUNTS)
                if (max(eligible, valid) > 64 or not max(0, eligible+valid-64) <= overlap <= min(eligible, valid)):
                    raise ValueError('eligible/valid point counts cannot represent Q64 masks')
                for key in _COUNTS:
                    totals[key] += counts[key]
                exposure['scenes'][scene] = exposure['scenes'].get(scene, 0) + 1
                exposure['anchors'][node] = exposure['anchors'].get(node, 0) + 1
            nonzero, zero_eligible = int(gradient > 0), int(totals['eligible_query_points'] == 0)
            exposure['successful_updates'] += 1
            exposure['sampled_anchors'] += len(batch)
            exposure['nonzero_gradient_updates'] += nonzero
            exposure['zero_eligible_updates'] += zero_eligible
            counter = exposure['modes'][mode]
            counter['updates'] += 1
            counter['anchors'] += len(batch)
            counter['nonzero_gradient_updates'] += nonzero
            counter['zero_eligible_updates'] += zero_eligible
            for key in _COUNTS:
                exposure[key] += totals[key]
                counter[key] += totals[key]
            rows.append(dict(step=expected_step, mode=mode, loss=loss, gradient_norm=gradient))
    if len(rows) != expected_updates:
        raise ValueError('training log is incomplete: expected every successful optimizer update')
    _same_counts(run.get('exposure'), exposure)
    if _integer(run.get('trained_scene_count'), 'trained_scene_count') != len(exposure['scenes']):
        raise ValueError('trained_scene_count disagrees with actual logged scene visits')
    clip = float(training['gradient_clip'])
    windows = [dict(first_step=start+1, last_step=min(start+100, len(rows)),
                    statistics=_window(rows[start:start+100], clip))
               for start in range(0, len(rows), 100)]
    return dict(status='COMPLETED', source_status=run['status'], test_opened=False,
        scope='support-gate-training-log-audit-v1', expected_updates=expected_updates,
        logged_successful_updates=len(rows), run_exposure_exact=True,
        exposure=exposure, declared_training_scene_count=len(scenes),
        sampled_training_scene_count=len(exposure['scenes']),
        unsampled_training_scene_ids=sorted(declared_scenes-set(exposure['scenes'])),
        unique_sampled_fragment_count=len(exposure['anchors']),
        repeated_fragment_draw_count=exposure['sampled_anchors']-len(exposure['anchors']),
        scene_visit_distribution=_stats(list(exposure['scenes'].values())),
        fragment_visit_distribution=_stats(list(exposure['anchors'].values())),
        window_size_updates=100, quantile_method='linear', windows=windows,
        overall=_window(rows, clip), configured_gradient_clip=clip,
        gradient_norm_semantics='total gradient norm returned by clip_grad_norm_, before clipping',
        linear_training_position_error_available=False,
        interpretation=dict(
            loss='batch mean of per-anchor mean log1p(position error metres / 1000), including valid queries outside eligibility',
            windows='fixed update windows use different sampled scenes and anchors; trends are not a fixed-population convergence test',
            linear_error='mean linear training error cannot be recovered by expm1 of logged mean log loss',
            exposure='visits count successful anchor draws; distinct fragments are not distinct target identities or complete epochs',
            gradients='nonzero total norm does not establish per-parameter learning or exclude partial sigmoid saturation',
            clipping='count of preclip norm > configured clip is not an exact count of gradient rescaling, which also uses a numerical epsilon',
            zero_eligible='zero opportunity batches still consumed an optimizer step; retained Adam momentum may change parameters',
            unavailable='no per-query training errors, gradient decomposition, gate/logit history or unexecuted candidate outcomes are reconstructed'))
