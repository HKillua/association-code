"""075: audit actual TSA successful-update logs without weights or data access.

Only run.json and training.jsonl are read. Scene co-membership describes the
InfoNCE negative inventory, not geographical separation or negative difficulty.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np


TASKS = ('ITSA', 'MSTSA')
SCOPE = 'tsactfer-native-level1-final-v1'


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _finite(value, name, minimum=0.):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (minimum is not None and value < minimum)):
        raise ValueError(f'{name} must be finite and >= {minimum}')
    return float(value)


def _schedule(step, training):
    # Exact scalar formula used by the source trainer, without importing torch.
    warmup, total = training['warmup_updates'], training['updates']
    multiplier = (step / warmup if step <= warmup else
        .5 * (1 + math.cos(math.pi * (step-warmup) / (total-warmup))))
    return training['learning_rate'] * multiplier


def negative_inventory(scene_ids):
    """Ordered negative comparisons over left[B], right[B]; omit self/positive.

    Every identity occurs exactly twice. Distinct identities are a caller
    invariant: for k identities in one scene, 2k tracks each see 2(k-1)
    same-scene negatives. This counts denominator comparisons, not distinct
    unordered trajectory pairs or actual high-loss/false-negative terms.
    """
    counts = Counter(scene_ids)
    batch_size = len(scene_ids)
    if batch_size < 2:
        raise ValueError('InfoNCE inventory needs at least two identity pairs')
    total = 4 * batch_size * (batch_size-1)
    same = 4 * sum(count*(count-1) for count in counts.values())
    return dict(ordered_negative_comparisons=total,
                same_scene_ordered_negatives=same,
                cross_scene_ordered_negatives=total-same,
                same_scene_fraction=same/total,
                batch_distinct_scenes=len(counts),
                maximum_identity_pairs_from_one_scene=max(counts.values()))


def _distribution(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return dict(count=0, mean=None, min=None, median=None, p95=None, p99=None, max=None)
    return dict(count=len(values), mean=float(values.mean()), min=float(values.min()),
                median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                p99=float(np.quantile(values, .99)), max=float(values.max()))


def _summary(rows, clip):
    total_negative = sum(r['ordered_negative_comparisons'] for r in rows)
    same_negative = sum(r['same_scene_ordered_negatives'] for r in rows)
    pairs = sum(r['pair_count'] for r in rows)
    input_points = sum(r['input_points'] for r in rows)
    output_points = sum(r['output_points'] for r in rows)
    clipped = sum(r['gradient_norm'] > clip for r in rows)
    return dict(successful_optimizer_calls=len(rows),
        first_global_step=rows[0]['step'], last_global_step=rows[-1]['step'],
        nonzero_lr_calls=sum(r['learning_rate'] > 0 for r in rows),
        loss=_distribution([r['loss'] for r in rows]),
        learning_rate=_distribution([r['learning_rate'] for r in rows]),
        gradient_norm=_distribution([r['gradient_norm'] for r in rows]),
        gradient_clip_scale=_distribution([r['gradient_clip_scale'] for r in rows]),
        clipped_calls=clipped, clipped_call_fraction=clipped/len(rows),
        positive_pair_exposures=pairs, track_exposures=2*pairs,
        negative_inventory=dict(ordered_negative_comparisons=total_negative,
            same_scene_ordered_negatives=same_negative,
            cross_scene_ordered_negatives=total_negative-same_negative,
            same_scene_fraction=same_negative/total_negative,
            per_batch_same_scene_fraction=_distribution([r['same_scene_fraction'] for r in rows]),
            batch_distinct_scenes=_distribution([r['batch_distinct_scenes'] for r in rows]),
            maximum_identity_pairs_from_one_scene=_distribution(
                [r['maximum_identity_pairs_from_one_scene'] for r in rows])),
        crop=dict(cropped_pairs=sum(r['cropped_pairs'] for r in rows),
            cropped_pair_fraction=sum(r['cropped_pairs'] for r in rows)/pairs,
            fallback_pairs=sum(r['fallback_pairs'] for r in rows),
            fallback_pair_fraction=sum(r['fallback_pairs'] for r in rows)/pairs,
            requested_fraction_mean=sum(r['fraction_sum'] for r in rows)/pairs,
            requested_fraction_min=min(r['fraction_min'] for r in rows),
            requested_fraction_max=max(r['fraction_max'] for r in rows),
            requested_window_seconds_mean=sum(r['duration_sum'] for r in rows)/pairs,
            requested_window_seconds_min=min(r['duration_min'] for r in rows),
            requested_window_seconds_max=max(r['duration_max'] for r in rows),
            input_points_before_crop_after_native_cap=input_points,
            output_points_after_crop=output_points,
            token_retention_fraction=output_points/input_points,
            token_count_per_update=_distribution([r['output_points'] for r in rows])),
        logged_training_seconds=sum(r['seconds'] for r in rows))


def analyze_training(fit_dir, expected_config, train_scene_ids, check=None):
    """Validate and summarize a complete, fixed-final source fit in one log pass.

    Checkpoint/weights and independent source-cache binding belong to the caller.
    A matching completed run does not itself prove successful scientific results.
    """
    fit_dir = Path(fit_dir)
    with (fit_dir/'run.json').open() as handle:
        run = json.load(handle)
    ids = [str(sid) for sid in train_scene_ids]
    config = expected_config
    settings = config['training']
    updates = _integer(settings['updates'], 'updates', 2)
    batch_size = _integer(settings['pair_batch_size'], 'pair_batch_size', 2)
    warmup = _integer(settings['warmup_updates'], 'warmup_updates', 1)
    clip = _finite(settings['gradient_clip'], 'gradient_clip')
    length = _integer(config['data']['max_points'], 'max_points', 2)
    minimum_fraction = _finite(config['data']['min_crop_fraction'], 'min_crop_fraction')
    if (updates % 2 or warmup >= updates or clip <= 0 or not 0 < minimum_fraction <= 1
            or not ids or len(ids) != len(set(ids)) or config['scope'] != SCOPE):
        raise ValueError('invalid fixed two-head configuration or ordered T population')
    if (run.get('scope') != SCOPE or run.get('status') != 'COMPLETED'
            or run.get('configuration') != config or run.get('training_scene_ids') != ids
            or run.get('optimizer_updates') != updates or run.get('final_checkpoint') != 'final.pt'
            or run.get('reference_read') is not False or run.get('test_opened') is not False
            or 'error' in run):
        raise ValueError('completed fit/configuration/population/endpoint binding differs')
    normalizer = run['normalizer']
    if (normalizer.get('training_scene_ids') != ids
            or normalizer.get('training_splits') != ['T_core']
            or normalizer.get('native_max_points') != length
            or normalizer.get('feature_order') != ['lon', 'lat']):
        raise ValueError('normalizer T population/native input contract differs')
    _integer(normalizer['observed_point_count'], 'normalizer observed_point_count', 1)
    for key in ('mean', 'std'):
        values = np.asarray(normalizer[key], dtype=np.float64)
        if values.shape != (2,) or not np.isfinite(values).all() or (key == 'std' and (values <= 0).any()):
            raise ValueError('normalizer moments invalid')
    source_exposure = run['exposure']
    map_names = ('scenes', 'identities', 'pairs')
    # Consume the saved maps to validate every exposure with bounded extra memory.
    remaining = {key: source_exposure[key] for key in map_names}
    unique = {key: len(remaining[key]) for key in map_names}
    for mapping in remaining.values():
        for value in mapping.values():
            _integer(value, 'saved exposure count', 1)
    totals = dict(successful_optimizer_calls=0, nonzero_lr_updates=0,
        positive_pair_exposures=0, track_exposures=0, token_exposures=0,
        task_calls={task: 0 for task in TASKS}, cropped_pairs=0, fallback_pairs=0)
    expected_totals = {key: value for key, value in source_exposure.items() if key not in map_names}
    rows, by_head = [], {task: [] for task in TASKS}
    head_scenes = {task: Counter() for task in TASKS}
    train_set = set(ids)
    with (fit_dir/'training.jsonl').open() as handle:
        for step, line in enumerate(handle, 1):
            if check:
                check()
            row = json.loads(line)
            task = 'ITSA' if step % 2 else 'MSTSA'
            if (step > updates or type(row.get('step')) is not int or row['step'] != step
                    or row.get('task') != task or row.get('learning_rate') != _schedule(step, settings)):
                raise ValueError('actual step/task/learning-rate schedule differs')
            for key in ('loss', 'gradient_norm', 'gradient_clip_scale', 'seconds', 'learning_rate'):
                _finite(row[key], key)
            expected_scale = min(1., clip/(row['gradient_norm']+1e-6))
            if row['gradient_clip_scale'] != expected_scale:
                raise ValueError('recorded gradient clip scale differs from actual norm')
            samples = row['batch']
            if not isinstance(samples, list) or len(samples) != batch_size:
                raise ValueError('actual batch size differs')
            identities = [s['identity'] for s in samples]
            if any(type(i) is not str or not i for i in identities) or len(set(identities)) != batch_size:
                raise ValueError('InfoNCE batch must contain distinct nonempty identities')
            fractions, durations, fallback = [], [], 0
            for sample in samples:
                sid = sample['scene_id']
                left = _integer(sample['left_index'], 'left_index')
                right = _integer(sample['right_index'], 'right_index')
                if (sid not in train_set or left == right or sample['left_node'] == sample['right_node']
                        or any(type(sample[k]) is not str or not sample[k] for k in ('left_node', 'right_node'))):
                    raise ValueError('exposure contains unknown scene or invalid/self node pair')
                keys = dict(scenes=sid, identities=sample['identity'],
                    pairs=json.dumps([sid, left, right], separators=(',', ':')))
                for field, key in keys.items():
                    if remaining[field].get(key, 0) <= 0:
                        raise ValueError('actual exposure exceeds or differs from saved audit')
                    remaining[field][key] -= 1
                head_scenes[task][sid] += 1
                crop = sample['crop']
                start = _finite(crop['start'], 'crop start', None)
                stop = _finite(crop['stop'], 'crop stop', None)
                fraction = _finite(crop['fraction'], 'crop fraction')
                if (stop <= start or not minimum_fraction <= fraction <= 1
                        or type(crop['fallback']) is not bool or (task == 'ITSA' and crop['fallback'])):
                    raise ValueError('recorded crop window/fraction/fallback differs')
                fractions.append(fraction); durations.append(stop-start)
                fallback += int(crop['fallback'])
            augmentation = row['augmentation']
            for key in ('pair_count', 'cropped_pairs', 'fallback_pairs', 'input_points', 'output_points'):
                _integer(augmentation[key], 'augmentation '+key)
            token_count = _integer(row['token_count'], 'token_count')
            if (augmentation['pair_count'] != batch_size or augmentation['fallback_pairs'] != fallback
                    or augmentation['cropped_pairs']+fallback > batch_size
                    or not 4*batch_size <= token_count <= augmentation['input_points'] <= 2*batch_size*length
                    or augmentation['output_points'] != token_count
                    or (augmentation['cropped_pairs'] == 0) != (augmentation['input_points'] == token_count)):
                raise ValueError('crop/token counts inconsistent with recorded batch')
            totals['successful_optimizer_calls'] += 1
            totals['nonzero_lr_updates'] += int(row['learning_rate'] > 0)
            totals['task_calls'][task] += 1
            totals['positive_pair_exposures'] += batch_size
            totals['track_exposures'] += 2*batch_size
            totals['token_exposures'] += token_count
            for key in ('cropped_pairs', 'fallback_pairs'):
                totals[key] += augmentation[key]
            compact = {key: row[key] for key in ('step', 'task', 'loss', 'learning_rate',
                'gradient_norm', 'gradient_clip_scale', 'seconds')}
            compact.update(augmentation)
            compact.update(negative_inventory([s['scene_id'] for s in samples]),
                fraction_sum=sum(fractions), fraction_min=min(fractions), fraction_max=max(fractions),
                duration_sum=sum(durations), duration_min=min(durations), duration_max=max(durations))
            rows.append(compact); by_head[task].append(compact)
    totals.update(unique_scenes=unique['scenes'], unique_identities=unique['identities'],
        unique_pairs=unique['pairs'], repeated_pair_exposures=totals['positive_pair_exposures']-unique['pairs'])
    if (totals['successful_optimizer_calls'] != updates or totals != expected_totals
            or any(any(value != 0 for value in mapping.values()) for mapping in remaining.values())):
        raise ValueError('actual completed exposure does not reproduce saved audit')
    clipping = sum(row['gradient_norm'] > clip for row in rows)
    if run.get('gradient_clipping') != {'clipped_updates': clipping}:
        raise ValueError('recorded gradient clipping aggregate differs')
    result = dict(status='VERIFIED', source_files=['run.json', 'training.jsonl'],
        scope='tsactfer-training-log-diagnostic-v1',
        verification=dict(completed_fixed_final=True, exact_configuration=True,
            training_population_and_normalizer=True, actual_successful_exposure=True,
            alternating_tasks_and_schedule=True, crop_and_token_counts=True,
            gradient_clipping_aggregate=True, model_weights_read=False,
            model_forward=False, archive_read=False, test_opened=False),
        denominators=dict(negative_comparisons='ordered 2B-track InfoNCE negatives, excluding self and paired positive; 4B(B-1)',
            same_scene='4*sum_s k_s*(k_s-1), k_s = actual identity pairs from scene s',
            window='100 successful optimizer calls PER HEAD; global step boundaries explicit',
            tokens='observed packed native tokens after the fixed cap; input_points precedes augmentation only'),
        limitations=['Scene equality does not establish spatial/temporal proximity or negative difficulty.',
            'No embeddings, similarities or per-negative loss are saved in this log.',
            'Requested crop intervals are not reconstructed from observations; fallback is reported explicitly.',
            'No per-pair retained token counts or original uncapped observation counts are inferred.',
            'Observed loss/gradient trends do not prove convergence or a causal failure mechanism.'],
        actual_exposure=totals, all_updates=_summary(rows, clip), heads={})
    for task, task_rows in by_head.items():
        result['heads'][task] = dict(summary=_summary(task_rows, clip),
            scene_positive_pair_exposures=dict(sorted(head_scenes[task].items())),
            windows=[dict(first_head_call=begin+1, last_head_call=begin+len(part), **_summary(part, clip))
                for begin in range(0, len(task_rows), 100) if (part := task_rows[begin:begin+100])])
    return result
