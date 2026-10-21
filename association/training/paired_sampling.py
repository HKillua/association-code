"""A fixed negative-slot intervention after the unchanged scene/pair sampler.

Only T_core labels and frozen OLS scores define the static hard pools. This
module neither calls nor shares the original sampler's RNG. Audit exposure is
committed separately, after a successful optimizer update.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


SOURCES = ('positive_uniform', 'negative_uniform', 'negative_hard')
SCORE_EDGES = np.array([-np.inf, -5., -3., -2., -1., 0.])
COUNTERS = ('sampled_pair_count', 'positive_slots', 'negative_slots', 'hard_source_slots',
            'changed_slots', 'final_negative_hard_pool_hits', 'uniform_negative_hard_pool_hits')


def validate_sampling_config(value):
    if not isinstance(value, dict) or set(value) != {
            'mode', 'intervention_seed', 'hard_slots', 'hard_fraction'}:
        raise ValueError('sampling needs mode, intervention_seed, hard_slots and hard_fraction')
    if value['mode'] not in ('uniform', 'hard_negative'):
        raise ValueError('sampling mode must be uniform or hard_negative')
    seed = value['intervention_seed']
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('intervention_seed must be a nonnegative integer')
    slots = value['hard_slots']
    fraction = value['hard_fraction']
    if isinstance(slots, (bool, np.bool_)) or not isinstance(slots, Integral) or slots != 64:
        raise ValueError('paired pilot fixes hard_slots at 64')
    if (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, (int, float))
            or not np.isfinite(fraction) or fraction != .1):
        raise ValueError('paired pilot fixes hard_fraction at 0.1')
    return dict(mode=value['mode'], intervention_seed=int(seed), hard_slots=64, hard_fraction=.1)


def _empty_scores():
    return dict(count=0, sum=0., sum_squared=0., minimum=None, maximum=None,
                histogram_counts=[0] * (len(SCORE_EDGES)-1))


def _add_scores(destination, values):
    if not len(values):
        return
    destination['count'] += len(values)
    destination['sum'] += float(values.sum())
    destination['sum_squared'] += float(np.square(values).sum())
    low, high = float(values.min()), float(values.max())
    destination['minimum'] = low if destination['minimum'] is None else min(low, destination['minimum'])
    destination['maximum'] = high if destination['maximum'] is None else max(high, destination['maximum'])
    bins = np.histogram(values, SCORE_EDGES)[0]
    destination['histogram_counts'] = [int(a+b) for a, b in zip(destination['histogram_counts'], bins)]


def _score_report(rows):
    count = sum(row['count'] for row in rows)
    total = sum(row['sum'] for row in rows)
    square = sum(row['sum_squared'] for row in rows)
    mean = total/count if count else None
    return dict(count=count, mean=mean,
        standard_deviation=float(np.sqrt(max(0., square/count-mean*mean))) if count else None,
        minimum=min((row['minimum'] for row in rows if row['count']), default=None),
        maximum=max((row['maximum'] for row in rows if row['count']), default=None),
        histogram_counts=[sum(row['histogram_counts'][i] for row in rows)
                          for i in range(len(SCORE_EDGES)-1)])


@dataclass
class _BatchRecord:
    owner: object
    groups: list
    hard_masks: list
    changed_by_group: list
    replacement: bool
    committed: bool = False


class PairedNegativeIntervention:
    """Apply uniform A or fixed 64-slot hard-negative B to a raw 256 batch.

    ``config`` is the direct ``sampling`` dictionary, not the whole trainer
    configuration. ``apply`` copies all arrays; ``record`` is opaque and only
    valid for this instance's returned batch. Never save records per update.
    """

    def __init__(self, train, config):
        self.config = validate_sampling_config(config)
        if not train or any(scene.get('split') != 'T_core' for scene in train):
            raise ValueError('paired sampling pools require nonempty T_core scenes only')
        self._rng = np.random.default_rng(self.config['intervention_seed'])
        self._owner = object()
        self._scene_ids, self._labels, self._scores, self._hard = [], [], [], []
        self._pool_scores, self._counts, self._source_scores, self._seen = [], [], [], []
        self._completed = self._replacement_batches = 0
        for scene in train:
            scene_id = str(scene['scene_id'])
            pairs, labels, scores = (np.asarray(scene[key]) for key in ('pairs', 'labels', 'base_scores'))
            if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu'
                    or labels.shape != (len(pairs),) or labels.dtype.kind not in 'biuf'
                    or not ((labels == 0) | (labels == 1)).all()
                    or scores.shape != labels.shape or scores.dtype.kind not in 'iuf'
                    or not np.isfinite(scores).all() or (scores > 0).any()):
                raise ValueError('paired sampling needs original pair indices, binary labels and finite OLS scores <=0')
            if scene_id in self._scene_ids:
                raise ValueError('paired sampling scene_ids must be unique')
            labels, scores = labels.astype(bool, copy=True), scores.astype(np.float64, copy=False)
            negative = np.flatnonzero(~labels)
            size = (len(negative)+9)//10
            # Exact K, not all tied values: original pair index is the tie-break.
            hard = negative[np.lexsort((negative, -scores[negative]))[:size]]
            pools = {name: _empty_scores() for name in ('negative', 'hard')}
            _add_scores(pools['negative'], scores[negative])
            _add_scores(pools['hard'], scores[hard])
            self._scene_ids.append(scene_id)
            self._labels.append(labels)
            self._scores.append(scores)
            self._hard.append(hard)
            self._pool_scores.append(pools)
            self._counts.append(dict(completed_batch_count=0, **dict.fromkeys(COUNTERS, 0)))
            self._source_scores.append({source: _empty_scores() for source in SOURCES})
            self._seen.append({source: set() for source in SOURCES})

    def _copy_groups(self, groups):
        copied, used, classes = [], set(), {True: set(), False: set()}
        totals = {True: 0, False: 0}
        for scene_index, indices in groups:
            if (isinstance(scene_index, (bool, np.bool_)) or not isinstance(scene_index, Integral)
                    or not 0 <= scene_index < len(self._scene_ids) or scene_index in used):
                raise ValueError('raw batch must contain distinct valid scene groups')
            scene_index, indices = int(scene_index), np.asarray(indices)
            if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
                    or (indices < 0).any() or (indices >= len(self._labels[scene_index])).any()):
                raise ValueError('raw batch pair indices are invalid')
            copied.append((scene_index, indices.copy()))
            used.add(scene_index)
            values = self._labels[scene_index][indices]
            for label in (True, False):
                count = int((values == label).sum())
                totals[label] += count
                if count:
                    classes[label].add(scene_index)
        if totals != {True: 128, False: 128} or any(len(scenes) != 1 for scenes in classes.values()):
            raise ValueError('paired pilot needs 256 slots, 128 per class from one scene each')
        return copied

    def apply(self, raw_groups):
        groups = self._copy_groups(raw_groups)
        masks = [np.zeros(len(indices), bool) for _, indices in groups]
        changes = [0] * len(groups)
        replacement = False
        if self.config['mode'] == 'hard_negative':
            for group_index, (scene_index, indices) in enumerate(groups):
                slots = np.flatnonzero(~self._labels[scene_index][indices])
                if not len(slots):
                    continue
                slots = slots[:64]  # original returned batch order, never a second shuffle
                pool = self._hard[scene_index]
                replacement = len(pool) < 64
                draws = self._rng.choice(pool, size=64, replace=replacement)
                changes[group_index] = int((indices[slots] != draws).sum())
                indices[slots] = draws
                masks[group_index][slots] = True
        saved = [(index, values.copy()) for index, values in groups]
        for _, values in saved:
            values.flags.writeable = False
        for mask in masks:
            mask.flags.writeable = False
        return groups, _BatchRecord(self._owner, saved, masks, changes, replacement)

    def record_completed_batch(self, groups, record):
        if not isinstance(record, _BatchRecord) or record.owner is not self._owner or record.committed:
            raise ValueError('record must be an uncommitted batch from this intervention')
        if (len(groups) != len(record.groups) or any(
                index != expected_index or not np.array_equal(values, expected_values)
                for (index, values), (expected_index, expected_values) in zip(groups, record.groups))):
            raise ValueError('completed groups differ from the intervention result')
        for group_index, (scene_index, indices) in enumerate(record.groups):
            labels = self._labels[scene_index][indices]
            hard_mask = record.hard_masks[group_index]
            hits = np.isin(indices, self._hard[scene_index]) & ~labels
            counts = self._counts[scene_index]
            counts['completed_batch_count'] += 1
            for key, value in dict(sampled_pair_count=len(indices), positive_slots=int(labels.sum()),
                    negative_slots=int((~labels).sum()), hard_source_slots=int(hard_mask.sum()),
                    changed_slots=record.changed_by_group[group_index],
                    final_negative_hard_pool_hits=int(hits.sum()),
                    uniform_negative_hard_pool_hits=int((hits & ~hard_mask).sum())).items():
                counts[key] += value
            for source, mask in zip(SOURCES, (labels, ~labels & ~hard_mask, hard_mask)):
                selected = indices[mask]
                self._seen[scene_index][source].update(map(int, selected))
                _add_scores(self._source_scores[scene_index][source], self._scores[scene_index][selected])
        self._completed += 1
        self._replacement_batches += int(record.replacement)
        record.committed = True

    def audit(self):
        def source_report(indices, source):
            scores = _score_report([self._source_scores[index][source] for index in indices])
            distinct = sum(len(self._seen[index][source]) for index in indices)
            return dict(draw_count=scores['count'], distinct_pair_count=distinct,
                        repeated_draw_count=scores['count']-distinct, ols_scores=scores)

        scenes = []
        for index, scene_id in enumerate(self._scene_ids):
            labels = self._labels[index]
            negative_union = self._seen[index]['negative_uniform'] | self._seen[index]['negative_hard']
            scenes.append(dict(scene_id=scene_id, scene_index=index,
                eligible_positive_pair_count=int(labels.sum()), eligible_negative_pair_count=int((~labels).sum()),
                hard_pool_pair_count=len(self._hard[index]),
                negative_pool_scores=_score_report([self._pool_scores[index]['negative']]),
                hard_pool_scores=_score_report([self._pool_scores[index]['hard']]),
                sources={source: source_report([index], source) for source in SOURCES},
                distinct_negative_pair_count=len(negative_union),
                repeated_negative_draw_count=self._counts[index]['negative_slots']-len(negative_union),
                **self._counts[index]))
        return dict(scope='paired-negative-slot-intervention-v1', split='T_core', config=dict(self.config),
            completed_batch_count=self._completed, hard_batches_with_replacement=self._replacement_batches,
            **{key: sum(row[key] for row in scenes) for key in COUNTERS},
            scene_count=len(scenes), visited_scene_count=sum(row['completed_batch_count'] > 0 for row in scenes),
            distinct_negative_pair_count=sum(row['distinct_negative_pair_count'] for row in scenes),
            repeated_negative_draw_count=sum(row['repeated_negative_draw_count'] for row in scenes),
            eligible_negative_pair_count=sum(row['eligible_negative_pair_count'] for row in scenes),
            hard_pool_pair_count=sum(len(pool) for pool in self._hard),
            negative_pool_scores=_score_report([row['negative'] for row in self._pool_scores]),
            hard_pool_scores=_score_report([row['hard'] for row in self._pool_scores]),
            sources={source: source_report(range(len(scenes)), source) for source in SOURCES},
            score_semantics='native OLS -log1p(rms_haversine_m/1000); larger is more similar',
            score_histogram_edges=['-inf', -5., -3., -2., -1., 0.],
            score_histogram_intervals='left closed, right open; final interval includes zero',
            interpretation='successful optimizer batches only; sources denote draw provenance, not pool membership; '
                'distinctness is scene plus original pair index, independently per source; cross-source overlap allowed',
            pool_rule='T_core negatives only; K=(N+9)//10; score descending then original pair index ascending',
            intervention_rule='first 64 negative slots in original returned batch order; independent RNG; '
                'replacement iff K<64; no exclusion, deduplication or forced value change', scenes=scenes)
