"""Shared cache-replay checks and full-population physical metric pooling.

Retained by 060/061/062; the closed 054 attribution workflow lives in Git history.
"""
from __future__ import annotations

import math

import numpy as np

from association.evaluation.physical import evaluate_aligned_fusion

BASE = 'ols_mean_supported'
ATTRIBUTES = {'position': (0, 1), 'speed': (2,), 'heading': (3,)}
COUNT_KEYS = ('available_target_count', 'available_prediction_count', 'intersection_count',
              'missing_prediction_count', 'prediction_without_target_count',
              'available_target_trajectory_count', 'covered_target_trajectory_count',
              'missing_target_trajectory_count')


def _same(actual, expected, name):
    a, b = np.asarray(actual), np.asarray(expected)
    equal = (a.shape == b.shape and
             (np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' and b.dtype.kind in 'fc'
              else np.array_equal(a, b)))
    if not equal:
        raise ValueError(f'cached replay mismatch: {name}')


def assert_metrics(actual, expected, name='metrics'):
    """Compare shared physical metrics, retaining source-only p95 diagnostics."""
    for region, rows in expected.items():
        for attr in ATTRIBUTES:
            for key in (*COUNT_KEYS, 'mae', 'mse', 'rmse', 'coverage', 'trajectory_coverage'):
                if key not in rows[attr]:
                    continue
                a, b = actual[region][attr][key], rows[attr][key]
                if a is None or b is None:
                    equal = a is b
                elif key in COUNT_KEYS:
                    equal = a == b
                else:
                    equal = math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9)
                if not equal:
                    raise ValueError(f'cached replay mismatch: {name}/{region}/{attr}/{key}')


def pool_metrics(scenes):
    """Point-weighted errors and separate equal-scene macros, including nulls."""
    result = {}
    if not scenes:
        raise ValueError('cannot pool an empty scene population')
    if any(set(scene) != set(scenes[0]) for scene in scenes):
        raise ValueError('region sets differ across scenes')
    for region in scenes[0]:
        result[region] = {}
        for attr in ATTRIBUTES:
            rows = [s[region][attr] for s in scenes]
            out = {k: sum(r[k] for r in rows) for k in COUNT_KEYS}
            n = out['intersection_count']
            for k in ('mae', 'mse'):
                out[k] = sum(r[k] * r['intersection_count'] for r in rows
                             if r['intersection_count']) / n if n else None
            out['rmse'] = math.sqrt(out['mse']) if n else None
            for k, num, den in (('coverage', 'intersection_count', 'available_target_count'),
                               ('trajectory_coverage', 'covered_target_trajectory_count',
                                'available_target_trajectory_count')):
                out[k] = out[num] / out[den] if out[den] else None
            for k in ('mae', 'coverage', 'trajectory_coverage'):
                values = [r[k] for r in rows if r[k] is not None]
                out['scene_macro_' + k] = float(np.mean(values)) if values else None
                out['scene_macro_' + k + '_scene_count'] = len(values)
            out['scene_count'] = len(rows)
            out['unit'] = rows[0]['unit']
            result[region][attr] = out
    return result


def _metrics(prediction, truth, truth_valid, valid, regions):
    return {name: evaluate_aligned_fusion(prediction, truth, truth_valid,
                prediction_valid=valid, region_mask=mask) for name, mask in regions.items()}


def _regions(support, nearest, exact=None):
    regions = dict(all=np.ones(support.shape, bool), inside_support=support,
                   outside_support=~support,
                   far_gt_1800s=(nearest > 1800) & np.isfinite(nearest),
                   distance_le_300s=nearest <= 300,
                   distance_gt_300_le_600s=(nearest > 300) & (nearest <= 600),
                   distance_gt_600_le_1800s=(nearest > 600) & (nearest <= 1800),
                   no_input=~np.isfinite(nearest))
    if exact is not None:
        regions.update(exact_observation=exact, not_exact_observation=~exact)
    return regions
