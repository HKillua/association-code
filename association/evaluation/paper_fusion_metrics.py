"""TMTAFF-style four-attribute reporting on already aligned unique targets.

This adds reports only. It does not match targets, convert unverified speed
units, change masks or select models. The printed Eq29 is ambiguous: report
its literal value separately from the conventional residual sample SD.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.physical import _check_valid_states, _mask, _states


ATTRIBUTES = {
    'longitude': (1, 10000., '1e-4 degree'),
    'latitude': (0, 10000., '1e-4 degree'),
    'speed': (2, 1., 'native_speed'),
    'heading': (3, 1., 'degree'),
}
STATISTICS = ('ME', 'MAE', 'absolute_ME', 'residual_SD', 'printed_Eq29_diagnostic')
COUNTS = ('declared_targets', 'declared_query_points', 'available_target_points',
          'evaluated_points', 'missing_prediction_points', 'missing_reference_points',
          'targets_with_reference', 'targets_with_evaluation', 'targets_with_two_points')


def _speed_units(prediction_unit='native', target_unit='native'):
    factors = {'native': 1., 'kn': 1., 'mps': 3600. / 1852.}
    if prediction_unit not in factors or target_unit not in factors:
        raise ValueError('speed units must be native, kn or mps')
    if (prediction_unit == 'native') != (target_unit == 'native'):
        raise ValueError('both speed units must be explicit, or both native for legacy reporting')
    return dict(prediction_unit=prediction_unit, target_unit=target_unit,
                report_unit='native_speed' if prediction_unit == 'native' else 'kn',
                prediction_factor=factors[prediction_unit], target_factor=factors[target_unit])


def _metadata(speed_units=None):
    speed_units = _speed_units() if speed_units is None else speed_units
    return dict(
        scope='unique declared target rows; no duplicate prediction treated as a target',
        weighting='per-target time statistic, equal targets within each scene, equal scenes',
        error_definition='truth minus prediction; longitude/heading use raw degree differences',
        ME_interpretation='signed bias, not MAE; proximity to zero matters, not numerically smaller',
        residual_SD_definition='std(truth-prediction, ddof=1) within each target',
        printed_Eq29_definition='sqrt(sum((truth-ME)**2)/(n-1)); diagnostic only',
        author_ESD_implementation_verified=False,
        speed_unit_status=('native_speed; conversion to paper kn has not been established'
                           if speed_units['report_unit'] == 'native_speed' else
                           'converted separately to kn using caller-declared prediction/target units'),
        speed_units=speed_units,
        speed_unit_basis='caller declaration; this function does not establish the original data units',
        source_speed_units_independently_verified=False,
        conditional_scope='means over defined targets/scenes; read their counts and coverage',
        strict_scope='null unless all declared target/query references and predictions exist; SD needs >=2 points')


def paper_fusion_scene(prediction, target, target_valid, *, prediction_valid,
                       prediction_speed_unit='native', target_speed_unit='native'):
    """Report [T,Q,4] unique targets, preserving unavailable targets and points."""
    speed_units = _speed_units(prediction_speed_unit, target_speed_unit)
    prediction, target = _states(prediction, 'prediction'), _states(target, 'target')
    if prediction.shape != target.shape:
        raise ValueError('paper fusion arrays must have identical shapes')
    tv = _mask(target_valid, target.shape, 'target_valid')
    pv = _mask(prediction_valid, target.shape, 'prediction_valid')
    _check_valid_states(target, tv, 'target')
    _check_valid_states(prediction, pv, 'prediction')
    result = dict(metadata=_metadata(speed_units), attributes={})
    targets, queries = target.shape[:2]
    for name, (column, scale, unit) in ATTRIBUTES.items():
        prediction_scale = target_scale = scale
        if name == 'speed':
            unit = speed_units['report_unit']
            prediction_scale *= speed_units['prediction_factor']
            target_scale *= speed_units['target_factor']
        available = tv[..., column]
        mask = available & pv[..., column]
        n = mask.sum(1)
        values = {key: [] for key in STATISTICS}
        for row in np.flatnonzero(n):
            truth = target[row, mask[row], column] * target_scale
            residual = truth - prediction[row, mask[row], column] * prediction_scale
            bias = float(residual.mean())
            values['ME'].append(bias)
            values['MAE'].append(float(np.abs(residual).mean()))
            values['absolute_ME'].append(abs(bias))
            if len(residual) >= 2:
                values['residual_SD'].append(float(np.std(residual, ddof=1)))
                values['printed_Eq29_diagnostic'].append(float(np.sqrt(
                    np.sum((truth - bias)**2) / (len(residual) - 1))))
        means = {key: float(np.mean(value)) if value else None for key, value in values.items()}
        defined = {key: len(value) for key, value in values.items()}
        complete = bool(targets and queries and available.all() and mask.all())
        result['attributes'][name] = dict(
            unit=unit, paper_unit='kn' if name == 'speed' else unit,
            paper_unit_verified=name != 'speed' or unit == 'kn',
            declared_targets=targets, declared_query_points=targets*queries,
            available_target_points=int(available.sum()), evaluated_points=int(mask.sum()),
            missing_prediction_points=int((available & ~pv[..., column]).sum()),
            missing_reference_points=int((~available).sum()),
            targets_with_reference=int(available.any(1).sum()),
            targets_with_evaluation=int((n > 0).sum()), targets_with_two_points=int((n > 1).sum()),
            coverage=float(mask.sum()/available.sum()) if available.any() else None,
            target_mean=means, defined_targets=defined,
            strict_complete_target_mean={key: means[key] if complete and defined[key] == targets else None
                                         for key in STATISTICS})
    return result


def aggregate_paper_fusion(scenes):
    """Scene-equal means of target-equal reports; never pool targets or points."""
    scenes = list(scenes)
    speed_units = (scenes[0]['metadata'].get('speed_units', _speed_units())
                   if scenes else _speed_units())
    if any(scene['metadata'].get('speed_units', _speed_units()) != speed_units for scene in scenes):
        raise ValueError('cannot combine different prediction/target speed-unit declarations')
    result = dict(metadata=_metadata(speed_units), scene_count=len(scenes), attributes={})
    for name in ATTRIBUTES:
        rows = [scene['attributes'][name] for scene in scenes]
        unit = speed_units['report_unit'] if name == 'speed' else ATTRIBUTES[name][2]
        if any(row['unit'] != unit for row in rows):
            raise ValueError('cannot combine different paper-report attribute units')
        row = {key: sum(value[key] for value in rows) for key in COUNTS}
        row.update(unit=unit, paper_unit='kn' if name == 'speed' else unit,
                   paper_unit_verified=name != 'speed' or unit == 'kn',
                   coverage=row['evaluated_points']/row['available_target_points']
                            if row['available_target_points'] else None,
                   scene_equal_target_mean={}, defined_scenes={}, defined_targets={},
                   strict_complete_population={})
        for key in STATISTICS:
            values = [value['target_mean'][key] for value in rows if value['target_mean'][key] is not None]
            row['scene_equal_target_mean'][key] = float(np.mean(values)) if values else None
            row['defined_scenes'][key] = len(values)
            row['defined_targets'][key] = sum(value['defined_targets'][key] for value in rows)
            strict = [value['strict_complete_target_mean'][key] for value in rows]
            row['strict_complete_population'][key] = float(np.mean(strict)) if strict and all(
                value is not None for value in strict) else None
        result['attributes'][name] = row
    return result
