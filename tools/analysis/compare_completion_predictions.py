"""Compare complete saved teacher predictions, without inference or selection.

Usage: --left TEACHER_RUN --right TEACHER_RUN --output comparison.json
All input fragments, including repeated identities and unavailable references,
remain in the reported population. This is a development comparison, not a test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.physical import evaluate_aligned_fusion
from association.evaluation.scene import INPUT_DISTANCE_REGIONS

REGIONS = ('all', 'inside_support', 'before_support', 'after_support',
           'exact_observation', 'inside_nonobserved')
COMMON = ('scene_id', 'node_ids', 'identities', 'sources', 'query_seconds',
          'target', 'target_valid', 'nearest_seconds')


def _read(path):
    return json.loads(Path(path).read_text())


def _equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' and b.dtype.kind in 'fc' else np.array_equal(a, b)


def _completed(directory):
    directory = Path(directory).resolve()
    run = _read(directory/'run.json')
    report = _read(directory/'completion-evaluation.json')
    entries = _read(directory/'completion-predictions/index.json')
    if (run.get('status') != 'COMPLETED' or run.get('phase') != 'teacher'
            or run.get('test_opened') is not False or report.get('test_opened') is not False):
        raise ValueError('only completed teacher development runs are accepted')
    planned = run['training']['teacher_updates']
    teacher = run['phases']['teacher']
    if (teacher.get('status') != 'COMPLETED' or teacher.get('completed_updates') != planned
            or run.get('completed_updates') != planned
            or run['phases']['completion_evaluation'].get('status') != 'COMPLETED'):
        raise ValueError('teacher updates and completion evaluation must be complete')
    contract = run['scene_contract'][1]
    ids = [row['scene_id'] for row in contract]
    if (not ids or len(set(ids)) != len(ids) or any(row['split'] != 'V_select' for row in contract)
            or not isinstance(entries, list) or [row['scene_id'] for row in entries] != ids
            or [row['scene_id'] for row in report['per_scene']] != ids
            or report['scene_count'] != len(ids)):
        raise ValueError('prediction index must cover every declared V scene in order')
    if (report['fragment_count'] != sum(len(row['node_ids']) for row in contract)
            or report['completion_forward_count'] != report['fragment_count']):
        raise ValueError('completion report must retain the full fragment population')
    arrays = []
    for entry, expected, summary in zip(entries, contract, report['per_scene']):
        name = entry['path']
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('prediction paths must be filenames inside completion-predictions')
        with np.load(directory/'completion-predictions'/name, allow_pickle=False) as saved:
            item = {key: saved[key] for key in saved.files}
        required = {*COMMON, 'prediction', 'position_base', *('region_'+key for key in REGIONS)}
        if not required.issubset(item) or {k for k in item if k.startswith('region_')} != {'region_'+k for k in REGIONS}:
            raise ValueError('completion artifact is missing population, prediction or region arrays')
        n, q = len(expected['node_ids']), len(expected['grid'])
        if (item['scene_id'].shape != () or str(item['scene_id']) != entry['scene_id']
                or entry['fragment_count'] != n or summary['fragment_count'] != n
                or any(item[key].shape != (n,) or not _equal(item[key], expected[key])
                       for key in ('node_ids', 'identities', 'sources'))
                or len(set(map(str, item['node_ids']))) != n
                or not _equal(item['query_seconds'], expected['grid'])
                or item['query_seconds'].shape != (q,) or not np.isfinite(item['query_seconds']).all()
                or (np.diff(item['query_seconds']) <= 0).any()):
            raise ValueError('artifact fragment identifiers or query grid differ from the V contract')
        if any(item[key].shape != (n, q, 4) for key in ('prediction', 'target', 'target_valid', 'position_base')):
            raise ValueError('completion states and masks must have shape [N,Q,4]')
        if item['target_valid'].dtype != np.bool_:
            raise ValueError('target_valid must be a per-attribute boolean mask')
        if any(item['region_'+key].shape != (n, q) or item['region_'+key].dtype != np.bool_ for key in REGIONS):
            raise ValueError('regions must be boolean [N,Q] arrays')
        nearest = item['nearest_seconds']
        if nearest.shape != (n, q) or np.isnan(nearest).any() or (nearest < 0).any():
            raise ValueError('nearest observation distances must be nonnegative seconds or infinity')
        inside, before, after = (item['region_'+key] for key in ('inside_support', 'before_support', 'after_support'))
        exact = item['region_exact_observation']
        if (not item['region_all'].all() or not ((inside.astype(int)+before+after) == 1).all()
                or not _equal(exact, nearest == 0) or (exact & ~inside).any()
                or not _equal(item['region_inside_nonobserved'], inside & ~exact)):
            raise ValueError('completion regions do not partition the full query population')
        # Reuse physical validation, including all predictions even without truth.
        evaluate_aligned_fusion(item['prediction'], item['target'], item['target_valid'],
            prediction_valid=np.ones((n, q, 4), bool), region_mask=item['region_all'])
        base_valid = np.zeros((n, q, 4), bool)
        base_valid[..., :2] = True
        evaluate_aligned_fusion(item['position_base'], item['target'], item['target_valid'],
            prediction_valid=base_valid, region_mask=item['region_all'])
        if not np.isnan(item['position_base'][..., 2:]).all():
            raise ValueError('position_base has no speed or heading prediction')
        arrays.append(item)
    return directory, run, report, arrays


def _speed_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(count=len(values), mean=float(values.mean()) if len(values) else None,
                std=float(values.std()) if len(values) else None)


def _region_metrics(data, mask):
    prediction, target, valid = (data[key] for key in ('prediction', 'target', 'target_valid'))
    physical = evaluate_aligned_fusion(prediction, target, valid,
        prediction_valid=np.ones(prediction.shape, bool), region_mask=mask)
    speed_valid = mask & valid[..., 2]
    p = torch.as_tensor(prediction[..., 2][speed_valid], dtype=torch.float32)
    t = torch.as_tensor(target[..., 2][speed_valid], dtype=torch.float32)
    raw_log = float((torch.log(p+1e-6)-torch.log(t+1e-6)).square().mean()) if len(p) else None
    if raw_log is not None and not np.isfinite(raw_log):
        raise ValueError('raw FP32 log-speed loss is nonfinite')
    return {**{key: physical[key] for key in ('position', 'speed', 'heading')},
        'raw_log_speed_mse': dict(count=len(p), mse=raw_log),
        'speed_statistics': dict(prediction_all=_speed_statistics(prediction[..., 2][mask]),
            prediction_evaluable=_speed_statistics(prediction[..., 2][speed_valid]),
            target_evaluable=_speed_statistics(target[..., 2][speed_valid]))}


def _metrics(data, *, position_base=False):
    nearest = data['nearest_seconds']
    distances = (nearest <= 300, (nearest > 300) & (nearest <= 600),
                 (nearest > 600) & (nearest <= 1800),
                 (nearest > 1800) & np.isfinite(nearest), ~np.isfinite(nearest))
    def metric(mask):
        if not position_base:
            return _region_metrics(data, mask)
        available = np.zeros(data['position_base'].shape, bool)
        available[..., :2] = True
        return evaluate_aligned_fusion(data['position_base'], data['target'], data['target_valid'],
            prediction_valid=available, region_mask=mask)['position']
    return dict(regions={key: metric(data['region_'+key]) for key in REGIONS},
        input_distance_metrics={key: metric(mask)
                                for key, mask in zip(INPUT_DISTANCE_REGIONS, distances)})


def _difference(left, right):
    """Report numeric metric differences; units/definitions remain on each side."""
    result = {}
    for key, value in left.items():
        other = right[key]
        if isinstance(value, dict):
            result[key] = _difference(value, other)
        elif value is None or other is None:
            result[key] = None
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = other-value
    return result


def _pair(left, right):
    a, b = _metrics(left), _metrics(right)
    bases = {side: _metrics(data, position_base=True)
             for side, data in (('left', left), ('right', right))}
    def learning_delta(side, learned):
        position = {group: {region: metrics['position'] for region, metrics in reports.items()}
                    for group, reports in learned.items()}
        return _difference(bases[side], position)
    return dict(fragment_count=len(left['node_ids']), query_count=left['prediction'].shape[0]*left['prediction'].shape[1],
                left=a, right=b, right_minus_left=_difference(a, b),
                position_base=bases,
                position_model_minus_own_base=dict(left=learning_delta('left', a),
                                                  right=learning_delta('right', b)))


def compare(left_directory, right_directory, output):
    """Write one JSON on the unchanged population of two saved teacher runs."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('comparison output already exists')
    ld, lr, le, left = _completed(left_directory)
    rd, rr, re, right = _completed(right_directory)
    if lr['scene_contract'][1] != rr['scene_contract'][1]:
        raise ValueError('complete validation scene contracts differ')
    for a, b in zip(left, right):
        for key in (*COMMON, *('region_'+key for key in REGIONS)):
            if not _equal(a[key], b[key]):
                raise ValueError('completion population or reference differs: '+key)
    widths = {a['prediction'].shape[1] for a in left}
    if len(widths) != 1:
        raise ValueError('teacher evaluation must retain one declared query width')
    pooled_keys = ('node_ids', 'identities', 'sources', 'prediction', 'position_base', 'target', 'target_valid',
                   'nearest_seconds', *('region_'+key for key in REGIONS))
    def pool(rows):
        return {key: np.concatenate([row[key] for row in rows], axis=0) for key in pooled_keys}
    lp, rp = pool(left), pool(right)
    sources = sorted(set(map(str, lp['sources'])))
    per_scene = []
    for a, b, summary_left, summary_right in zip(left, right, le['per_scene'], re['per_scene']):
        absent = summary_left['reference_targets_without_input']
        if absent != summary_right['reference_targets_without_input']:
            raise ValueError('reference-only target populations differ')
        per_scene.append(dict(scene_id=str(a['scene_id']), reference_targets_without_input=absent, **_pair(a, b)))
    by_source = {}
    for source in sources:
        selected = lp['sources'] == source
        by_source[source] = _pair({key: value[selected] for key, value in lp.items()},
                                  {key: value[selected] for key, value in rp.items()})
    result = dict(status='COMPLETED', scope='saved complete V_select teacher self predictions only',
        left=str(ld), right=str(rd), scene_count=len(left), fragment_count=len(lp['node_ids']),
        query_count=int(lp['prediction'].shape[0]*lp['prediction'].shape[1]), test_opened=False,
        optimizer_updates=0, inference_calls=0, threshold_reselected=False,
        model_config=dict(left=lr['model_config'], right=rr['model_config']),
        training=dict(left=lr['training'], right=rr['training']),
        population_policy='all input fragments and declared V scenes retained; repeated identities are not merged; '
                          'reference-only identities are listed per scene, not invented as input fragments',
        raw_log_speed_definition='mean((log(prediction+1e-6)-log(target+1e-6))^2) in torch.float32 on each '
                                 'region and speed-valid target intersection; no learned weights/regularizer, '
                                 'cross reconstruction or BCE; not the mixed teacher loss',
        speed_statistics_definition='native units, population std (ddof=0); all emitted predictions and '
                                    'speed-valid target intersection are reported separately',
        difference_definition='right minus left on the same population; descriptive only, no significance or pass threshold',
        position_learning_delta_definition='model position error minus its own saved position-base error, '
                                           'on identical target/region masks; negative error deltas mean improvement. '
                                           'The two arms may use different bases; compare their model errors '
                                           'separately from each arm learning increment.',
        overall=_pair(lp, rp), by_source=by_source, per_scene=per_scene)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(payload)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('left', 'right', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    result = compare(args.left, args.right, args.output)
    print(json.dumps({key: result[key] for key in ('status', 'scene_count', 'fragment_count', 'query_count')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
