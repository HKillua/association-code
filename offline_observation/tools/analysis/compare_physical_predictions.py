"""Compare completed development predictions on their saved common support only.

No inference, matching, threshold selection or test-set access. Each side's
original full-population coverage, including duplicate predictions, is retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.evaluation.tmtaff import aggregate_fusion_reports
from association.evaluation.scene import _regions
from association.evaluation.artifacts import load_scene_prediction


def _read(path):
    return json.loads(Path(path).read_text())


def _completed(directory):
    directory = Path(directory).resolve()
    index, run = _read(directory/'index.json'), _read(directory.parent/'run.json')
    if (index.get('format') != 'scene_physical_predictions_v1'
            or index.get('test_opened') is not False or run.get('status') != 'COMPLETED'
            or run.get('test_opened') is not False):
        raise ValueError('only completed development runs with prediction artifacts are accepted')
    if directory.name == 'predictions':
        evaluation = _read(directory.parent/'summary.json')['evaluation']
    elif directory.name.endswith('-predictions'):
        route = directory.name.removesuffix('-predictions')
        if route not in run.get('completed_routes', []):
            raise ValueError('candidate route did not complete')
        evaluation = _read(directory.parent/(route+'-evaluation.json'))
    else:
        raise ValueError('use the original predictions or <route>-predictions directory')
    entries = index['scenes']
    ids = [x['scene_id'] for x in entries]
    if (not ids or len(set(ids)) != len(ids)
            or [x['scene_index'] for x in entries] != list(range(len(entries)))
            or ids != [x['scene_id'] for x in evaluation['per_scene']]):
        raise ValueError('prediction index must cover the entire completed V evaluation in order')
    paths = []
    for entry in entries:
        name = entry['metadata']
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('scene metadata must remain inside its artifact directory')
        paths.append(directory/name)
    return entries, paths


def _equal(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' and b.dtype.kind in 'fc' else np.array_equal(a, b)


def _validate_pair(left, right, entry_left, entry_right, method):
    for item, entry in ((left, entry_left), (right, entry_right)):
        if (item['scene']['split'] != 'V_select' or item['metadata']['scene_id'] != entry['scene_id']
                or item['metadata']['scene_index'] != entry['scene_index']):
            raise ValueError('artifact must describe its indexed V_select scene')
        required = {'grid', 'values', 'observed', 'pairs', 'labels', 'intervals', 'node_ids', 'identities', 'sources'}
        if item['scene']['input_adapter'] == 'native_time_v1':
            required |= {'native_values', 'native_times', 'native_observed'}
        if not required.issubset(item['scene']):
            raise ValueError('artifact must preserve complete original input and candidate evidence')
    if set(left['scene']) != set(right['scene']):
        raise ValueError('scene/input inventories differ')
    for key in left['scene']:
        if not _equal(left['scene'][key], right['scene'][key]):
            raise ValueError('scene/input population differs: '+key)
    if method not in left['methods'] or method not in right['methods']:
        raise ValueError('both artifacts must contain the requested method')
    a, b = left['methods'][method], right['methods'][method]
    for key in ('target_ids', 'split', 'grouping_kind', 'correspondence_policy'):
        if not _equal(a[key], b[key]):
            raise ValueError('method population/protocol differs: '+key)
    if a['split'] != 'V_select':
        raise ValueError('only V_select physical comparisons are permitted')
    n = len(a['target_ids'])
    if any(x['aligned_truth_valid'][n:].any() for x in (a, b)):
        raise ValueError('duplicate prediction rows must not add assigned truth')
    for key in ('aligned_truth', 'aligned_truth_valid', 'aligned_observed'):
        if not _equal(a[key][:n], b[key][:n]):
            raise ValueError('truth/region population differs: '+key)
    return a, b, n


def compare(left_directory, right_directory, output, *, method='predicted'):
    output = Path(output)
    if output.exists():
        raise FileExistsError('comparison output already exists')
    le, lp = _completed(left_directory)
    re, rp = _completed(right_directory)
    if [x['scene_id'] for x in le] != [x['scene_id'] for x in re]:
        raise ValueError('complete validation scene populations differ')
    reports = {kind: {side: [] for side in ('left', 'right')} for kind in ('original', 'common')}
    scenes, improved, worsened = [], [], []
    for i, (left_path, right_path) in enumerate(zip(lp, rp)):
        left, right = load_scene_prediction(left_path), load_scene_prediction(right_path)
        a, b, n = _validate_pair(left, right, le[i], re[i], method)
        common = a['aligned_prediction_valid'][:n] & b['aligned_prediction_valid'][:n] & a['aligned_truth_valid'][:n]
        row = dict(scene_id=le[i]['scene_id'], target_count=n, original={}, common={})
        for side, artifact in (('left', a), ('right', b)):
            for kind in ('original', 'common'):
                limit = len(artifact['aligned_prediction']) if kind == 'original' else n
                valid = artifact['aligned_prediction_valid'] if kind == 'original' else common
                metrics = _regions(artifact['aligned_prediction'][:limit], artifact['aligned_truth'][:limit],
                    artifact['aligned_truth_valid'][:limit], valid, artifact['aligned_observed'][:limit])
                row[kind][side] = metrics
                reports[kind][side].append({'full_target_metrics': metrics})
        for target in range(n):
            errors = []
            for artifact in (a, b):
                metrics = _regions(artifact['aligned_prediction'][target:target+1],
                    artifact['aligned_truth'][target:target+1], artifact['aligned_truth_valid'][target:target+1],
                    common[target:target+1], artifact['aligned_observed'][target:target+1])
                errors.append(metrics['all']['position'])
            if errors[0]['rmse'] is None:
                continue
            delta = errors[1]['rmse'] - errors[0]['rmse']
            case = dict(scene_id=row['scene_id'], target_id=str(a['target_ids'][target]), target_row=target,
                right_minus_left_rmse_metres=delta, left_rmse_metres=errors[0]['rmse'],
                right_rmse_metres=errors[1]['rmse'], common_position_points=errors[0]['intersection_count'],
                left_metadata=str(left_path), right_metadata=str(right_path), method=method,
                left_npz=str(left_path.parent/left['metadata']['npz']),
                right_npz=str(right_path.parent/right['metadata']['npz']),
                array_fields=['aligned_prediction', 'aligned_prediction_valid', 'aligned_truth', 'aligned_truth_valid'])
            if delta < 0:
                improved.append(case)
            elif delta > 0:
                worsened.append(case)
        scenes.append(row)
    result = dict(status='COMPLETED', scope='saved complete V_select predictions; common-support diagnostic only',
        method=method, left=str(Path(left_directory).resolve()), right=str(Path(right_directory).resolve()),
        scene_count=len(scenes), target_count=sum(r['target_count'] for r in scenes), test_opened=False,
        population_policy='all declared targets retained in original order; existing correspondences unchanged',
        common_policy='per-feature intersection of both predictions and reference; duplicate rows have no common assigned truth',
        interpretation='common errors do not replace original full-target coverage, missed targets or extra predictions',
        original={s: aggregate_fusion_reports(reports['original'][s]) for s in ('left', 'right')},
        common={s: aggregate_fusion_reports(reports['common'][s]) for s in ('left', 'right')}, per_scene=scenes,
        case_policy='at most five largest improvements and worsenings in all-region target position RMSE on common support; descriptive, not selected evaluation population',
        improved=sorted(improved, key=lambda x: x['right_minus_left_rmse_metres'])[:5],
        worsened=sorted(worsened, key=lambda x: -x['right_minus_left_rmse_metres'])[:5])
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    output.mkdir(parents=True, exist_ok=False)
    (output/'comparison.json').write_text(payload)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('left', 'right', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--method', default='predicted', help='same saved method on both sides; never mix predicted/oracle')
    args = parser.parse_args(argv)
    result = compare(args.left, args.right, args.output, method=args.method)
    print(json.dumps({k: result[k] for k in ('status', 'method', 'scene_count', 'target_count')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
