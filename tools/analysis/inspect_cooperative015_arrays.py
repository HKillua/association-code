"""One-off CPU inspection of saved 015 physical arrays; no model or ZIP access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.prediction_artifacts import load_scene_prediction
from tools.analysis.compare_physical_predictions import _completed, compare

ARMS = ('plain', 'identity', 'utility', 'learned')
METHODS = ('predicted', 'oracle')


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def speed_statistics(values):
    """Describe already-valid native speed values, retaining invalid-finite counts."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    qs = (0., .01, .05, .25, .5, .75, .95, .99, 1.)
    return dict(valid_mask_count=len(values), finite_count=len(finite),
        nonfinite_count=int((~np.isfinite(values)).sum()),
        quantiles={str(q): float(x) for q, x in zip(qs, np.quantile(finite, qs))} if len(finite) else None,
        mean=float(finite.mean()) if len(finite) else None,
        negative_count=int((finite < 0).sum()),
        near_zero_abs_le_1e_6_count=int((np.abs(finite) <= 1e-6).sum()),
        greater_than_100_count=int((finite > 100).sum()),
        greater_than_1000_count=int((finite > 1000).sum()),
        greater_than_10000_count=int((finite > 10000).sum()))


def _scene_key(scene_id):
    value = str(scene_id)
    return (0, int(value), value) if value.isdecimal() else (1, 0, value)


def select_cases(identity_vs_utility):
    """Use all-scene common-support MAE deltas; no manual favourable cases."""
    rows = []
    for scene in identity_vs_utility['per_scene']:
        left = scene['common']['left']['all']['position']['mae']
        right = scene['common']['right']['all']['position']['mae']
        if left is not None and right is not None:
            rows.append(dict(scene_id=str(scene['scene_id']),
                utility_minus_identity_mae_metres=float(right-left)))
    rows.sort(key=lambda x: (x['utility_minus_identity_mae_metres'], _scene_key(x['scene_id'])))
    selected = []

    def add(item, reason):
        existing = next((x for x in selected if x['scene_id'] == item['scene_id']), None)
        if existing is not None:
            existing['selection_reasons'].append(reason)
        else:
            selected.append(dict(item, selection_reasons=[reason]))

    if rows:
        add(rows[0], 'minimum_delta_maximum_improvement_if_negative')
        add(rows[-1], 'maximum_delta_maximum_worsening_if_positive')
        add(rows[(len(rows)-1)//2], 'lower_median_delta')
    all_ids = [str(x['scene_id']) for x in identity_vs_utility['per_scene']]
    if all_ids:
        smallest = min(all_ids, key=_scene_key)
        item = next((x for x in rows if x['scene_id'] == smallest),
                    dict(scene_id=smallest, utility_minus_identity_mae_metres=None))
        add(item, 'smallest_scene_id_numeric_for_decimal_ids')
    return dict(rule='oracle common all-region position MAE: utility minus identity; sort delta then numeric scene_id; lower median; also smallest scene_id; deduplicate',
                eligible_scene_count=len(rows), all_scene_count=len(all_ids), cases=selected)


def _range(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = x[np.isfinite(x)]
    return dict(count=len(x), finite_count=len(finite), nonfinite_count=len(x)-len(finite),
                minimum=float(finite.min()) if len(finite) else None,
                maximum=float(finite.max()) if len(finite) else None)


def inspect(run_directory, output, *, expected_scenes=64):
    source, output = Path(run_directory).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError('diagnostic output must be outside the preserved source run')
    if output.exists():
        raise FileExistsError('diagnostic output already exists')
    run = json.loads((source/'run.json').read_text())
    if run.get('status') != 'COMPLETED' or run.get('test_opened') is not False:
        raise ValueError('only completed development runs can be inspected')
    inventory = {}
    for arm in ARMS:
        entries, paths = _completed(source/(arm+'-predictions'))
        if len(entries) != expected_scenes:
            raise ValueError('full declared validation scene count differs from expected')
        inventory[arm] = (entries, paths)
    scene_ids = [x['scene_id'] for x in inventory['plain'][0]]
    if any([x['scene_id'] for x in inventory[a][0]] != scene_ids for a in ARMS):
        raise ValueError('four-arm scene order differs')
    output.mkdir(parents=True, exist_ok=False)
    _write(output/'progress.json', dict(status='RUNNING', test_opened=False))
    try:
        comparisons = {}
        for control in ('plain', 'identity', 'learned'):
            for method in METHODS:
                key = control+'-vs-utility-'+method
                result = compare(source/(control+'-predictions'), source/'utility-predictions',
                                 output/key, method=method)
                comparisons[key] = result
        selection = select_cases(comparisons['identity-vs-utility-oracle'])
        stats, support_rows = {}, []
        for arm in ARMS:
            collected = {method: {key: [] for key in ('prediction_valid', 'reference_valid',
                'prediction_on_reference_intersection', 'reference_on_prediction_intersection')}
                for method in METHODS}
            for entry, path in zip(*inventory[arm]):
                saved = load_scene_prediction(path)
                if saved['scene']['split'] != 'V_select':
                    raise ValueError('only saved V_select arrays are allowed')
                for method in METHODS:
                    artifact = saved['methods'][method]
                    prediction, truth = artifact['aligned_prediction'][..., 2], artifact['aligned_truth'][..., 2]
                    pmask, tmask = artifact['aligned_prediction_valid'][..., 2], artifact['aligned_truth_valid'][..., 2]
                    intersection = pmask & tmask
                    collected[method]['prediction_valid'].append(prediction[pmask])
                    collected[method]['reference_valid'].append(truth[tmask])
                    collected[method]['prediction_on_reference_intersection'].append(prediction[intersection])
                    collected[method]['reference_on_prediction_intersection'].append(truth[intersection])
                if arm == 'plain':
                    scene = saved['scene']
                    row = dict(scene_id=entry['scene_id'], query_time_seconds=_range(scene['grid']))
                    if 'native_times' in scene and 'native_observed' in scene:
                        times, mask = scene['native_times'], scene['native_observed']
                        spans = [float(t[m].max()-t[m].min()) for t, m in zip(times, mask) if m.any()]
                        row.update(native_time_seconds=_range(times[mask]),
                                   native_track_span_seconds=_range(spans),
                                   empty_native_track_count=int((~mask.any(axis=1)).sum()))
                    support_rows.append(row)
            stats[arm] = {method: {key: speed_statistics(np.concatenate(parts))
                for key, parts in items.items()} for method, items in collected.items()}
        copied = []
        for case in selection['cases']:
            scene_id = case['scene_id']
            for arm in ARMS:
                entry, metadata_path = next((e, p) for e, p in zip(*inventory[arm]) if str(e['scene_id']) == scene_id)
                # Use fixed numeric scene_index directories; never put user IDs into paths.
                target = output/'cases'/f"{entry['scene_index']:06d}"/arm
                target.mkdir(parents=True, exist_ok=False)
                metadata = json.loads(metadata_path.read_text())
                npz_path = metadata_path.parent/metadata['npz']
                for path in (metadata_path, npz_path):
                    shutil.copyfile(path, target/path.name)
                copied.append(dict(scene_id=scene_id, scene_index=entry['scene_index'], arm=arm,
                    source_metadata=str(metadata_path), source_npz=str(npz_path),
                    copied_metadata=str((target/metadata_path.name).relative_to(output)),
                    copied_npz=str((target/npz_path.name).relative_to(output)),
                    npz_bytes=npz_path.stat().st_size, metadata_bytes=metadata_path.stat().st_size))
        result = dict(status='COMPLETED', source_run=str(source), test_opened=False,
            scope='saved-array CPU diagnosis; no model load, inference, calibration, threshold selection or archive reads',
            scene_count=len(scene_ids), comparison_count=6,
            speed_units='native_speed; fixed counts at 100/1000/10000 are descriptive, not physical plausibility cutoffs',
            speed_population='aligned saved rows, including extra prediction rows; reference-only rows retained; masks unchanged; nonfinite flagged separately',
            speed=stats, support_time_scope='original seconds and native track spans only; prediction artifacts omit time_unit_seconds, so no tau or tau-product values are invented',
            support_times=support_rows, case_selection=selection, copied_cases=copied,
            comparison_files={key:key+'/comparison.json' for key in comparisons})
        _write(output/'inspection.json', result)
        _write(output/'progress.json', dict(status='COMPLETED', comparison_count=6,
               case_scene_count=len(selection['cases']), test_opened=False))
        return result
    except Exception as exc:
        _write(output/'progress.json', dict(status='FAILED', failure_type=type(exc).__name__,
               failure=str(exc), test_opened=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = inspect(args.run, args.output)
    print(json.dumps({k:result[k] for k in ('status', 'scene_count', 'comparison_count', 'test_opened')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
