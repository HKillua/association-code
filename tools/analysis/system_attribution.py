"""054A: replay saved053A NPZs and zero-training full-system counterfactuals."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.system_attribution import (
    BASE, INDEPENDENT, ANCHOR, ATTRIBUTES, analyze_scene, assert_metrics, pool_metrics,
)
from association.utils.runtime import jsonable, write_json


def read(path):
    path = Path(path)
    if not path.exists():
        path = path.with_suffix(path.suffix+'.gz')
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8')) as stream:
        return json.load(stream)


def write_gzip(path, value):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, allow_nan=False)


def aggregate(rows):
    result = dict(methods={}, attribution={}, pairwise={}, coverage_transitions={})
    for name in (BASE, INDEPENDENT, ANCHOR):
        items = [r['methods'][name] for r in rows]
        methods = result['methods'][name] = {
            key: pool_metrics([r[key] for r in items])
            for key in ('fragment', 'emitted_fragment', 'fixed_target_strata')}
        methods['system'] = {
            key: pool_metrics([r['system'][key] for r in items])
            for key in ('full_target_metrics', 'input_support_metrics', 'input_distance_metrics')}
        methods['population'] = {key: sum(r['system'][key] for r in items) for key in (
            'target_identity_count', 'predicted_group_count', 'duplicate_prediction_group_count',
            'anchor_matched_target_count', 'reference_unavailable_target_count')}
        methods['duplicate_outputs'] = {attr: {
            key: sum(r['duplicate_outputs'][attr][key] for r in items)
            for key in ('trajectory_count', 'point_count')} for attr in ATTRIBUTES}
    for attr in ATTRIBUTES:
        items = [r['attribution'][attr] for r in rows]
        pooled = result['attribution'][attr] = {}
        for key in ('primary_counts', 'overlap_counts', 'missing_point_causes'):
            count = Counter()
            for r in items:
                count.update(r[key])
            pooled[key] = dict(count)
        for key in ('registry_target_count', 'available_target_trajectory_count',
                    'covered_target_trajectory_count', 'missing_target_trajectory_count'):
            pooled[key] = sum(r[key] for r in items)
    for pair, first in rows[0]['pairwise'].items():
        result['pairwise'][pair] = dict(methods={name: pool_metrics([
            r['pairwise'][pair]['methods'][name] for r in rows]) for name in first['methods']},
            **{k: sum(r['pairwise'][pair][k] for r in rows) for k in (
                'same_representative_target_count', 'changed_representative_target_count')})
        result['pairwise'][pair]['error_change_second_minus_first'] = {
            region: {attr: {key: sum(r['pairwise'][pair]['error_change_second_minus_first'][region][attr][key]
                                    for r in rows) for key in values}
                     for attr, values in attrs.items()}
            for region, attrs in first['error_change_second_minus_first'].items()}
    for name in (INDEPENDENT, ANCHOR):
        result['coverage_transitions'][name] = {attr: {
            key: sum(r['coverage_transitions'][name][attr][key] for r in rows)
            for key in rows[0]['coverage_transitions'][name][attr]} for attr in ATTRIBUTES}
    pair = '__vs__'.join(sorted((BASE, ANCHOR)))
    result['fixed_group_fusion_effect'] = {attr: dict(
        mean_minus_anchor_mae=(result['pairwise'][pair]['methods'][BASE]['all'][attr]['mae']
                              - result['pairwise'][pair]['methods'][ANCHOR]['all'][attr]['mae'])
            if result['pairwise'][pair]['methods'][BASE]['all'][attr]['mae'] is not None else None,
        common_point_count=result['pairwise'][pair]['methods'][BASE]['all'][attr]['intersection_count'],
        scene_mean_worse_count=sum(
            r['pairwise'][pair]['methods'][BASE]['all'][attr]['mae'] is not None and
            r['pairwise'][pair]['methods'][BASE]['all'][attr]['mae'] >
            r['pairwise'][pair]['methods'][ANCHOR]['all'][attr]['mae'] for r in rows),
        eligible_scene_count=sum(r['pairwise'][pair]['methods'][BASE]['all'][attr]['mae'] is not None for r in rows),
        meaning='positive means ordinary fusion harms accuracy on fixed-group common points; not a coverage score')
        for attr in ATTRIBUTES}
    return result


def load_source(source, manifest):
    """Shared complete053A cache contract; reads metadata and source vocabulary."""
    source = Path(source)
    original, source_run = read(source/'summary.json'), read(source/'run.json')
    ids = original['scene_ids']
    required = [str(x) for x in manifest['splits']['V_select']]
    wp = original.get('ols_working_point', {})
    if (manifest.get('dataset_level') != 'Level1' or manifest.get('split_unit') != 'scene'
            or not ids or len(ids) != len(set(ids)) or set(ids) != set(required)
            or original.get('scene_count') != len(ids) or original.get('status') != 'COMPLETED'
            or original.get('dataset_level') != 'Level1' or original.get('split') != 'V_select'
            or original.get('method_names') != [BASE] or original.get('test_opened') is not False
            or original.get('optimizer_updates') != 0 or source_run.get('status') != 'COMPLETED'
            or source_run.get('test_opened') is not False or source_run.get('optimizer_updates') != 0
            or source_run.get('motion_observation_policy') != 'canonical_all'
            or source_run.get('threshold_reselected') is not False
            or source_run.get('normalizer_refitted') is not False
            or wp.get('motion_observation_policy') != 'canonical_all'
            or wp.get('validation_scene_ids') != required
            or source_run.get('completed_scenes') != len(ids) or source_run.get('total_scenes') != len(ids)):
        raise ValueError('complete053A Level1 V population and zero-training source required')
    if any(not str(sid).isdigit() for sid in ids):
        raise ValueError('scene IDs must be numeric file identifiers')
    expected = {r['scene_id']: r for r in original['per_scene']}
    if len(original['per_scene']) != len(ids) or set(expected) != set(ids):
        raise ValueError('source per-scene report population differs')
    names = {f'states-{sid}.npz' for sid in ids}
    for directory in ('predictions', 'system-predictions'):
        if {p.name for p in (source/directory).glob('*.npz')} != names:
            raise ValueError(f'incomplete or extra cached scenes in {directory}')
    sources = set()
    for sid in ids:
        with np.load(source/'predictions'/f'states-{sid}.npz', allow_pickle=False) as f:
            sources.update(str(x) for x in f['sources'])
    return original, source_run, expected, sorted(sources)


def run(args):
    source, output = Path(args.source_system).resolve(), Path(args.output).resolve()
    if source == output or source in output.parents:
        raise ValueError('054 outputs must not modify the saved053A directory')
    original, source_run, expected, sources = load_source(source, read(args.manifest))
    ids = original['scene_ids']
    output.mkdir(parents=True, exist_ok=False)
    (output/'scenes').mkdir()
    (output/'predictions').mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', scope='054A-cached-system-attribution', dataset_level='Level1',
        split='V_select', source_system=str(source), optimizer_updates=0, gpu_used=False,
        archive_members_read=0, reference_members_read=0, cached_reference_used=True,
        threshold_reselected=False, test_opened=False, completed_scenes=0, total_scenes=len(ids))
    write_json(output/'run.json', state)
    try:
        rows, fragment_count = [], 0
        for sid in ids:
            with np.load(source/'predictions'/f'states-{sid}.npz', allow_pickle=False) as f:
                fragment = {k: f[k] for k in f.files}
            with np.load(source/'system-predictions'/f'states-{sid}.npz', allow_pickle=False) as f:
                saved = {k[len(BASE)+1:]: f[k] for k in f.files if k.startswith(BASE+'_')}
            report, arrays = analyze_scene(fragment, saved, expected[sid], source_names=sorted(sources))
            report['scene_id'] = sid
            fragment_count += len(fragment['node_ids'])
            write_gzip(output/'scenes'/f'scene-{sid}.json.gz', report)
            np.savez_compressed(output/'predictions'/f'states-{sid}.npz', **arrays)
            # Attribution rows and group membership are already persisted per
            # scene; retain only aggregate inputs in memory.
            for attr in ATTRIBUTES:
                report['attribution'][attr].pop('target_rows')
            for method in report['methods'].values():
                method['system'].pop('all_predicted_groups')
            rows.append(report)
            state.update(completed_scenes=len(rows), last_scene_id=sid, elapsed_seconds=time.monotonic()-started)
            write_json(output/'run.json', state)
        if fragment_count != original['fragment_count']:
            raise ValueError('source fragment count changed')
        pooled = aggregate(rows)
        assert_metrics(pooled['methods'][BASE]['system']['full_target_metrics'],
                       {k: original['unique_target'][BASE][k] for k in ('all', 'observed', 'missing')},
                       'source aggregate full target')
        assert_metrics(pooled['methods'][BASE]['fragment'], original['fragment'][BASE], 'source aggregate fragment')
        # One small return artifact; full target tables/arrays stay per scene.
        write_gzip(output/'scene-summary.json.gz', [dict(scene_id=r['scene_id'],
            attribution=r['attribution'], coverage_transitions=r['coverage_transitions'],
            methods={name: dict(all=m['system']['full_target_metrics']['all'],
                predicted_group_count=m['system']['predicted_group_count'],
                duplicate_prediction_group_count=m['system']['duplicate_prediction_group_count'],
                duplicate_outputs=m['duplicate_outputs']) for name, m in r['methods'].items()},
            pairwise={pair: dict(methods={name: m['all'] for name, m in item['methods'].items()},
                error_change_second_minus_first=item['error_change_second_minus_first']['all'])
                for pair, item in r['pairwise'].items()}) for r in rows])
        pooled.update(status='COMPLETED', scope=state['scope'], dataset_level='Level1', split='V_select',
            scene_ids=ids, scene_count=len(ids), fragment_count=fragment_count, all_source_scenes_replayed=True,
            ols_working_point=original['ols_working_point'], source_system=str(source),
            optimizer_updates=0, test_opened=False, threshold_reselected=False,
            definitions=dict(
                correspondence='anchor_identity_first; fixed input order, no geometry rematching or oracle rescue',
                attribution='exclusive primary reasons partition every registry identity per attribute; overlapping flags separate',
                fragment='BASE preserves053A per-node fallback for replay; emitted_fragment is the fair group-output broadcast comparison',
                strata='fixed target true-identity input support union and nearest distance; evaluation only; actual predicted-group support is secondary',
                pairwise='truth-valid AND both predictions valid, target rows only; full truth denominator retained',
                duplicates='extra same-anchor-identity groups; reported separately from reference-invalid prediction points',
                independent='diagnostic only: increased coverage may entail many duplicate outputs; not a complete-system winner',
                inference='054 only copies cached native OLS/group predictions; no refitting, regrouping threshold or model inference'),
            limitations=['repeated development V; no independent test claim',
                         'missing identity representation under this correspondence is not proof no geometric rematching can recover it',
                         'singleton vs grouped anchor can change representative; paired strata report that change',
                         'MAE is conditional on valid points; coverage/duplicates cannot be discarded',
                         'native speed units retained; no unverified unit conversion'])
        write_json(output/'summary.json', pooled)
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return pooled
    except Exception as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-system', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    run(args)
    print(f'COMPLETED: {args.output}/summary.json')


if __name__ == '__main__':
    main()
