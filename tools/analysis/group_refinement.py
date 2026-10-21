"""055A: one fixed-score clique refinement, cached V-only CPU evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.association import _aggregate, evaluate_scenes
from association.evaluation.group_refinement import reconstruct_graph
from association.evaluation.refinement_diagnostics import (
    METHODS, analyze_refinement, aggregate_refinement,
)
from association.evaluation.system_attribution import ANCHOR, ATTRIBUTES, BASE, INDEPENDENT, assert_metrics
from association.utils.runtime import write_json
from tools.analysis.system_attribution import load_source, read, write_gzip

NODE_BINS = ((0, 32, 'nodes_le_32'), (33, 64, 'nodes_33_64'),
             (65, 128, 'nodes_65_128'), (129, float('inf'), 'nodes_ge_129'))


def assert_same(actual, expected, path='replay'):
    """Compare complete association/count records without ignoring metadata."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f'{path}: fields differ')
        for key in expected:
            assert_same(actual[key], expected[key], path+'.'+key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f'{path}: list population differs')
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_same(a, e, f'{path}[{i}]')
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not np.isclose(actual, expected, rtol=1e-10, atol=1e-9):
            raise ValueError(f'{path}: numeric value differs')
    elif actual != expected or type(actual) is not type(expected):
        raise ValueError(f'{path}: value differs')


def _source054(path, original):
    summary, run = read(path/'summary.json'), read(path/'run.json')
    ids = original['scene_ids']
    if (summary.get('status') != 'COMPLETED' or run.get('status') != 'COMPLETED'
            or summary.get('scene_ids') != ids or summary.get('scene_count') != len(ids)
            or summary.get('fragment_count') != original['fragment_count']
            or summary.get('dataset_level') != 'Level1' or summary.get('split') != 'V_select'
            or summary.get('scope') != '054A-cached-system-attribution'
            or summary.get('all_source_scenes_replayed') is not True
            or run.get('completed_scenes') != len(ids) or run.get('total_scenes') != len(ids)
            or any(r.get('optimizer_updates') != 0 or r.get('test_opened') is not False
                   or r.get('threshold_reselected') is not False for r in (summary, run))):
        raise ValueError('complete054A zero-training V-only source required')
    assert_same(summary['ols_working_point'], original['ols_working_point'], '054 working point')
    rows = read(path/'scene-summary.json')
    by_id = {r['scene_id']: r for r in rows}
    if len(rows) != len(ids) or set(by_id) != set(ids):
        raise ValueError('054 scene population differs')
    # This list is fixed from054 before any055 output is evaluated.
    worst = sorted(ids, key=lambda sid: (
        -by_id[sid]['attribution']['position']['missing_target_trajectory_count'], int(sid)))[:30]
    return summary, by_id, worst


def _replay054(report, previous):
    for name in (BASE, ANCHOR, INDEPENDENT):
        method = report['methods'][name]
        assert_metrics({'all': method['system']['full_target_metrics']['all']},
                       {'all': previous['methods'][name]['all']}, f'054 {name}')
        for key in ('predicted_group_count', 'duplicate_prediction_group_count'):
            assert_same(method['system'][key], previous['methods'][name][key], f'054 {name}.{key}')
        assert_same(method['duplicate_outputs'], previous['methods'][name]['duplicate_outputs'],
                    f'054 {name}.duplicate_outputs')
    for attr in ATTRIBUTES:
        actual = {k: v for k, v in report['attribution'][attr].items() if k != 'target_rows'}
        assert_same(actual, previous['attribution'][attr], '054 attribution.'+attr)


def _association_summary(evaluations, threshold):
    rows = [r['per_scene'][0] for r in evaluations]
    result = dict(_aggregate(rows), threshold=threshold, per_scene=rows)
    for kind in ('itsa', 'mstsa'):
        part = [r[kind]['per_scene'][0] for r in evaluations]
        result[kind] = dict(_aggregate(part), per_scene=part)
    active = sum(r['pair_count'] > 0 for r in rows)
    result['coverage'] = dict(all_scenes=len(rows), eligible_scenes=active,
                              scenes_without_pairs=len(rows)-active)
    result['empty_f1_convention'] = evaluations[0]['empty_f1_convention']
    return result


def _brief(pooled):
    """Small scene-stratum table; full detailed diagnostics remain per scene."""
    return dict(scene_count=pooled['scene_count'], grouping=pooled['grouping'],
        refinement_accounting=pooled['refinement_accounting'],
        methods={name: dict(all=r['system']['full_target_metrics']['all'],
                           population=r['population']) for name, r in pooled['methods'].items()},
        joint_fusion_effect=pooled['joint_four_arm']['fusion_effect'],
        joint_effective_helper_fusion_effect=pooled['joint_effective_helpers']['fusion_effect'])


def run(args):
    source, previous, output = (Path(x).resolve() for x in (
        args.source_system, args.source_attribution, args.output))
    if any(output == p or p in output.parents for p in (source, source.parent/'ols', previous)):
        raise ValueError('055 outputs must not modify053/054 source directories')
    original, _, expected, sources = load_source(source, read(args.manifest))
    prior_summary, prior_rows, worst = _source054(previous, original)
    ids, wp = original['scene_ids'], original['ols_working_point']
    baseline = read(source.parent/'ols'/'baseline.json')
    assert_same(read(source.parent/'ols'/'ols-working-point.json'), wp, '053 working point')
    if (baseline.get('scene_ids') != ids or baseline.get('test_opened') is not False
            or baseline.get('reference_read') is not False or baseline.get('optimizer_updates') != 0
            or baseline['selection']['threshold'] != wp['threshold']
            or baseline['evaluation']['threshold'] != wp['threshold']):
        raise ValueError('original pureOLS baseline population/threshold differs')
    edge_rows = {r['scene_id']: r for r in baseline['evaluation']['per_scene']}
    if len(baseline['evaluation']['per_scene']) != len(ids) or set(edge_rows) != set(ids):
        raise ValueError('original edge report population differs')
    output.mkdir(parents=True, exist_ok=False)
    (output/'scenes').mkdir()
    (output/'predictions').mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', scope='055A-cached-group-refinement', dataset_level='Level1',
        split='V_select', source_system=str(source), source_attribution=str(previous),
        optimizer_updates=0, gpu_used=False, archive_members_read=0, reference_members_read=0,
        cached_reference_used=True, ols_refitted=False, threshold_reselected=False,
        test_opened=False, completed_scenes=0, total_scenes=len(ids), cpu_threads=args.cpu_threads)
    write_json(output/'run.json', state)
    try:
        rows, edge_reports, fragment_count = [], [], 0
        for sid in ids:
            with np.load(source/'predictions'/f'states-{sid}.npz', allow_pickle=False) as f:
                fragment = {k: f[k] for k in f.files}
            with np.load(source/'system-predictions'/f'states-{sid}.npz', allow_pickle=False) as f:
                saved = {k[len(BASE)+1:]: f[k] for k in f.files if k.startswith(BASE+'_')}
            pairs, scores, old, new = reconstruct_graph(fragment, wp['threshold'])
            # Truth is introduced only after the inference-only partition call.
            labels = fragment['identities'][pairs[:, 0]] == fragment['identities'][pairs[:, 1]]
            edge = evaluate_scenes([dict(scene_id=sid, component_id=edge_rows[sid]['component_id'],
                pairs=pairs, scores=scores, labels=labels, sources=fragment['sources'])], wp['threshold'])
            assert_same(edge['per_scene'][0], edge_rows[sid], 'original association scene '+sid)
            for kind in ('itsa', 'mstsa'):
                old_rows = baseline['evaluation'][kind]['per_scene']
                matched = [r for r in old_rows if r['scene_id'] == sid]
                if len(matched) != 1:
                    raise ValueError('original '+kind+' scene population differs')
                assert_same(edge[kind]['per_scene'][0], matched[0], kind+' scene '+sid)
            report, arrays = analyze_refinement(fragment, saved, old, new, expected[sid], sources)
            _replay054(report, prior_rows[sid])
            n = len(fragment['node_ids'])
            report.update(scene_id=sid, fragment_count=n, original_association=edge,
                          source054_replayed=True)
            arrays.update(ols_pairs=pairs, ols_scores=scores, ols_threshold=np.asarray(
                np.nan if wp['threshold'] is None else wp['threshold']),
                ols_threshold_all_reject=np.asarray(wp['threshold'] is None))
            write_gzip(output/'scenes'/f'scene-{sid}.json.gz', report)
            np.savez_compressed(output/'predictions'/f'states-{sid}.npz', **arrays)
            for attr in ATTRIBUTES:
                report['attribution'][attr].pop('target_rows')
            for method in report['methods'].values():
                method['system'].pop('all_predicted_groups')
            rows.append(report)
            edge_reports.append(edge)
            fragment_count += n
            state.update(completed_scenes=len(rows), last_scene_id=sid, elapsed_seconds=time.monotonic()-started)
            write_json(output/'run.json', state)
        if fragment_count != original['fragment_count']:
            raise ValueError('source fragment population changed')
        association = _association_summary(edge_reports, wp['threshold'])
        assert_same(association, baseline['evaluation'], 'original association aggregate')
        pooled = aggregate_refinement(rows)
        for name in (BASE, ANCHOR, INDEPENDENT):
            assert_metrics(pooled['methods'][name]['system']['full_target_metrics'],
                           prior_summary['methods'][name]['system']['full_target_metrics'], '054 aggregate '+name)
        strata = {}
        for low, high, label in NODE_BINS:
            selected = [r for r in rows if low <= r['fragment_count'] <= high]
            strata[label] = dict(scene_ids=[r['scene_id'] for r in selected],
                results=_brief(aggregate_refinement(selected)) if selected else None)
        selected = [r for r in rows if r['scene_id'] in worst]
        strata['fixed054_worst30'] = dict(scene_ids=worst,
            selection='054 missing position trajectories descending, numeric scene ID ascending; exploratory',
            results=_brief(aggregate_refinement(selected)))
        write_gzip(output/'scene-summary.json.gz', [dict(scene_id=r['scene_id'],
            fragment_count=r['fragment_count'], grouping=r['grouping'],
            refinement_accounting=r['refinement_accounting'],
            methods={name: dict(all=m['system']['full_target_metrics']['all'],
                predicted_group_count=m['system']['predicted_group_count'],
                duplicate_prediction_group_count=m['system']['duplicate_prediction_group_count'])
                for name, m in r['methods'].items()},
            joint_fusion_effect=r['joint_four_arm']['fusion_effect'],
            joint_effective_helper_fusion_effect=r['joint_effective_helpers']['fusion_effect']) for r in rows])
        pooled.update(status='COMPLETED', scope=state['scope'], dataset_level='Level1', split='V_select',
            scene_ids=ids, fragment_count=fragment_count, method_names=list(METHODS),
            all_source_scenes_replayed=True, original_association=association, scene_strata=strata,
            ols_working_point=wp, optimizer_updates=0, gpu_used=False, threshold_reselected=False,
            test_opened=False, archive_members_read=0, reference_members_read=0, cached_reference_used=True,
            definitions=dict(
                grouping='within-parent first-fit all-pairs clique refinement; fixed node order; global anchor sort',
                replay='original partition, system results and overall/ITSA/MSTSA association statistics; original edge score vectors were not saved',
                graph='regenerated exact original Q64 float64 RMS/log1p implementation; saved pairs/scores; no threshold sweep',
                population='full registry; anchor_identity_first; identity/purity only after predictions for evaluation',
                duplicates='G-U, U includes reference-unavailable identities; deltaG=deltaU+deltaD',
                capacity='sum_s min(G_s,T_s)/sum_s T_s for each attribute; not pooled min, not achievable guarantee',
                common='pairwise and four-arm truth-valid AND prediction-valid points, full denominator retained',
                speed='native dataset units; not converted',
                empty='undefined errors/precision/recall are null; no candidate pair F1 is null; eligible empty-positive F1 is1'),
            limitations=['development V only; no independent test or innovation claim',
                'splitting increases outputs and cannot reduce duplicate groups under this correspondence',
                'changed representative and newly covered targets alter conditional MAE; use paired cohorts',
                'helper-active means supported contribution, not proven causal usefulness',
                'fixed worst30 is known post-hoc diagnostic subset; all V scenes remain primary',
                'no automatic next pilot, GPU budget, threshold or method search'])
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
    parser.add_argument('--source-attribution', required=True, type=Path)
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
