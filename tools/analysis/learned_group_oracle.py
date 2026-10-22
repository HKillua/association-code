"""061: fixed056 learned groups, three cached controls and a restricted selector."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.learned_group_oracle import analyze_scene, aggregate, aggregation_arrays
from association.evaluation.system_attribution import assert_metrics
from association.utils.runtime import write_json
from tools.analysis.compare_association_duration import ARMS, _load_arrays, _load_metadata, _replay
from association.utils.runtime import read_json_or_gzip as read, write_json_gzip as write_gzip

SOURCE_RUN = 'LEVEL1-ASSOCIATION-20261022-056-attempt-1'
CONFIG = Path(__file__).resolve().parents[2]/'configs/evidence-association-level1.json'


def _bind_source(source):
    """Bind this diagnostic to056, not whichever final1000 looks best later."""
    if source.name != 'system' or source.parent.name != SOURCE_RUN:
        raise ValueError('061 requires the original056 attempt-1/system cache')
    system = read(source/'run.json')
    inputs = system.get('inputs', {})
    training = source.parent/'association'
    for key, filename in (
            ('association_run', 'run.json'), ('association_summary', 'summary.json'),
            ('association_data_audit', 'data_audit.json'), ('association_checkpoint', 'final.pt'),
            ('association_scores', 'final-scores.npz')):
        value = inputs.get(key)
        if not isinstance(value, str) or Path(value).resolve() != training/filename:
            raise ValueError('056 source must bind the original sibling association/' + filename)
    record = read(training/'run.json')
    expected = read(CONFIG)
    if (record.get('config') != expected or record.get('kind') != expected['scope']
            or any(record.get(key) is not False for key in ('synthetic', 'reference_read', 'test_opened'))):
        raise ValueError('056 original training configuration and data-use contract required')
    # Actual/configured1000 updates, COMPLETED, final endpoint and full V cache
    # population are checked by the shared060 loader. No checkpoint is reloaded.
    return record


def run(args):
    source, output = Path(args.source_system).resolve(), Path(args.output).resolve()
    protected = source.parent
    if output == protected or output in protected.parents or protected in output.parents:
        raise ValueError('061 output must be separate from the entire immutable056 run')
    if type(args.cpu_threads) is not int or args.cpu_threads < 1:
        raise ValueError('cpu_threads must be a positive integer')
    torch.set_num_threads(args.cpu_threads)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', scope='061-frozen056-learned-group-oracle', dataset_level='Level1',
        split='V_select', source_system=str(source), source_task=SOURCE_RUN,
        association_endpoint='final', association_endpoint_step=1000, cpu_threads=args.cpu_threads,
        optimizer_updates=0, gpu_used=False, archive_members_read=0, reference_members_read=0,
        cached_reference_used=True, ols_refitted=False, threshold_reselected=False,
        normalizer_refitted=False, scores_recomputed=False, groups_recomputed=False,
        test_opened=False, completed_scenes=0, total_scenes=None)
    write_json(output/'run.json', state)
    try:
        _bind_source(source)
        manifest = read(args.manifest)
        ids = manifest.get('splits', {}).get('V_select', [])
        if (manifest.get('dataset_level') != 'Level1' or manifest.get('split_unit') != 'scene'
                or not ids or len(ids) != len(set(ids))
                or any(not isinstance(sid, str) or not sid.isdigit() for sid in ids)):
            raise ValueError('complete Level1 scene-level V manifest required')
        metadata = _load_metadata(source, manifest, 1000)
        state['total_scenes'] = len(ids)
        write_json(output/'run.json', state)
        (output/'scenes').mkdir()
        (output/'predictions').mkdir()
        rows, point_arrays, fragment_count = [], [], 0
        for sid in ids:
            fragment, saved = _load_arrays(source, sid)
            replay = _replay(fragment, saved, metadata['expected'][sid])
            row, arrays = analyze_scene(fragment, replay)
            row['scene_id'] = sid
            fragment_count += row['fragment_count']
            write_gzip(output/'scenes'/f'scene-{sid}.json.gz', row)
            np.savez_compressed(output/'predictions'/f'states-{sid}.npz', **arrays)
            rows.append(row)
            point_arrays.append(aggregation_arrays(arrays))
            state.update(completed_scenes=len(rows), last_scene_id=sid, elapsed_seconds=time.monotonic()-started)
            write_json(output/'run.json', state)
        if fragment_count != metadata['fragment_count']:
            raise ValueError('056 fragment population changed')
        result = aggregate(rows, point_arrays)
        for arm in ARMS:
            assert_metrics(result['methods'][arm]['full_target_metrics'],
                           metadata['pooled']['learned_assoc_' + arm], '056 pooled ' + arm)
        result.update(status='COMPLETED', scope=state['scope'], dataset_level='Level1', split='V_select',
            scene_ids=ids, completed_scenes=len(ids), fragment_count=fragment_count,
            method_names=list(result['methods']), source_system=str(source), source_task=SOURCE_RUN,
            association_endpoint='final', association_endpoint_step=1000,
            association_threshold=metadata['association_threshold'], ols_working_point=metadata['ols_working_point'],
            all_source_scenes_replayed=True, optimizer_updates=0, gpu_used=False,
            archive_members_read=0, reference_members_read=0, cached_reference_used=True,
            test_opened=False, threshold_reselected=False, ols_refitted=False, normalizer_refitted=False,
            scores_recomputed=False, groups_recomputed=False,
            definitions=dict(
                source='original056 final1000 learned groups; original OLS controls replayed only as source checks',
                population='fixed full target registry, first representative, duplicate rows and original masks',
                selector='strict lower-error Anchor/Mean choice; entire position pair, scalar speed/circular heading; tie/invalid truth uses Anchor',
                opportunity='candidate physical difference greater than1e-9 in each attribute unit; not a practical-effect threshold or a truth-derived label',
                selection_denominators='reference-and-candidate-valid group points including duplicates; unique retained target points; nontrivial subset of each',
                numerical_rules='exact/equal_nan cached arrays; strict < selection; absolute1e-9/relative0 error assertion; +/-1e-9 descriptive changes',
                quantiles='unique-target common-valid point errors; numpy linear quantile; empty=null; per-scene quantiles separate',
                visible_strata='anchor own support/distance and helpers in the frozen predicted group',
                identity_strata='true-identity input-support union retained only for posthoc comparison'),
            limitations=['two fixed candidates only, not a bound on continuous fusion or other estimators',
                'oracle uses cached reference and cannot be deployed; attribute switching need not be dynamically continuous',
                'oracle cannot recover targets missing from the original056 output correspondence',
                'repeated development V; shared-A dependence; no independent generalization or learnability claim'])
        write_gzip(output/'scene-summary.json.gz', rows)
        write_json(output/'summary.json', result)
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
                     error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-system', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    run(args)
    print(f'COMPLETED: {args.output}/summary.json')


if __name__ == '__main__':
    main()
