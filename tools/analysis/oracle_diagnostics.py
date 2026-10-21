"""058: CPU-only restricted oracle diagnostics from complete frozen053 V caches."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.oracle_diagnostics import METHODS, aggregate_oracles, analyze_oracles
from association.evaluation.system_attribution import BASE, assert_metrics
from association.utils.runtime import write_json
from tools.analysis.system_attribution import load_source, read, write_gzip


def run(args):
    source, output = Path(args.source_system).resolve(), Path(args.output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('058 output must be separate from the saved053 source')
    if not isinstance(args.cpu_threads, int) or isinstance(args.cpu_threads, bool) or args.cpu_threads < 1:
        raise ValueError('cpu_threads must be a positive integer')
    torch.set_num_threads(args.cpu_threads)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', scope='058-cached-restricted-oracles', dataset_level='Level1',
        split='V_select', source_system=str(source), cpu_threads=args.cpu_threads,
        optimizer_updates=0, gpu_used=False, archive_members_read=0, reference_members_read=0,
        cached_reference_used=True, ols_refitted=False, threshold_reselected=False,
        test_opened=False, completed_scenes=0, total_scenes=None)
    write_json(output/'run.json', state)
    try:
        original, _, expected, sources = load_source(source, read(args.manifest))
        ids = original['scene_ids']
        state['total_scenes'] = len(ids)
        write_json(output/'run.json', state)
        (output/'scenes').mkdir()
        (output/'predictions').mkdir()
        rows, fragment_count = [], 0
        for sid in ids:
            with np.load(source/'predictions'/f'states-{sid}.npz', allow_pickle=False) as f:
                fragment = {key: f[key] for key in f.files}
            with np.load(source/'system-predictions'/f'states-{sid}.npz', allow_pickle=False) as f:
                saved = {key[len(BASE)+1:]: f[key] for key in f.files if key.startswith(BASE+'_')}
            report, arrays = analyze_oracles(fragment, saved, expected[sid], source_names=sources)
            report['scene_id'] = sid
            fragment_count += report['fragment_count']
            write_gzip(output/'scenes'/f'scene-{sid}.json.gz', report)
            np.savez_compressed(output/'predictions'/f'states-{sid}.npz', **arrays)
            for method in report['methods'].values():
                method['system'].pop('all_predicted_groups')
            rows.append(report)
            state.update(completed_scenes=len(rows), last_scene_id=sid, elapsed_seconds=time.monotonic()-started)
            write_json(output/'run.json', state)
        if fragment_count != original['fragment_count']:
            raise ValueError('original fragment population changed')
        pooled = aggregate_oracles(rows)
        assert_metrics(pooled['methods']['predicted_mean_supported']['full_target_metrics'],
            {key: original['unique_target'][BASE][key] for key in ('all', 'observed', 'missing')},
            '053 pooled mean replay')
        write_gzip(output/'scene-summary.json.gz', [dict(scene_id=row['scene_id'],
            source_replay_verified=row['source_replay_verified'],
            methods={name: dict(all=method['system']['full_target_metrics']['all'],
                predicted_group_count=method['system']['predicted_group_count'],
                duplicate_prediction_group_count=method['system']['duplicate_prediction_group_count'])
                for name, method in row['methods'].items()},
            pairwise={name: dict(methods={method: metrics['all'] for method, metrics in pair['methods'].items()},
                coverage_transitions=pair['coverage_transitions'],
                same_representative_target_count=pair['same_representative_target_count'],
                changed_representative_target_count=pair['changed_representative_target_count'],
                error_change_second_minus_first={key: pair['error_change_second_minus_first'][key]
                    for key in ('all', 'same_representative', 'changed_representative')},
                newly_covered=pair['newly_covered']['all'], lost_coverage=pair['lost_coverage']['all'])
                for name, pair in row['pairwise'].items()}, selector_choices=row['selector_choices']) for row in rows])
        pooled.update(status='COMPLETED', scope=state['scope'], dataset_level='Level1', split='V_select',
            scene_ids=ids, fragment_count=fragment_count, method_names=list(METHODS),
            all_source_scenes_replayed=True, ols_working_point=original['ols_working_point'],
            source_system=str(source), optimizer_updates=0, gpu_used=False,
            archive_members_read=0, reference_members_read=0, cached_reference_used=True,
            test_opened=False, threshold_reselected=False, ols_refitted=False,
            definitions=dict(
                grouping='fixed original predicted partition or complete truth-identity partition; first input representative',
                selector='per-query position pair/speed/circular heading chooses lower-error fixed Anchor/Mean; tie/invalid truth uses Anchor',
                bound='oracle selector is a two-candidate error lower bound, not a bound for all estimators or deployable decisions',
                population='all original targets including no-input/reference-invalid targets; duplicate predictions retained separately',
                common='truth-valid AND both prediction-valid; same/changed representatives reported separately',
                gained='newly covered target points under same output rule; separate from common-point errors',
                ordinary='native states/support/nearest only, no reference states used',
                selector_choices='group-point counts include duplicate outputs; main errors use unique-target rows',
                strata='fixed true-identity input support union; evaluation only; no regrouping by strata'),
            limitations=['repeated V development population; not independent generalization',
                'truth grouping coverage/duplicate benefits are construction effects',
                'pointwise attribute selection need not satisfy joint motion consistency',
                'small selector gains cannot rule out other fusion candidates or methods',
                'points within trajectories/scenes and shared identities are not independent replicates'])
        write_json(output/'summary.json', pooled)
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return pooled
    except BaseException as error:
        status = 'INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED'
        state.update(status=status, error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
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
