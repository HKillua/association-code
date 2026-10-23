"""073: cached two-seed, two-method final4000 comparison, without inference."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.utils.runtime import read_json, write_json, write_json_gzip
from association.evaluation.support_gate_replication import _fit as gate_fit_records
from tools.analysis.compare_position_benchmarks import (
    CONTROLS, _metadata, _pool, _sequence_sampling, scene_arrays)

ARMS = ('073A_gate4000_seed23', '073B_gate4000_seed21',
        '073C_transformer4000_seed21', '072C_transformer4000')
METHODS = (*CONTROLS, 'fixed_prior095', *ARMS)


def _samples(rows):
    return [{key: row[key] for key in ('step', 'mode', 'batch')} for row in rows]


def verify_method_sampling(gate_fit, sequence_fit):
    gate, gr, _, _, ge = gate_fit_records(gate_fit, 'linear')
    sequence, sr, se = _sequence_sampling(sequence_fit)
    # The gate has additional per-mode eligibility counters, already checked
    # against its raw batches by gate_fit_records. Compare shared counters here.
    for counters in ge['modes'].values():
        counters.pop('eligible_query_points')
        counters.pop('eligible_valid_position_points')
    # Model/routing schemas deliberately differ; common data and training do not.
    for key in ('data', 'training', 'train_splits', 'motion_observation_policy', 'position_loss'):
        if gate['configuration'][key] != sequence['configuration'][key]:
            raise ValueError('073 paired methods differ in shared configuration/'+key)
    if (gate['training_scene_ids'] != sequence['training_scene_ids']
            or _samples(gr) != _samples(sr) or ge != se):
        raise ValueError('073 actual gate/sequence batches or exposure differ')
    return dict(status='PAIR_VERIFIED', seed=gate['configuration']['training']['seed'],
        actual_successful_batches_exact=True, matched_updates=len(gr),
        matched_anchors=ge['sampled_anchors'], exposure=ge,
        interpretation='Actual sampling is paired; different models have different initial parameters and optimization paths.')


def verify_gate_prefix(current_fit, previous_fit):
    current, cr, left, _, _ = gate_fit_records(current_fit, 'linear')
    previous, pr, right, _, _ = gate_fit_records(previous_fit, 'linear')
    config = deepcopy(previous['configuration'])
    config['training']['updates'] = current['configuration']['training']['updates']
    if (config != current['configuration'] or len(cr) <= len(pr)
            or current['training_scene_ids'] != previous['training_scene_ids']
            or _samples(cr[:len(pr)]) != _samples(pr)):
        raise ValueError('073 gate budget or actual earlier sampling prefix differs')
    if (left['actual_updates'] != 0 or right['actual_updates'] != 0
            or set(left['model']) != set(right['model']) or any(
            left['model'][key].dtype != right['model'][key].dtype
            or not torch.equal(left['model'][key], right['model'][key]) for key in left['model'])
            or left['numpy_rng'] != right['numpy_rng']):
        raise ValueError('073 gate initial parameters or sampler RNG differ')
    return dict(previous_fit=str(Path(previous_fit).resolve()), initial_parameters_exact=True,
        initial_sampler_rng_exact=True, actual_prefix_batches_exact=True, matched_prefix_updates=len(pr),
        interpretation='No intermediate1000 checkpoint in the new run: no bitwise optimizer-trajectory claim.')


def run(args):
    paths = dict(zip(ARMS, map(lambda p: Path(p).resolve(),
        (args.gate_run073A, args.gate_run073B, args.transformer_run073C, args.transformer_run072C))))
    prefixes = [Path(args.source071B).resolve(), Path(args.source070).resolve()]
    output = Path(args.output).resolve()
    if args.cpu_threads != 4:
        raise ValueError('073 cached comparison fixes CPU4')
    torch.set_num_threads(4)
    if output.exists():
        raise FileExistsError('comparison output already exists; no overwrite/retry')
    sources = [Path(read_json(p/'run.json')['source_system']).resolve().parent for p in paths.values()]
    for path in (*paths.values(), *prefixes, *sources):
        if output == path or output in path.parents or path in output.parents:
            raise ValueError('073 output must be separate from immutable inputs')
    output.mkdir(parents=True)
    state = dict(status='RUNNING', model_forwards=0, optimizer_updates=0,
                 test_opened=False, completed_scenes=0)
    write_json(output/'run.json', state)
    try:
        arms = {name: _metadata(path, name) for name, path in paths.items()}
        pairing = {
            '20261023': verify_method_sampling(paths[ARMS[0]]/'fit', paths[ARMS[3]]/'fit'),
            '20261021': verify_method_sampling(paths[ARMS[1]]/'fit', paths[ARMS[2]]/'fit')}
        prefix = {name: verify_gate_prefix(paths[name]/'fit', previous/'fit')
                  for name, previous in zip(ARMS[:2], prefixes)}
        if any(item['matched_prefix_updates'] != 1000 for item in prefix.values()):
            raise ValueError('073 gate prefixes must bind the completed070/071B1000-update fits')
        first = arms[ARMS[0]]
        for arm in arms.values():
            if any(arm[key] != first[key] for key in ('source_system', 'scene_ids', 'components')):
                raise ValueError('073 full population, groups source or component registry differs')
        rows, arrays = [], []
        for sid in first['scene_ids']:
            row, data = scene_arrays(arms, sid, paths[ARMS[0]]/'initial')
            row['component_id'] = first['components'][sid]
            rows.append(row); arrays.append(data)
            state['completed_scenes'] = len(rows)
            write_json(output/'run.json', state)
        summary = _pool(rows, arrays, METHODS)
        for name, arm in arms.items():
            actual = summary['position_weightings']['all'][name]['point_equal']
            expected = arm['summary']['methods'][arm['method']]['full_target_metrics']['all']['position']
            if actual['unit_count'] != expected['intersection_count'] or any(
                    not np.isclose(actual[k], expected[k], rtol=1e-10, atol=1e-9) for k in ('mae', 'rmse')):
                raise ValueError('073 cache metrics disagree with producer/'+name)
        components = {}
        for component in sorted(set(first['components'].values())):
            indices = [i for i, row in enumerate(rows) if row['component_id'] == component]
            components[component] = _pool([rows[i] for i in indices], [arrays[i] for i in indices], METHODS)
        largest = min(components, key=lambda key: (
            -components[key]['position_weightings']['all']['anchor']['point_equal']['unit_count'], key))
        rest = [i for i, row in enumerate(rows) if row['component_id'] != largest]
        summary.update(status='COMPLETED', methods=list(METHODS), pairing=pairing, gate_prefixes=prefix,
            components=components, largest_component_id=largest,
            remaining_components=_pool([rows[i] for i in rest], [arrays[i] for i in rest], METHODS) if rest else None,
            inputs={name: str(path) for name, path in paths.items()}, source_system=first['source_system'],
            model_forwards=0, optimizer_updates=0, test_opened=False, old066_acceptance='REJECTED',
            interpretation='Both declared seeds and fixed4000 endpoints are reported; no best-seed/step selection, no convergence or unseen-identity claim.',
            comparability='Equal updates and actual samples do not equal capacity, compute, output freedom or optimal hyperparameters; speed/course remain Mean.')
        write_json_gzip(output/'summary.json.gz', summary)
        write_json_gzip(output/'scene-summary.json.gz', [dict(_pool([r], [a], METHODS),
            component_id=r['component_id']) for r, a in zip(rows, arrays)])
        table = ['# 073 两seed同4000更新位置比较', '',
                 '| 方法 | MAE m | RMSE m | P95 m | P99 m |', '|---|---:|---:|---:|---:|']
        for name in METHODS:
            metric = summary['position_weightings']['all'][name]['point_equal']
            table.append('| '+name+' | '+' | '.join(f'{metric[k]:.4f}' for k in ('mae', 'rmse', 'p95', 'p99'))+' |')
        table += ['', '同056完整分组/人口/控制，逐场、组件、机会区和四权重见JSON；重复使用V，不是封存测试结论。']
        (output/'report.md').write_text('\n'.join(table)+'\n')
        state.update(status='COMPLETED', output_population=summary['population'])
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('gate-run073A', 'gate-run073B', 'transformer-run073C', 'transformer-run072C',
                'source071B', 'source070', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    result = run(args)
    print(result['status']+': '+str(Path(args.output).resolve()), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
