"""078 cached same-population screening against072C and matched helper arms.

Accept one completed helper arm for immediate comparison, or all three for the
primary A/B/C closeout. Never rerun inference or promote a single-seed winner.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.helper_sequence_position import helper_population, BUCKETS, POPULATIONS
from association.evaluation.sequence_position import REGIONS
from association.evaluation.support_gate_objectives import _read
from association.evaluation.system_attribution import _same
from association.training.helper_sequence import load_checkpoint
from association.utils.runtime import read_json, write_json, write_json_gzip
from tools.analysis.compare_position_benchmarks import _metadata, _pool, _sequence_sampling, CONTROLS
from tools.analysis.compare_position_harm import scene_pair, harm_statistics
from tools.experiments.helper_sequence_fit import CONFIGS, ROOT, bind_reference


def helper_metadata(path):
    path = Path(path).resolve()
    run, summary, fit = [read_json(path/name) for name in ('run.json', 'summary.json', 'fit/run.json')]
    config = run['configuration']
    mode = config['model']['aggregation']
    if config != read_json(ROOT/'configs'/CONFIGS[mode]):
        raise ValueError('078 comparison requires the frozen aggregation configuration')
    steps = config['training']['updates']
    for record in (run, summary, fit):
        if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                or record.get('optimizer_updates') != steps or record.get('test_opened') is not False):
            raise ValueError('078 requires completed actual updates/configuration')
    if (run.get('completed_endpoints') != ['initial', 'final'] or run.get('validation_selection') is not False
            or summary.get('source_system') != run['source_system']
            or summary.get('paired_control') != run.get('paired_control') or fit.get('final_checkpoint') != 'final.pt'):
        raise ValueError('078 final endpoint/provenance is incomplete')
    manifest = read_json(ROOT/'configs/data/mtad-level1-scene3000.json')
    if fit.get('training_scene_ids') != manifest['splits']['T_core']:
        raise ValueError('078 complete training population differs')
    bind_reference(run['paired_control'], config, run['source_system'], manifest)
    for label, step, filename in (('initial', 0, 'step-0.pt'), ('final', steps, 'final.pt')):
        with torch.random.fork_rng(devices=[]):
            model, checkpoint = load_checkpoint(path/'fit'/filename, device='cpu')
        del model
        if (checkpoint['configuration'] != config or checkpoint['step'] != step or checkpoint['actual_updates'] != step):
            raise ValueError('078 actual endpoint checkpoint differs')
        final = read_json(path/label/'summary.json')
        for record in (read_json(path/label/'run.json'), final):
            provenance = record.get('checkpoint', {})
            if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                    or record.get('source_system') != run['source_system'] or record.get('device') != 'cpu'
                    or record.get('optimizer_updates') != 0 or record.get('test_opened') is not False
                    or record.get('groups_recomputed') is not False or record.get('threshold_reselected') is not False
                    or provenance.get('step') != step or provenance.get('actual_updates') != step
                    or provenance.get('configured_final_updates') != steps
                    or Path(provenance.get('path', '')).resolve() != path/'fit'/filename):
                raise ValueError('078 endpoint output does not bind actual checkpoint')
        if final['scene_ids'] != manifest['splits']['V_select']:
            raise ValueError('078 complete validation population differs')
        for directory in ('predictions', 'system-predictions'):
            if {p.name for p in (path/label/directory).glob('*.npz')} != {f'states-{s}.npz' for s in final['scene_ids']}:
                raise ValueError('078 endpoint cache inventory differs')
    components = {sid: str(cid) for cid, group in final['components'].items() for sid in group['scene_ids']}
    if components != {sid: str(manifest['components'][sid]) for sid in final['scene_ids']}:
        raise ValueError('078 component registry differs')
    return mode, dict(path=path, method='learned_sequence', configuration=config,
        source_system=run['source_system'], scene_ids=final['scene_ids'], components=components, summary=final)


def verify_pairing(arms):
    original, old_rows, old_exposure = _sequence_sampling(arms['linear']['path']/'fit')
    canonical = lambda rows: [{k: row[k] for k in ('step', 'mode', 'batch')} for row in rows]
    with torch.random.fork_rng(devices=[]):
        from association.training.sequence_position import load_checkpoint as old_checkpoint
        old_model, old = old_checkpoint(arms['linear']['path']/'fit/step-0.pt', device='cpu')
    del old_model
    snapshots, reports = {}, {}
    for name, arm in arms.items():
        if name == 'linear':
            continue
        fit, rows, exposure = _sequence_sampling(arm['path']/'fit')
        if (canonical(rows) != canonical(old_rows) or exposure != old_exposure
                or fit['training_scene_ids'] != original['training_scene_ids']):
            raise ValueError('078 actual successful samples/exposure differ from072C')
        with torch.random.fork_rng(devices=[]):
            model, saved = load_checkpoint(arm['path']/'fit/step-0.pt', device='cpu')
        del model
        for key, value in old['model'].items():
            candidate = saved['model'][key]
            if key == 'input_projection.0.weight':
                if (candidate[:, 76:] != 0).any():
                    raise ValueError('078 added projection columns must start at zero')
                candidate = candidate[:, :76]
            if not torch.equal(candidate, value):
                raise ValueError('078 inherited common initial parameter differs/'+key)
        if saved['numpy_rng'] != old['numpy_rng']:
            raise ValueError('078 initial sampling RNG differs')
        snapshots[name] = saved['model']
        if name == 'weighted_pool' and (saved['model']['scorer.weight'] != 0).any():
            raise ValueError('078 weighted scorer must start at zero')
        reports[name] = dict(actual_batches_exact=True, common_initial_parameters_exact=True,
            initial_sampling_rng_exact=True, sampling_exposure_exact=True, matched_updates=len(rows),
            parameter_count=fit['parameter_count'], helper_support_exposure=fit['helper_support_exposure'],
            gradient_clipping=fit['gradient_clipping'], helper_gradients=fit['helper_gradients'])
    values = list(snapshots.items())
    for i, (left_name, left) in enumerate(values):
        for right_name, right in values[i+1:]:
            for key in set(left) & set(right):
                if not torch.equal(left[key], right[key]):
                    raise ValueError('078 shared new initial parameters differ/'+left_name+'/'+right_name+'/'+key)
    return dict(status='PAIR_VERIFIED', seed=original['configuration']['training']['seed'],
        matched_updates=len(old_rows), matched_anchors=old_exposure['sampled_anchors'], arms=reports,
        shared_added_initial_parameters_exact=True, equal_compute_claimed=False)


def helper_cohorts(fragment, system):
    counts = fragment['group_supported_helper_count']
    t = len(system['anchor_target_ids'])
    selected = system['anchor_aligned_group_indices'][:t]
    shape = system['anchor_aligned_truth_valid'][:t].shape[:2]
    target_counts = np.zeros(shape, np.int64)
    exists = selected >= 0
    target_counts[exists] = counts[selected[exists]]
    return {name: exists[:, None] & test for name, test in zip(
        BUCKETS, (target_counts == 0, target_counts == 1, target_counts >= 2))}


def summarize(rows, arrays, methods):
    result = _pool(rows, arrays, methods)
    result['relative_harm'] = {}
    for region in REGIONS:
        use = lambda a: a['region_'+region] & np.isfinite(a['anchor_position_error'])
        values = {name: np.concatenate([a[name+'_position_error'][use(a)] for a in arrays]) for name in methods}
        result['relative_harm'][region] = {name: harm_statistics(values[name], values['fixed_prior095'])
                                          for name in methods if name not in (*CONTROLS, 'fixed_prior095')}
        pairs = [(name, 'linear') for name in methods if name not in (*CONTROLS, 'fixed_prior095', 'linear')]
        pairs += [(a, b) for a, b in (('pre_pool', 'post_pool'), ('weighted_pool', 'pre_pool'))
                  if a in values and b in values]
        for new, old in pairs:
            result['relative_harm'][region][new+'__minus__'+old] = harm_statistics(values[new], values[old])
    result['helper_cohorts'] = {}
    for cohort in BUCKETS:
        local = [dict(a, region_all=a['helper_cohort_'+cohort]) for a in arrays]
        result['helper_cohorts'][cohort] = _pool(rows, local, methods)['position_weightings']['all']
    return result


def run(args):
    if args.cpu_threads != 4:
        raise ValueError('078 comparison fixes CPU4')
    torch.set_num_threads(4)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('078 comparison exists; no retry/overwrite')
    paths = [Path(p).resolve() for p in args.helper_run]
    linear_path = Path(args.linear_run).resolve()
    for path in [linear_path, *paths]:
        source = Path(read_json(path/'run.json')['source_system']).resolve().parent
        for protected in (path, source):
            if output == protected or output in protected.parents or protected in output.parents:
                raise ValueError('078 comparison output overlaps an immutable input')
    output.mkdir(parents=True)
    state = dict(status='RUNNING', completed_scenes=0, model_forwards=0, optimizer_updates=0, test_opened=False)
    write_json(output/'run.json', state)
    try:
        arms = {'linear': _metadata(linear_path, '072C_transformer4000')}
        for path in paths:
            name, arm = helper_metadata(path)
            if name in arms:
                raise ValueError('078 duplicate aggregation arm')
            arms[name] = arm
        first = arms['linear']
        if len(arms) < 2 or any(arm[key] != first[key] for arm in arms.values()
            for key in ('scene_ids', 'source_system', 'components')):
            raise ValueError('078 requires shared source/population/components and at least one helper arm')
        pairing = verify_pairing(arms)
        methods = (*CONTROLS, 'fixed_prior095', *arms)
        rows, arrays = [], []
        helper_populations = []
        for sid in first['scene_ids']:
            row, data = scene_pair(arms, sid)
            row['component_id'] = first['components'][sid]
            ref = None
            for name in arms:
                if name == 'linear':
                    continue
                fragment = _read(arms[name]['path']/'final/predictions'/f'states-{sid}.npz')
                system = _read(arms[name]['path']/'final/system-predictions'/f'states-{sid}.npz')
                population = helper_population(fragment, system)
                if ref is None:
                    ref = fragment
                    data.update({'helper_cohort_'+k: v for k, v in helper_cohorts(fragment, system).items()})
                    helper_populations.append(population)
                else:
                    for key in ('group_supported_helper_count', 'group_helper_offsets', 'group_helper_node_indices'):
                        _same(fragment[key], ref[key], '078 paired helper membership/'+key)
            rows.append(row); arrays.append(data)
            state['completed_scenes'] = len(rows)
            write_json(output/'run.json', state)
        summary = summarize(rows, arrays, methods)
        for name, arm in arms.items():
            actual = summary['position_weightings']['all'][name]['point_equal']
            expected = arm['summary']['methods']['learned_sequence']['full_target_metrics']['all']['position']
            if actual['unit_count'] != expected['intersection_count'] or any(
                not np.isclose(actual[k], expected[k], rtol=1e-10, atol=1e-9) for k in ('mae', 'rmse')):
                raise ValueError('078 cache does not reproduce producer physical metrics/'+name)
        scene_rows = [dict(summarize([r], [a], methods), component_id=r['component_id']) for r, a in zip(rows, arrays)]
        components = {}
        for cid in sorted(set(first['components'].values())):
            ids = [i for i, row in enumerate(rows) if row['component_id'] == cid]
            components[cid] = summarize([rows[i] for i in ids], [arrays[i] for i in ids], methods)
        largest = min(components, key=lambda cid: (-components[cid]['position_population']['covered_target_points'], cid))
        rest = [i for i, row in enumerate(rows) if row['component_id'] != largest]
        population = {name: dict(total=sum(row[name]['total'] for row in helper_populations), counts={
            bucket: sum(row[name]['counts'][bucket] for row in helper_populations) for bucket in BUCKETS})
            for name in POPULATIONS}
        summary.update(status='COMPLETED', pairing=pairing, methods=list(methods), components=components,
            largest_component_id=largest, remaining_components=summarize([rows[i] for i in rest],
            [arrays[i] for i in rest], methods) if rest else None, helper_population=population,
            inputs={name: str(arm['path']) for name, arm in arms.items()}, single_seed_screen=True,
            complete_abc=set(arms) == {'linear', *CONFIGS}, model_forwards=0, optimizer_updates=0,
            test_opened=False, old066_acceptance='REJECTED',
            interpretation='Single-seed fixed-endpoint screening. Reused V/shared identities; improvement is not independent generalization. Feature-pooling weights are not reliability attribution.')
        write_json_gzip(output/'summary.json.gz', summary)
        write_json_gzip(output/'scene-summary.json.gz', scene_rows)
        lines = ['# 078 helper representation screening', '', '| Method | MAE m | RMSE m | P95 m | P99 m |',
                 '|---|---:|---:|---:|---:|']
        for name in methods:
            value = summary['position_weightings']['all'][name]['point_equal']
            lines.append('| '+name+' | '+' | '.join(f'{value[k]:.4f}' for k in ('mae', 'rmse', 'p95', 'p99'))+' |')
        lines += ['', 'All outputs, duplicate groups and valid target masks retained. H/I and fixed100/1000/10000m harm counts, four weightings, support/helper strata and components are in JSON.',
                  'A/B/C are exploratory single-seed results. No automatic promotion or retry.']
        (output/'report.md').write_text('\n'.join(lines)+'\n')
        state.update(status='COMPLETED', population=summary['population'])
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helper-run', nargs='+', required=True)
    parser.add_argument('--linear-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    result = run(parser.parse_args(argv))
    print(result['status'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
