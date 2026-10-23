"""077 fixed half-residual control: two seeds, existing CPU caches only.

No checkpoint loading, network calls, archives, fitting or validation selection.
The alpha=.5 control is generated for every original output group before truth
alignment. The prior 074 pairing attestation is inherited, not repeated here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.sequence_position import REGIONS
from association.evaluation.support_gate_objectives import _artifact, _check_artifact, _read
from association.evaluation.system_attribution import _same
from association.models.sequence_position import tangent_position
from association.utils.runtime import read_json_or_gzip as read_json, write_json, write_json_gzip
from tools.analysis.compare_position_benchmarks import _pool
from tools.analysis.compare_position_harm import scene_pair, harm_statistics

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('fixed_prior095', 'linear', 'harm', 'half_linear')
CONFIGS = {20261023: ('extended-level1', 'relative-harm-level1'),
           20261021: ('extended-replica-level1', 'relative-harm-replica-level1')}
SCOPES = ('all_output_groups', 'duplicate_output_groups', 'unique_selected_groups', 'unique_valid_points')


def bind_pair(linear, harm, paired, seed):
    """Bind frozen metadata and the original completed CPU pairing certificate."""
    manifest = read_json(ROOT/'configs/data/mtad-level1-scene3000.json')
    ids = manifest['splits']['V_select']
    components = {sid: str(manifest['components'][sid]) for sid in ids}
    arms = {}
    for name, path, suffix in zip(('linear', 'harm'), (linear, harm), CONFIGS[seed]):
        path = Path(path).resolve()
        config = read_json(ROOT/f'configs/sequence-position-transformer-{suffix}.json')
        run, summary, fit = [read_json(path/p) for p in ('run.json', 'summary.json', 'fit/run.json')]
        if config['training']['seed'] != seed or config['training']['updates'] != 4000:
            raise ValueError('077 requires the two frozen seed/final4000 contracts')
        if any(r.get('status') != 'COMPLETED' or r.get('configuration') != config
               or r.get('optimizer_updates') != 4000 or r.get('test_opened') is not False
               for r in (run, summary, fit)):
            raise ValueError('077 source run/configuration/final4000 must be completed')
        if (run.get('source_system') != summary.get('source_system')
                or run.get('completed_endpoints') != ['initial', 'final']
                or run.get('validation_selection') is not False or fit.get('final_checkpoint') != 'final.pt'
                or fit.get('training_scene_ids') != manifest['splits']['T_core']):
            raise ValueError('077 source provenance, complete T population or endpoints differ')
        for label, step, filename in (('initial', 0, 'step-0.pt'), ('final', 4000, 'final.pt')):
            endpoint = read_json(path/label/'summary.json')
            for r in (read_json(path/label/'run.json'), endpoint):
                cp = r.get('checkpoint', {})
                if (r.get('status') != 'COMPLETED' or r.get('configuration') != config
                        or r.get('source_system') != run['source_system'] or r.get('device') != 'cpu'
                        or r.get('optimizer_updates') != 0 or r.get('test_opened') is not False
                        or r.get('groups_recomputed') is not False or r.get('threshold_reselected') is not False
                        or cp.get('step') != step or cp.get('actual_updates') != step
                        or cp.get('configured_final_updates') != 4000
                        or Path(cp.get('path', '')).resolve() != path/'fit'/filename):
                    raise ValueError('077 endpoint metadata does not bind original CPU checkpoint')
            actual_components = {sid: str(cid) for cid, group in endpoint.get('components', {}).items()
                                 for sid in group['scene_ids']}
            if endpoint.get('scene_ids') != ids or actual_components != components:
                raise ValueError('077 complete V population or components differ')
            for directory in ('predictions', 'system-predictions'):
                if {p.name for p in (path/label/directory).glob('*.npz')} != {f'states-{sid}.npz' for sid in ids}:
                    raise ValueError('077 source cache inventory incomplete or contains extra scenes')
        arms[name] = dict(path=path, source_system=run['source_system'], summary=endpoint)
    if arms['linear']['source_system'] != arms['harm']['source_system']:
        raise ValueError('077 same seed arms must share source056')
    paired = Path(paired).resolve()
    run, report = read_json(paired/'run.json'), read_json(paired/'summary.json')
    if (any(r.get('status') != 'COMPLETED' or r.get('model_forwards') != 0
            or r.get('optimizer_updates') != 0 or r.get('test_opened') is not False for r in (run, report))
            or report.get('inputs') != {name: str(arm['path']) for name, arm in arms.items()}
            or report.get('seed') != seed or report.get('scene_ids') != ids
            or run.get('completed_scenes') != len(ids)):
        raise ValueError('077 completed original paired table provenance differs')
    certificate = report.get('pairing', {})
    expected = dict(status='PAIR_VERIFIED', seed=seed, matched_updates=4000, matched_anchors=32000,
        initial_parameters_exact=True, initial_sampler_rng_exact=True, actual_successful_batches_exact=True)
    if any(certificate.get(key) != value for key, value in expected.items()):
        raise ValueError('077 original PAIR_VERIFIED attestation is incomplete')
    return arms, dict(scene_ids=ids, components=components, paired_path=str(paired), paired_summary=report,
        inherited_pairing=certificate, checkpoint_loaded_this_run=False,
        pairing_evidence='Inherited completed074 attestation; no weights or training log loaded in077')


def geometry(linear, harm):
    """Per-query geometry; zero-vector angle and projection are undefined."""
    linear, harm = np.asarray(linear, np.float64), np.asarray(harm, np.float64)
    if linear.shape != harm.shape or linear.shape[-1] != 2 or not np.isfinite([linear, harm]).all():
        raise ValueError('077 finite corresponding tangent vectors required')
    ln, hn = np.linalg.norm(linear, axis=-1), np.linalg.norm(harm, axis=-1)
    dot = np.sum(linear*harm, axis=-1)
    ratio = np.divide(hn, ln, out=np.full_like(ln, np.nan), where=ln > 0)
    projection = np.divide(dot, ln*ln, out=np.full_like(ln, np.nan), where=ln > 0)
    cosine = np.divide(dot, ln*hn, out=np.full_like(ln, np.nan), where=(ln > 0) & (hn > 0))
    cosine = np.clip(cosine, -1., 1.)
    orthogonal = np.linalg.norm(harm-projection[..., None]*linear, axis=-1)
    if not np.isfinite(ln).all() or not np.isfinite(hn).all():
        raise ValueError('077 residual norms overflow')
    if any(not np.isfinite(value[ln > 0]).all() for value in (ratio, projection, orthogonal)):
        raise ValueError('077 defined residual geometry must be finite')
    return dict(linear_norm_m=ln, harm_norm_m=hn, norm_ratio=ratio, cosine=cosine,
                projection_scale=projection, orthogonal_norm_m=orthogonal)


def distribution(value):
    value = np.asarray(value).ravel()
    valid = value[np.isfinite(value)]
    result = dict(total_count=len(value), defined_count=len(valid), undefined_count=len(value)-len(valid))
    return dict(result, mean=float(valid.mean()) if len(valid) else None,
        **{key: float(np.quantile(valid, q)) if len(valid) else None
           for key, q in (('p05', .05), ('p50', .5), ('p95', .95), ('p99', .99))})


def _physical(prior, residual):
    """Keep the original producer's per-group Q64 arithmetic shape."""
    if prior.ndim != 3 or prior.shape[-1] != 4 or residual.shape != (*prior.shape[:2], 2):
        raise ValueError('077 matching all-group prior/residual arrays required')
    return np.stack([tangent_position(torch.from_numpy(base), torch.from_numpy(delta)).numpy()
                     for base, delta in zip(prior, residual)])


def half_residual_states(prior, residual):
    """Inference-only map: no target, validity, group identity or fitted alpha."""
    return _physical(prior, residual*.5)


def scene_analysis(arms, sid):
    row, data = scene_pair(arms, sid)  # Replays ordinary masks/initials and all groups, no model loading.
    source = arms['linear']['path']/'final'
    fragment = _read(source/'predictions'/f'states-{sid}.npz')
    system = _read(source/'system-predictions'/f'states-{sid}.npz')
    base = _artifact(system, 'fixed_prior095')
    prior = base['group_states']
    residuals = {}
    for name, arm in arms.items():
        for label in ('initial', 'final'):
            saved = _read(arm['path']/label/'predictions'/f'states-{sid}.npz')
            residual = saved['group_tangent_residual_m']
            if residual.shape != (*prior.shape[:2], 2) or not np.isfinite(residual).all():
                raise ValueError('077 residual shape/value differs from all output groups')
            if label == 'initial':
                _same(residual, np.zeros_like(residual), '077 initial residual exactly zero')
            physical = _physical(prior, residual)
            _same(physical, saved['learned_sequence_group_states'], '077 all-group tangent replay/'+name+'/'+label)
        residuals[name] = residual
    half = half_residual_states(prior, residuals['linear'])
    _same(half[..., 2:], prior[..., 2:], '077 half preserves native speed/heading')
    changed = dict(base, group_states=half, aligned_prediction=np.full_like(base['aligned_prediction'], np.nan))
    for i, group in enumerate(base['aligned_group_indices']):
        if group >= 0:
            changed['aligned_prediction'][i] = half[group]
    t = len(base['target_ids'])
    mask = (base['aligned_truth_valid'][:t, :, :2] & base['aligned_prediction_valid'][:t, :, :2]).all(-1)
    half_error = attribute_errors(changed['aligned_prediction'][:t], base['aligned_truth'][:t])['position']
    candidate = dict(fragment, half_linear_group_states=half, half_linear_position_error=np.where(mask, half_error, np.nan))
    data['half_linear_position_error'] = _check_artifact(candidate, changed, base, 'half_linear')
    vectors = geometry(residuals['linear'], residuals['harm'])
    selected = base['aligned_group_indices'][:t]
    adopted = np.zeros(len(prior), bool)
    adopted[selected[selected >= 0]] = True
    duplicate = np.zeros(len(prior), bool)
    duplicate[base['aligned_group_indices'][t:]] = True
    routes = {}
    for scope, use in (('all_output_groups', np.ones(len(prior), bool)),
                       ('duplicate_output_groups', duplicate), ('unique_selected_groups', adopted)):
        routes[scope] = {key: value[use].ravel() for key, value in vectors.items()}
    mapped = {}
    for key, value in vectors.items():
        aligned = np.full(mask.shape, np.nan)
        aligned[selected >= 0] = value[selected[selected >= 0]]
        mapped[key] = aligned
    routes['unique_valid_points'] = {key: value[mask] for key, value in mapped.items()}
    difference = data['linear_position_error']-data['fixed_prior095_position_error']
    for name, use in (('linear_improved', difference < -1e-9), ('linear_harmed', difference > 1e-9),
                      ('linear_tied', abs(difference) <= 1e-9)):
        routes[name] = {key: value[mask & use] for key, value in mapped.items()}
    data['geometry'] = routes
    payload = dict(half_linear_group_states=half, half_linear_position_error=data['half_linear_position_error'],
        aligned_group_indices=base['aligned_group_indices'], aligned_prediction_valid=base['aligned_prediction_valid'],
        target_ids=base['target_ids'], group_anchor_node_indices=fragment['group_anchor_node_indices'],
        source_linear_final=np.asarray(str(source)), alpha=np.asarray(.5))
    return row, data, payload


def summarize(rows, arrays):
    result = _pool(rows, arrays, METHODS)
    result['relative_harm'] = {}
    for region in REGIONS:
        values = {m: np.concatenate([a[m+'_position_error'][a['region_'+region]
            & np.isfinite(a['fixed_prior095_position_error'])] for a in arrays]) for m in METHODS}
        result['relative_harm'][region] = {
            m: harm_statistics(values[m], values['fixed_prior095']) for m in METHODS}
        for candidate, control in (('half_linear', 'linear'), ('harm', 'linear'), ('harm', 'half_linear')):
            result['relative_harm'][region][candidate+'_minus_'+control] = harm_statistics(values[candidate], values[control])
    h, s = (result['relative_harm']['all'][name] for name in ('harm', 'half_linear'))
    guard = 1e-6*h['common_point_count']
    delta_h, delta_i = s['harm_sum']-h['harm_sum'], s['improvement_sum']-h['improvement_sum']
    half_dominates = delta_h <= guard and delta_i >= -guard and (delta_h < -guard or delta_i > guard)
    harm_dominates = delta_h >= -guard and delta_i <= guard and (delta_h > guard or delta_i < -guard)
    result['half_vs_harm_I_H'] = dict(cumulative_guard_m=guard, delta_H_m=delta_h, delta_I_m=delta_i,
        direction=('half_dominates_two_dimensions' if half_dominates else 'harm_dominates_two_dimensions'
                   if harm_dominates else 'numerical_tie' if abs(delta_h) <= guard and abs(delta_i) <= guard else 'tradeoff'),
        interpretation='I higher and H lower; numerical directions only, not meaningful-effect significance or MAE/tail promotion.')
    result['residual_geometry'] = {}
    for scope in arrays[0]['geometry']:
        pooled = {key: np.concatenate([a['geometry'][scope][key] for a in arrays])
                  for key in arrays[0]['geometry'][scope]}
        result['residual_geometry'][scope] = dict(
            metrics={key: distribution(value) for key, value in pooled.items()},
            linear_zero_count=int((pooled['linear_norm_m'] == 0).sum()),
            harm_zero_count=int((pooled['harm_norm_m'] == 0).sum()),
            both_zero_count=int(((pooled['linear_norm_m'] == 0) & (pooled['harm_norm_m'] == 0)).sum()))
    return result


def verify_source_metrics(result, arms, binding):
    for method in ('linear', 'harm'):
        actual = result['position_weightings']['all'][method]['point_equal']
        expected = arms[method]['summary']['methods']['learned_sequence']['full_target_metrics']['all']['position']
        if actual['unit_count'] != expected['intersection_count'] or any(not np.isclose(actual[k], expected[k],
                rtol=1e-10, atol=1e-9) for k in ('mae', 'rmse')):
            raise ValueError('077 physical errors do not reproduce completed producer/'+method)
        old = binding['paired_summary']
        for region in REGIONS:
            for weighting, metrics in old['position_weightings'][region][method].items():
                current = result['position_weightings'][region][method][weighting]
                for key, expected_value in metrics.items():
                    value = current.get(key)
                    equal = value is expected_value if value is None or expected_value is None else np.isclose(
                        value, expected_value, rtol=1e-10, atol=1e-9)
                    if not equal:
                        raise ValueError('077 existing paired table metrics differ')
        if result['population'] != old['population'] or result['position_population'] != old['position_population']:
            raise ValueError('077 existing paired table population differs')
        for region in REGIONS:
            for key in ('improvement_sum', 'harm_sum', 'common_point_count', 'squared_error_change_sum_m2'):
                if not np.isclose(result['relative_harm'][region][method][key],
                                  old['relative_harm'][region][method][key], rtol=1e-10, atol=1e-9):
                    raise ValueError('077 existing paired I/H accounting differs')


def run(args):
    if args.cpu_threads != 4:
        raise ValueError('077 fixes CPU4')
    torch.set_num_threads(4)
    inputs = {seed: {name: Path(getattr(args, name+str(seed)[-2:])).resolve()
                    for name in ('linear', 'harm', 'paired')} for seed in CONFIGS}
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('077 output exists; no retry/overwrite')
    sources = [p for group in inputs.values() for p in group.values()]
    sources += [Path(read_json(group['linear']/'run.json')['source_system']).resolve().parent for group in inputs.values()]
    if any(output == p or output in p.parents or p in output.parents for p in sources):
        raise ValueError('077 output overlaps immutable inputs')
    output.mkdir(parents=True)
    started = time.monotonic()
    state = dict(scope='077-fixed-half-residual', status='RUNNING', cpu_threads=4, completed_scenes=0,
        model_forwards=0, optimizer_updates=0, checkpoints_loaded=0, raw_archives_read=0, test_opened=False,
        threshold_reselected=False, groups_recomputed=False, validation_selection=False,
        alpha=.5, retries=0, advisory_reached=False, advisory_seconds=3600, time_policy='advisory', old066_acceptance='REJECTED')
    write_json(output/'run.json', state)
    try:
        bindings = {seed: bind_pair(**paths, seed=seed) for seed, paths in inputs.items()}
        source_ids = [arms['linear']['source_system'] for arms, _ in bindings.values()]
        if len(set(source_ids)) != 1:
            raise ValueError('077 both seeds must retain the same056 source')
        results, all_scenes = {}, {}
        for seed, (arms, binding) in bindings.items():
            directory = output/str(seed)
            (directory/'predictions').mkdir(parents=True)
            rows, arrays = [], []
            for sid in binding['scene_ids']:
                row, data, payload = scene_analysis(arms, sid)
                row['component_id'] = binding['components'][sid]
                rows.append(row); arrays.append(data)
                np.savez_compressed(directory/'predictions'/f'states-{sid}.npz', **payload)
                state.update(completed_scenes=state['completed_scenes']+1, elapsed_seconds=time.monotonic()-started)
                if state['elapsed_seconds'] >= 3600 and not state['advisory_reached']:
                    state['advisory_reached'] = True
                    print('ADVISORY: 077 exceeded one hour; continuing this single attempt', flush=True)
                write_json(output/'run.json', state)
            result = summarize(rows, arrays)
            verify_source_metrics(result, arms, binding)
            result['components'] = {}
            for component in sorted(set(binding['components'].values())):
                indices = [i for i, r in enumerate(rows) if r['component_id'] == component]
                result['components'][component] = summarize([rows[i] for i in indices], [arrays[i] for i in indices])
            largest = min(result['components'], key=lambda c: (
                -result['components'][c]['position_population']['covered_target_points'], c))
            rest = [i for i, r in enumerate(rows) if r['component_id'] != largest]
            result['largest_component_id'] = largest
            result['remaining_components'] = summarize([rows[i] for i in rest], [arrays[i] for i in rest]) if rest else None
            result.update(status='COMPLETED', seed=seed, inputs={k: str(v) for k, v in inputs[seed].items()},
                inherited_pairing=binding['inherited_pairing'], pairing_evidence=binding['pairing_evidence'])
            results[str(seed)] = result
            all_scenes[str(seed)] = [dict(summarize([r], [a]), component_id=r['component_id']) for r, a in zip(rows, arrays)]
        summary = dict(status='COMPLETED', seeds=results, methods=list(METHODS), alpha=.5,
            model_forwards=0, optimizer_updates=0, test_opened=False, checkpoints_loaded=0,
            interpretation='One prespecified half-residual control; no V alpha search. Failure does not exclude other shrink values. Success is not a learned-risk contribution. Geometry strata use truth only after predictions; reused V and shared identities are not independent generalization.')
        write_json_gzip(output/'summary.json.gz', summary)
        write_json_gzip(output/'scene-summary.json.gz', all_scenes)
        text = ['# 077 固定残差减半诊断', '', '两seed全部报告；零训练／前向，未重开原权重验证。', '',
                '| seed | method | MAE m | RMSE m | P95 m | P99 m | I m | H m |', '|---|---|---:|---:|---:|---:|---:|---:|']
        for seed, result in results.items():
            for method in METHODS:
                metrics = result['position_weightings']['all'][method]['point_equal']
                changes = result['relative_harm']['all'][method]
                text.append(f'| {seed} | {method} | '+' | '.join(f'{v:.4f}' for v in (
                    *(metrics[k] for k in ('mae', 'rmse', 'p95', 'p99')), changes['improvement_sum'], changes['harm_sum']))+' |')
        text += ['', *(f"{seed} I/H两维方向：{result['half_vs_harm_I_H']['direction']}；"
                       f"half−harm ΔI={result['half_vs_harm_I_H']['delta_I_m']:.4f} m，"
                       f"ΔH={result['half_vs_harm_I_H']['delta_H_m']:.4f} m；仅数值方向，不自动晋级。"
                       for seed, result in results.items()),
                 '重复组全部参与输出变换和几何统计；未人为补真值计算其误差。定义为零的向量方向报null。',
                 summary['interpretation']]
        (output/'report.md').write_text('\n'.join(text)+'\n')
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('linear23', 'harm23', 'paired23', 'linear21', 'harm21', 'paired21', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    result = run(parser.parse_args(argv))
    print(result['status'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
