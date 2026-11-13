"""099 cached peer comparison and prespecified two-learning-rate selection.

Every comparison rechecks actual inputs, groups, truth, masks, ordinary controls
and cached Haversine errors. No model forward, training, test access or automatic
scientific qualification occurs. Learning-rate selection minimizes the peer's
own fixed-final MAE; it cannot select the peer that is easiest for us to beat.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.frozen_position_adapter import exact
from association.evaluation.sequence_position import REGIONS, WEIGHTINGS
from association.evaluation.support_gate_objectives import _artifact, _check_artifact, _structure
from association.training import deepaf_peer as trainer
from association.training.frozen_position_adapter import _exact
from association.utils.runtime import read_json, write_json, write_json_gzip
from tools.analysis import compare_frozen_position_adapter as budget_compare
from association.evaluation import relation_sources as relation_compare
from association.evaluation.position_sources import (
    ROOT, CONTROLS, INPUT_KEYS, _metadata, _pool, _sequence_sampling,
)
from tools.analysis.compare_position_harm import _cache, _visible_opportunity

SCOPE = '099-deepaf-peer-cached-comparison'
SELECTION_SCOPE = '099-deepaf-peer-lr-selection'
SELECTION_RULE = 'minimum_final8000_full_population_point_mae_tie_lr_0.001'
EVALUATION_SCOPE = '099-deepaf-pv-position-evaluation'
FIXED_METHODS = (*CONTROLS, 'fixed_prior095')
CONFIG_NAMES = {
    (20261023, .001): 'deepaf-peer-level1-lr1e3.json',
    (20261023, .0003): 'deepaf-peer-level1-lr3e4.json',
    (20261021, .001): 'deepaf-peer-replica-lr1e3.json',
    (20261021, .0003): 'deepaf-peer-replica-lr3e4.json',
}
EPS_M = 1e-6


def declared_manifest():
    return read_json(ROOT/'configs/data/mtad-level1-scene3000.json')


def peer_metadata(path):
    """Bind complete production endpoints to actual scratch fits and replay."""
    from tools.experiments.deepaf_peer_fit import frozen_configuration

    path = Path(path).resolve()
    run, summary, fit = [read_json(path/name) for name in ('run.json', 'summary.json', 'fit/run.json')]
    config = frozen_configuration(run['configuration'])
    seed, lr = config['training']['seed'], config['training']['learning_rate']
    steps, middle = config['training']['updates'], config['training']['source_updates']
    endpoint_plan = (('initial', 0, 'step-0.pt'), ('mid', middle, f'step-{middle}.pt'),
                     ('final', steps, 'final.pt'))
    config_name = CONFIG_NAMES.get((seed, lr))
    if config_name is None or read_json(ROOT/'configs'/config_name) != config:
        raise ValueError('099 peer configuration is not one of the four declared production recipes')
    for record in (run, summary, fit):
        if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                or record.get('optimizer_updates') != steps or record.get('test_opened') is not False
                or record.get('source_weights_inherited') is not False):
            raise ValueError('099 peer completed scratch configuration/update chain differs')
    if (run.get('completed_endpoints') != [item[0] for item in endpoint_plan]
            or summary.get('source_system') != run.get('source_system')
            or not run.get('paired_control')
            or summary.get('paired_control') != run.get('paired_control')
            or fit.get('final_checkpoint') != 'final.pt'):
        raise ValueError('099 peer complete endpoint/source/fixed-final provenance differs')
    for record in (run, summary):
        selected_lr = seed == 20261021
        if (record.get('validation_checkpoint_selection') is not False
                or record.get('validation_learning_rate_selection') is not selected_lr
                or record.get('validation_selection') is not selected_lr):
            raise ValueError('099 peer checkpoint/LR validation selection disclosures differ')
    manifest = declared_manifest()
    if fit.get('training_scene_ids') != manifest['splits']['T_core']:
        raise ValueError('099 peer complete training population differs')
    paired_control = Path(run['paired_control']).resolve()
    pairing = trainer.verify_replay(path/'fit', paired_control/'fit')
    endpoints, checkpoints = {}, {}
    for label, step, filename in endpoint_plan:
        endpoint = read_json(path/label/'summary.json')
        for record in (read_json(path/label/'run.json'), endpoint):
            saved = record.get('checkpoint', {})
            if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                    or record.get('scope') != EVALUATION_SCOPE
                    or record.get('source_system') != run['source_system']
                    or record.get('device') != 'cpu' or record.get('optimizer_updates') != 0
                    or record.get('test_opened') is not False
                    or record.get('groups_recomputed') is not False
                    or record.get('threshold_reselected') is not False
                    or record.get('initial') != (label == 'initial')
                    or saved.get('step') != step or saved.get('actual_updates') != step
                    or saved.get('configured_final_updates') != steps
                    or Path(saved.get('path', '')).resolve() != path/'fit'/filename):
                raise ValueError('099 peer endpoint fails its CPU checkpoint/configuration binding')
        if endpoint.get('scene_ids') != manifest['splits']['V_select']:
            raise ValueError('099 peer full V population/order differs')
        if type(endpoint.get('chart_excursion_count')) is not int or endpoint['chart_excursion_count'] < 0:
            raise ValueError('099 peer must report its finite-chart excursion count')
        components = {sid: str(cid) for cid, item in endpoint['components'].items() for sid in item['scene_ids']}
        if components != {sid: str(manifest['components'][sid]) for sid in endpoint['scene_ids']}:
            raise ValueError('099 peer shared-identity component registry differs')
        for directory in ('predictions', 'system-predictions'):
            if {p.name for p in (path/label/directory).glob('*.npz')} != {
                    f'states-{sid}.npz' for sid in endpoint['scene_ids']}:
                raise ValueError('099 peer endpoint cache inventory incomplete or has extra scenes')
        model, checkpoint = trainer.load_checkpoint(path/'fit'/filename, device='cpu')
        if (checkpoint['configuration'] != config or checkpoint['step'] != step
                or Path(checkpoint['source_sampling']['fit']).resolve() != paired_control/'fit'):
            raise ValueError('099 actual checkpoint differs from the declared peer endpoint')
        count = sum(parameter.numel() for parameter in model.parameters())
        if (fit.get('parameter_count') != count or fit.get('trainable_parameter_count') != count
                or not all(parameter.requires_grad for parameter in model.parameters())):
            raise ValueError('099 peer trainable parameter count differs from its actual model')
        del model
        endpoints[label], checkpoints[label] = endpoint, checkpoint
    if _exact(checkpoints['initial']['model'], checkpoints['final']['model']):
        raise ValueError('099 completed peer weights did not change')
    reference_name = '072C_transformer4000' if seed == 20261023 else '073C_transformer4000_seed21'
    reference = _metadata(paired_control, reference_name)
    if any(reference[key] != value for key, value in (
            ('source_system', run['source_system']), ('scene_ids', endpoint['scene_ids']), ('components', components))):
        raise ValueError('099 source sampling run and peer evaluation populations differ')
    if seed == 20261021:
        from tools.experiments.deepaf_peer_fit import _selection
        selection = _selection(argparse.Namespace(selection_record=run.get('selection_record')), config)
        if (run.get('learning_rate_selection') != selection
                or summary.get('learning_rate_selection') != selection
                or summary.get('selection_record') != run.get('selection_record')):
            raise ValueError('099 second-seed peer does not bind the actual first-seed LR selection')
    return dict(path=path, configuration=config, config_name=config_name, method='learned_sequence',
        source_system=run['source_system'], scene_ids=endpoint['scene_ids'], components=components,
        summary=endpoints['final'], endpoints=endpoints, paired_control=paired_control,
        reference_endpoint=paired_control/'final', pairing=pairing,
        initial_state=checkpoints['initial']['model'], initial_rng=checkpoints['initial']['numpy_rng'],
        published_method='DeepAF')


def _verify_relation_arm(arm, reference):
    """Verify one completed 098 arm without requiring a different arm to finish."""
    from association.models.query_relation import QueryRelationPosition, initialize_from_reference
    from association.training.query_relation import load_checkpoint, source_configuration
    from association.training.sequence_position import load_checkpoint as load_reference

    old, old_rows, old_exposure = _sequence_sampling(reference/'fit')
    run, rows, exposure = _sequence_sampling(arm['path']/'fit')
    config = run['configuration']
    samples = lambda items: [{key: item[key] for key in ('step', 'mode', 'batch')} for item in items]
    if (source_configuration(config) != old['configuration'] or samples(rows) != samples(old_rows)
            or exposure != old_exposure or run['training_scene_ids'] != old['training_scene_ids']):
        raise ValueError('099 ours 098 arm source/batch/exposure pairing differs')
    _, original_initial = load_reference(reference/'fit/step-0.pt')
    _, original_final = load_reference(reference/'fit/final.pt')
    snapshots = []
    for filename, step in (('step-0.pt', 0), ('final.pt', config['training']['updates'])):
        _, saved = load_checkpoint(arm['path']/'fit'/filename)
        source = saved['source_teacher']
        if (saved['configuration'] != config or saved['step'] != step
                or Path(source['checkpoint']).resolve() != reference/'fit/final.pt'
                or source['configuration'] != old['configuration'] or source['step'] != original_final['step']
                or not _exact(source['model'], original_final['model'])):
            raise ValueError('099 ours 098 checkpoint source-final binding differs')
        snapshots.append(saved)
    initial, final = snapshots
    with torch.random.fork_rng(devices=[]):
        expected = QueryRelationPosition(**config['model'])
        initialize_from_reference(expected, original_final['model'], branch_seed=config['training']['seed']+98000)
    if (not _exact(initial['model'], expected.state_dict())
            or initial['numpy_rng'] != original_initial['numpy_rng']
            or final['numpy_rng'] != original_final['numpy_rng']):
        raise ValueError('099 ours 098 initialization/sampler RNG differs')
    return dict(status='PAIR_VERIFIED', actual_successful_batches_exact=True,
        source_final_parameters_exact=True, sampling_exposure_exact=True,
        source_updates=original_final['step'], added_updates=len(rows),
        seed=config['training']['seed'], source_fit=str(reference/'fit'),
        interpretation='One 098 arm checked; the separate 098 two-seed mechanism gate is not evaluated here.')


def ours_metadata(path):
    path = Path(path).resolve()
    config = read_json(path/'run.json')['configuration']
    seed = config['training']['seed']
    if seed not in (20261023, 20261021):
        raise ValueError('099 ours run must use a declared paired seed')
    if config['model'].get('mode') == 'full_finetune':
        name, arm = budget_compare.adapter_metadata(path, profile='090' if seed == 20261023 else '097')
        if name != 'finetune':
            raise ValueError('099 budget control must be full finetune')
        reference = arm['paired_control']
        source_name = '072C_transformer4000' if seed == 20261023 else '073C_transformer4000_seed21'
        source = _metadata(reference, source_name)
        pairing = budget_compare.verify_pairing(dict(linear=source, finetune=arm))
        method_name = '090C' if seed == 20261023 else '097A'
    elif config['model'].get('mode') in ('generic', 'relation'):
        name, arm = relation_compare.relation_metadata(path)
        reference = arm['paired_control']
        source_name = '072C_transformer4000' if seed == 20261023 else '073C_transformer4000_seed21'
        source = _metadata(reference, source_name)
        pairing = _verify_relation_arm(arm, reference)
        method_name = '098_'+name
    else:
        raise ValueError('099 accepts 090C/097A or an actual completed 098 arm as ours')
    if any(arm[key] != source[key] for key in ('source_system', 'scene_ids', 'components')):
        raise ValueError('099 ours source-final and evaluated populations differ')
    return dict(arm, method_name=method_name, reference_endpoint=reference/'final', pairing=pairing)


def scene_pair(arms, sid):
    """Recompute errors from actual predictions after exact population checks."""
    first = next(iter(arms.values()))
    reference, original_system = _cache(first.get('reference_endpoint', first['path']/'final'), sid)
    reference['group_eligible'] = _visible_opportunity(reference)
    base = _artifact(original_system, 'anchor')
    _, _, regions, population = _structure(reference, base)
    data = {'region_'+key: value for key, value in regions.items()}
    for method in FIXED_METHODS:
        data[method+'_position_error'] = _check_artifact(reference, _artifact(original_system, method), base, method)
    common = np.isfinite(data['anchor_position_error'])
    original = _artifact(original_system, 'learned_sequence')
    chart_counts = {}
    for name, arm in arms.items():
        fragment, system = _cache(arm['path']/'final', sid)
        for key in INPUT_KEYS:
            exact(fragment[key], reference[key], '099 identical input/'+key)
        for region in REGIONS:
            exact(fragment['region_'+region], data['region_'+region], '099 identical region/'+region)
        for method in FIXED_METHODS:
            budget_compare._fixed_artifact(_artifact(system, method), _artifact(original_system, method),
                '099 unchanged ordinary control/'+method, predictions=True)
            _check_artifact(fragment, _artifact(system, method), base, method)
        artifact = _artifact(system, 'learned_sequence')
        budget_compare._fixed_artifact(artifact, original, '099 identical target/population/masks')
        error = _check_artifact(fragment, artifact, base, 'learned_sequence')
        exact(np.isfinite(error), common, '099 full common valid position population')
        exact(artifact['group_states'][..., 2:], _artifact(system, 'mean_supported')['group_states'][..., 2:],
            '099 unchanged Mean speed/heading')
        if arm.get('published_method') == 'DeepAF':
            positions = fragment['group_deepaf_positions_m']
            if positions.shape != artifact['group_states'].shape[:2]+(2,) or not np.isfinite(positions).all():
                raise ValueError('099 peer local-position diagnostic has invalid shape/values')
            actual_counts = (np.linalg.norm(positions, axis=-1) >= np.pi*6371000.).sum(-1)
            exact(fragment['group_deepaf_chart_excursion_count'], actual_counts,
                '099 peer independently recomputed finite-chart excursions')
            chart_counts[name] = int(actual_counts.sum())
        data[name+'_position_error'] = error
    count = len(base['target_ids'])
    available = base['aligned_truth_valid'][:count, :, :2].all(-1)
    covered = available & base['aligned_prediction_valid'][:count, :, :2].all(-1)
    return dict(scene_id=str(sid), population=population, chart_excursions=chart_counts, position_population=dict(
        available_target_points=int(available.sum()), covered_target_points=int(covered.sum()),
        missing_prediction_points=int((available & ~covered).sum()),
        available_target_trajectories=int(available.any(-1).sum()),
        covered_target_trajectories=int(covered.any(-1).sum()))), data


def cached_comparison(arms, *, progress=None):
    first = next(iter(arms.values()))
    for arm in arms.values():
        if any(arm[key] != first[key] for key in ('source_system', 'scene_ids', 'components', 'paired_control')):
            raise ValueError('099 methods differ in source, full population, components or paired seed source')
    methods = (*FIXED_METHODS, *arms)
    rows, arrays = [], []
    for sid in first['scene_ids']:
        row, data = scene_pair(arms, sid)
        row['component_id'] = first['components'][sid]
        rows.append(row); arrays.append(data)
        if progress:
            progress(len(rows))
    if not rows:
        raise ValueError('099 empty evaluation population cannot establish a peer comparison')
    result = _pool(rows, arrays, methods)
    for name, arm in arms.items():
        actual = result['position_weightings']['all'][name]['point_equal']
        expected = arm['summary']['methods']['learned_sequence']['full_target_metrics']['all']['position']
        if actual['unit_count'] != expected['intersection_count'] or any(
                not np.isclose(actual[key], expected[key], rtol=1e-10, atol=1e-9) for key in ('mae', 'rmse')):
            raise ValueError('099 actual cache fails to reproduce producer metrics/'+name)
        if arm.get('published_method') == 'DeepAF' and sum(row['chart_excursions'][name] for row in rows) != arm['summary']['chart_excursion_count']:
            raise ValueError('099 cached chart excursions disagree with producer diagnostics')
    components = {}
    for cid in sorted(set(first['components'].values())):
        indices = [i for i, row in enumerate(rows) if row['component_id'] == cid]
        components[cid] = _pool([rows[i] for i in indices], [arrays[i] for i in indices], methods)
    largest = min(components, key=lambda cid: (-components[cid]['position_population']['covered_target_points'], cid))
    rest = [i for i, row in enumerate(rows) if row['component_id'] != largest]
    scene_rows = [dict(_pool([row], [data], methods), component_id=row['component_id'])
        for row, data in zip(rows, arrays)]
    result.update(methods=list(methods), components=components, largest_component_id=largest,
        remaining_components=_pool([rows[i] for i in rest], [arrays[i] for i in rest], methods) if rest else None,
        source_system=first['source_system'], inputs={name: str(arm['path']) for name, arm in arms.items()},
        population_and_masks_exact=True, errors_recomputed_from_predictions=True,
        speed_heading_equal_mean=True, source_sampling_control=str(first['paired_control']),
        chart_excursions={name: sum(row['chart_excursions'][name] for row in rows)
            for name, arm in arms.items() if arm.get('published_method') == 'DeepAF'})
    result['component_equal_mae'] = {name: float(np.mean([
        item['position_weightings']['all'][name]['point_equal']['mae']
        for item in components.values() if item['position_weightings']['all'][name]['point_equal']['unit_count'] > 0
    ])) for name in methods}
    return result, scene_rows


def numerical_comparison(summary, scenes):
    """Describe this seed only; scientific peer qualification remains external."""
    table = summary['position_weightings']['all']
    ours, peer = table['ours']['point_equal'], table['deepaf']['point_equal']
    fraction = 1-ours['mae']/peer['mae'] if peer['mae'] > 0 else None
    improves = lambda item: (item['position_weightings']['all']['ours']['point_equal']['mae']
        < item['position_weightings']['all']['deepaf']['point_equal']['mae']-EPS_M)
    rest = summary['remaining_components']
    excursions = summary.get('chart_excursions', {}).get('deepaf', 0)
    return dict(relative_mae_reduction=fraction,
        delta_ours_minus_peer={key: ours[key]-peer[key] for key in ('mae', 'rmse', 'p95', 'p99')},
        mae_reduction_at_least_three_percent=bool(fraction is not None and fraction >= .03),
        rmse_p99_nonworse=bool(all(ours[key] <= peer[key]+EPS_M for key in ('rmse', 'p99'))),
        all_mae_weightings_nonworse=bool(all(table['ours'][key]['mae'] <= table['deepaf'][key]['mae']+EPS_M
            for key in WEIGHTINGS)), improved_scenes=sum(improves(item) for item in scenes),
        total_scenes=len(scenes), improved_components=sum(improves(item) for item in summary['components'].values()),
        total_components=len(summary['components']), outside_largest_component_improved=bool(rest and improves(rest)),
        peer_quality_certified=False, stable_peer_superiority_established=False,
        peer_chart_excursion_count=excursions,
        scientific_decision=('UNRESOLVED_PEER_GEOMETRY' if excursions else 'REQUIRES_PEER_QUALITY_AND_MULTISEED_REVIEW'),
        old098_gates_evaluated=False, default_promoted=False, test_opened=False)


def select_learning_rate(peer_a, peer_b, *, progress=None):
    """Read-only selection; also called by the second-seed runner to reverify."""
    paths = [Path(peer_a).resolve(), Path(peer_b).resolve()]
    if paths[0] == paths[1]:
        raise ValueError('099 selection requires distinct A/B runs')
    arms = {name: peer_metadata(path) for name, path in zip(('peer_a', 'peer_b'), paths)}
    a, b = arms.values()
    for arm, lr in ((a, .001), (b, .0003)):
        training = arm['configuration']['training']
        if training['seed'] != 20261023 or training['learning_rate'] != lr or training['updates'] != 8000:
            raise ValueError('099 selection must bind A=.001/B=.0003, seed20261023, final8000')
    ac, bc = deepcopy(a['configuration']), deepcopy(b['configuration'])
    ac['training'].pop('learning_rate'); bc['training'].pop('learning_rate')
    if (ac != bc or not _exact(a['initial_state'], b['initial_state']) or a['initial_rng'] != b['initial_rng']
            or a['pairing']['source_fit'] != b['pairing']['source_fit']
            or a['pairing']['exposure'] != b['pairing']['exposure']):
        raise ValueError('099 learning-rate arms differ beyond learning rate or actual paired initialization/sampling')
    comparison, scenes = cached_comparison(arms, progress=progress)
    metrics = comparison['position_weightings']['all']
    candidates = []
    for name, arm in arms.items():
        item = metrics[name]['point_equal']
        if item['unit_count'] <= 0 or not np.isfinite(item['mae']):
            raise ValueError('099 learning-rate selection requires a nonempty finite common population')
        candidates.append(dict(label=name, learning_rate=arm['configuration']['training']['learning_rate'],
            peer_path=str(arm['path']), config_name=arm['config_name'], configuration=arm['configuration'],
            metrics={key: item[key] for key in ('mae', 'rmse', 'p95', 'p99', 'unit_count')}))
    winner = min(candidates, key=lambda row: (row['metrics']['mae'], 0 if row['learning_rate'] == .001 else 1))
    return dict(scope=SELECTION_SCOPE, status='COMPLETED', selection_rule=SELECTION_RULE, seed=20261023,
        selected_learning_rate=winner['learning_rate'],
        selected=dict(learning_rate=winner['learning_rate'], peer_path=winner['peer_path'],
            config_name=winner['config_name'], seed=20261023, endpoint='final', step=8000),
        candidates=candidates, source_system=a['source_system'], scene_ids=a['scene_ids'],
        source_runs={name: str(arm['path']) for name, arm in arms.items()},
        source_sampling_control=str(a['paired_control']),
        pairing={name: arm['pairing'] for name, arm in arms.items()}, initial_parameters_exact=True,
        model_forwards=0, optimizer_updates=0, test_opened=False, default_promoted=False,
        peer_quality_certified=False, stable_peer_superiority_established=False,
        geometry_review_required=bool(any(comparison.get('chart_excursions', {}).values())),
        comparison=comparison, scene_summary=scenes,
        interpretation='Only the lower fixed-final peer MAE selects the learning rate; quality, convergence and superiority over published methods require separate review.')


def _report(summary, *, selection):
    table = summary['comparison'] if selection else summary
    lines = ['# 099 DeepAF 同人口缓存比较', '',
        '| 方法 | MAE m | RMSE m | P95 m | P99 m | 有效位置点 |',
        '|---|---:|---:|---:|---:|---:|']
    for name in table['methods']:
        metric = table['position_weightings']['all'][name]['point_equal']
        lines.append('| '+name+' | '+' | '.join(f'{metric[key]:.6f}' for key in ('mae', 'rmse', 'p95', 'p99'))
            +' | '+str(metric['unit_count'])+' |')
    if selection:
        lines += ['', '按预定fixed-final全人口点MAE选择学习率：'+str(summary['selected_learning_rate'])+'。',
            '严格数值平手选择0.001；没有按我方是否胜出选择较弱peer，没有从mid选checkpoint。']
    else:
        lines += ['', '配对seed：'+str(summary['seed'])+'；我方：'+summary['ours_method']+'。']
    lines += ['', '本表从实际缓存逐值核对输入、组、GT和mask，再重算Haversine误差；零前向、零训练。',
        '这里只报告数值与数据一致性，不自动认证同行训练质量、不宣布稳定胜过同行、不改变098合同或默认模型。',
        'DeepAF为完整P/V的fixed056离线位置融合适配；速度、航向仍为共同Mean。',
        'V反复开发且身份跨split共享；逐场、组件及最大组件外数据见JSON，不能将位置点当独立样本。']
    if any(table.get('chart_excursions', {}).values()):
        lines += ['存在局部预测越过πR的球面坐标非单射区间；peer几何质量待审查，不能因折回后的米制误差较小自动判定合格。']
    return '\n'.join(lines)+'\n'


def run(args):
    if args.cpu_threads != 4:
        raise ValueError('099 cached comparison requires CPU4')
    torch.set_num_threads(4)
    selection = bool(getattr(args, 'select_lr', False))
    keys = ('peer_a', 'peer_b') if selection else ('ours_run', 'peer_run')
    other = ('ours_run', 'peer_run') if selection else ('peer_a', 'peer_b')
    if any(not getattr(args, key, None) for key in keys) or any(getattr(args, key, None) for key in other):
        raise ValueError('099 specify exactly one comparison mode and its two required runs')
    paths = [Path(getattr(args, key)).resolve() for key in keys]
    if len(set(paths)) != 2:
        raise ValueError('099 requires two distinct completed runs')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('099 comparison exists; no overwrite/retry')
    for path in paths:
        source = Path(read_json(path/'run.json')['source_system']).resolve().parent
        for protected in (path, source):
            if output == protected or output in protected.parents or protected in output.parents:
                raise ValueError('099 output overlaps an immutable source')
    output.mkdir(parents=True)
    state = dict(scope=SELECTION_SCOPE if selection else SCOPE, status='RUNNING',
        model_forwards=0, optimizer_updates=0, test_opened=False, default_promoted=False, completed_scenes=0)
    write_json(output/'run.json', state)

    def progress(count):
        state['completed_scenes'] = count
        write_json(output/'run.json', state)

    try:
        if selection:
            summary = select_learning_rate(*paths, progress=progress)
            scenes = summary['scene_summary']
        else:
            ours, peer = ours_metadata(paths[0]), peer_metadata(paths[1])
            seed = peer['configuration']['training']['seed']
            if ours['configuration']['training']['seed'] != seed:
                raise ValueError('099 ours and peer must have the same declared seed')
            summary, scenes = cached_comparison(dict(ours=ours, deepaf=peer), progress=progress)
            summary.update(scope=SCOPE, status='COMPLETED', seed=seed, ours_method=ours['method_name'],
                peer_config_name=peer['config_name'], peer_learning_rate=peer['configuration']['training']['learning_rate'],
                pairing=dict(ours=ours['pairing'], peer=peer['pairing']),
                model_forwards=0, optimizer_updates=0, test_opened=False, default_promoted=False,
                peer_quality_certified=False, stable_peer_superiority_established=False)
            summary['numerical_comparison'] = numerical_comparison(summary, scenes)
        write_json_gzip(output/'summary.json.gz', summary)
        write_json_gzip(output/'scene-summary.json.gz', scenes)
        (output/'report.md').write_text(_report(summary, selection=selection))
        if selection:
            write_json(output/'selection.json', summary)
        state.update(status='COMPLETED', seed=summary['seed'], peer_quality_certified=False,
            stable_peer_superiority_established=False)
        if selection:
            state['selected_learning_rate'] = summary['selected_learning_rate']
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--select-lr', action='store_true')
    for name in ('ours-run', 'peer-run', 'peer-a', 'peer-b'):
        parser.add_argument('--'+name)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    result = run(parser.parse_args(argv))
    print(result['status'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
