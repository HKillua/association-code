"""070 frozen068 binding and paired CPU-cache comparisons.

The old run is read only. Its initial output is reused only after reconstructing
the constant .95 gate and the actual geometric mixing operation. No model is
loaded or forwarded here, and the original failed066 acceptance is unchanged.
"""
from __future__ import annotations

import math
import gzip
import json
from pathlib import Path

import torch

from association.evaluation.support_gate import compare_endpoints
from association.evaluation.support_gate_objectives import (
    _read, load_scene, summarize, verify_source_summaries,
)
from association.evaluation.system_attribution import _same
from association.models.support_gate import mix_gate_states
from association.utils.runtime import read_json, write_json_gzip

# Shared source binding retained after retiring the standalone 069 diagnostic.
ROOT = Path(__file__).resolve().parents[2]
TASK = 'LEVEL1-SUPPORT-GATE-20261023-068-attempt-1'
METHODS = ['anchor', 'mean_supported', 'attribute_hybrid', 'support_switch', 'learned_gate']


def read(path):
    path = Path(path)
    if not path.exists():
        path = path.with_suffix(path.suffix + '.gz')
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8')) as stream:
        return json.load(stream)


def bind_source(source):
    """Bind existing records and inventory, never deserialize a checkpoint."""
    if source.name != TASK:
        raise ValueError('069 requires the original068 attempt-1 directory')
    config = read(ROOT/'configs/support-gate-level1.json')
    manifest = read(ROOT/'configs/data/mtad-level1-scene3000.json')
    ids = manifest['splits']['V_select']
    components = {sid: str(manifest['components'][sid]) for sid in ids}
    run, summary = read(source/'run.json'), read(source/'summary.json')
    for record in (run, summary):
        if (record.get('status') != 'COMPLETED'
                or record.get('scope') != '068-support-prior-position-pilot'
                or record.get('configuration') != config or record.get('optimizer_updates') != 1000
                or record.get('test_opened') is not False or record.get('inference_device') != 'cpu'):
            raise ValueError('068 completed fixed-1000 CPU endpoint protocol required')
    parent = Path(run.get('source_system', ''))
    if (summary.get('source_system') != str(parent) or parent.name != 'system'
            or parent.parent.name != 'LEVEL1-ASSOCIATION-20261022-056-attempt-1'
            or run.get('completed_endpoints') != ['initial', 'final']
            or run.get('old066_acceptance') != 'REJECTED' or run.get('retries') != 0
            or any(run.get(key) is not False for key in ('association_recomputed',
                'threshold_reselected', 'validation_groups_recomputed', 'validation_selection'))):
        raise ValueError('068 immutable056 provenance/selection boundary differs')
    preflight = read(source/'preflight/report.json')
    if (preflight.get('status') != 'PASS' or preflight.get('requested_scene_ids') != ['3035', '2535']
            or preflight.get('training_gate_passed') is not True
            or preflight.get('data_contract_reproduced') is not True
            or preflight.get('anomalies_removed') is not False):
        raise ValueError('068 original two-scene preflight required')
    endpoints = {}
    expected_inventory = {f'states-{sid}.npz' for sid in ids}
    for label, step, filename in (('initial', 0, 'step-0.pt'), ('final', 1000, 'final.pt')):
        endpoint = source/label
        report, record = read(endpoint/'summary.json'), read(endpoint/'run.json')
        checkpoint = dict(path=str(source/'fit'/filename), step=step, actual_updates=step,
            configured_final_updates=1000, routing_semantics=config['routing_semantics'])
        for item in (report, record):
            if (item.get('status') != 'COMPLETED' or item.get('device') != 'cpu'
                    or item.get('source_system') != str(parent) or item.get('scene_ids') != ids
                    or item.get('checkpoint') != checkpoint or item.get('test_opened') is not False
                    or item.get('optimizer_updates') != 0
                    or any(item.get(key) is not False for key in
                           ('groups_recomputed', 'scores_recomputed', 'threshold_reselected'))):
                raise ValueError('068 endpoint provenance differs: ' + label)
        if (report.get('method_names') != METHODS or report.get('scene_count') != 300
                or report.get('component_count') != 38 or report.get('fragment_count') != 24599
                or report.get('dataset_level') != 'Level1' or report.get('split') != 'V_select'
                or report.get('ordinary_source_replay_exact') is not True
                or report.get('fixed_population_verified') is not True
                or record.get('completed_scenes') != 300 or record.get('total_scenes') != 300
                or {sid: cid for cid, part in report['components'].items() for sid in part['scene_ids']} != components):
            raise ValueError('068 endpoint full V300 population differs: ' + label)
        for directory in ('predictions', 'system-predictions'):
            if {path.name for path in (endpoint/directory).glob('*.npz')} != expected_inventory:
                raise ValueError('068 missing/extra cached scenes: ' + str(endpoint/directory))
        endpoints[label] = report
    fit = read(source/'fit/run.json')
    if (fit.get('configuration') != config or fit.get('training_scene_ids') != manifest['splits']['T_core']
            or fit.get('test_opened') is not False):
        raise ValueError('068 training config/T2100 registry differs')
    return dict(scene_ids=ids, components=components, endpoints=endpoints, source_run=run)


def _verify_route(directory, sid, *, initial=False):
    """Bind stored logits, gates and every group state, including duplicates."""
    data = _read(Path(directory)/'predictions'/f'states-{sid}.npz')
    logits = torch.as_tensor(data['group_logits'], dtype=torch.float32)
    _same(data['group_logits'], logits.numpy(), '070 original float32 logits')
    if initial:
        expected = torch.full_like(logits, math.log(19.))
        _same(logits.numpy(), expected.numpy(), '070 immutable .95 initial logits')
    for index in range(len(logits)):
        # Producer evaluates one original representative with a Q-vector;
        # reproduce that operation shape, without forwarding a network.
        gate = torch.sigmoid(logits[index])
        _same(data['group_gate'][index], gate.numpy(), '070 saved sigmoid route')
        actual = mix_gate_states(
            torch.as_tensor(data['anchor_group_states'][index], dtype=torch.float64),
            torch.as_tensor(data['mean_supported_group_states'][index], dtype=torch.float64),
            gate, torch.as_tensor(data['group_eligible'][index], dtype=torch.bool))
        _same(data['learned_gate_group_states'][index], actual.numpy(),
              '070 saved gate/state mixing relation')
    for attr in ('speed', 'heading'):
        _same(data['learned_gate_'+attr+'_error'], data['mean_supported_'+attr+'_error'],
              '070 fixed Mean physical errors/'+attr)


def bind_linear_source(source068, source056, config070, manifest, *, check=None):
    """Validate the completed original run and its reusable CPU endpoints.

    Production metadata and inventory are the same frozen068 contract used by
    069. Actual errors, population, routes and geometric output are independently
    reconstructed before training, rather than trusting the aggregate summary.
    """
    source, parent = Path(source068).resolve(), Path(source056).resolve()
    binding = bind_source(source)
    old = binding['source_run']['configuration']
    expected = dict(old, position_loss='linear')
    if config070 != expected:
        raise ValueError('070 may change only position_loss to linear')
    if (str(parent) != binding['source_run']['source_system']
            or binding['scene_ids'] != manifest['splits']['V_select']
            or binding['components'] != {sid: str(manifest['components'][sid])
                                        for sid in binding['scene_ids']}):
        raise ValueError('070 immutable056 source/manifest differs from068')
    fit = read_json(source/'fit/run.json')
    updates = old['training']['updates']
    if (fit.get('status') != 'COMPLETED' or fit.get('optimizer_updates') != updates
            or fit.get('configuration') != old or fit.get('test_opened') is not False
            or fit.get('loss_semantics') != 'mean_valid_position_log1p_haversine_m_div_1000_all_queries_no_auxiliary'
            or fit.get('training_scene_ids') != manifest['splits']['T_core']
            or fit.get('exposure', {}).get('successful_updates') != updates
            or fit.get('final_checkpoint') != 'final.pt'
            or not (source/'fit/step-0.pt').is_file()
            or not (source/'fit/final.pt').is_file()):
        raise ValueError('070 requires completed original068 fit and endpoint checkpoint records')
    rows = []
    for sid in binding['scene_ids']:
        if check:
            check()
        rows.append(load_scene(source/'initial', source/'final', sid))
        _verify_route(source/'initial', sid, initial=True)
        _verify_route(source/'final', sid)
    verified = verify_source_summaries(rows, binding['endpoints']['initial'], binding['endpoints']['final'])
    return dict(binding, source_068=str(source), source_system=str(parent),
                configuration=config070, original_configuration=old,
                initial_cache_reused=True, initial_route_and_mix_exact=True,
                model_forwards=0, checkpoint_loaded=False, **verified)


def _final_record(final, binding):
    config = binding['configuration']
    steps = config['training']['updates']
    fit = read_json(final.parent/'fit/run.json')
    pairing = fit.get('pairing', {})
    if (fit.get('status') != 'COMPLETED' or fit.get('configuration') != config
            or fit.get('optimizer_updates') != steps or fit.get('test_opened') is not False
            or fit.get('loss_semantics') != 'mean_valid_position_linear_haversine_m_div_1000_all_queries_no_auxiliary'
            or fit.get('exposure', {}).get('successful_updates') != steps
            or fit.get('final_checkpoint') != 'final.pt'
            or pairing.get('initial_parameters_exact') is not True
            or pairing.get('ordered_training_scene_ids_exact') is not True
            or pairing.get('source_successful_updates') != steps
            or pairing.get('matched_updates') != steps
            or pairing.get('matched_anchors') != steps*config['training']['batch_size']
            or pairing.get('reference_fit') != str(Path(binding['source_068'])/'fit')):
        raise ValueError('070 comparison requires completed matched linear training')
    expected_checkpoint = dict(path=str(final.parent/'fit/final.pt'), step=steps, actual_updates=steps,
        configured_final_updates=steps, routing_semantics=config['routing_semantics'])
    records = [read_json(final/name) for name in ('run.json', 'summary.json')]
    for record in records:
        checkpoint = record.get('checkpoint', {})
        if (record.get('status') != 'COMPLETED' or record.get('device') != 'cpu'
                or record.get('configuration') != config or record.get('position_loss') != 'linear'
                or any(checkpoint.get(key) != value for key, value in expected_checkpoint.items())
                or checkpoint.get('loss_semantics') != 'mean_valid_position_linear_haversine_m_div_1000_all_queries_no_auxiliary'
                or record.get('source_system') != binding['source_system']
                or record.get('scene_ids') != binding['scene_ids']
                or record.get('test_opened') is not False
                or record.get('optimizer_updates') != 0
                or any(record.get(key) is not False for key in
                       ('groups_recomputed', 'scores_recomputed', 'threshold_reselected'))):
            raise ValueError('070 final CPU endpoint/checkpoint provenance differs')
    expected_inventory = {f'states-{sid}.npz' for sid in binding['scene_ids']}
    for directory in ('predictions', 'system-predictions'):
        if {p.name for p in (final/directory).glob('*.npz')} != expected_inventory:
            raise ValueError('070 final cache scene inventory differs')
    return records[1]


def _component_split(rows, report, components):
    # The split is fixed by covered population, not by method performance.
    largest = min(report['components'], key=lambda key: (
        -report['components'][key]['regions']['all']['point_count'], key))
    remaining = [row for row in rows if components[row['scene_id']] != largest]
    rest = summarize(remaining, {r['scene_id']: components[r['scene_id']] for r in remaining}) if remaining else None
    return dict(selection='Largest component by original covered position-point count; lexical tie break.',
        largest_component_id=largest, largest=report['components'][largest],
        remaining=(dict(scene_count=rest['scene_count'], scene_ids=rest['scene_ids'],
                        component_count=rest['component_count'], regions=rest['regions']) if rest else None),
        interpretation='Descriptive shared-identity components; not independent training replications.')


def compare_linear_run(source068, final070, binding, output=None, *, check=None):
    """Compare final070 against both frozen068 endpoints on identical target rows.

    Each report uses the established objective helper's initial/final column
    names, with explicit labels indicating which immutable control is initial.
    All four weightings and support strata are retained. No source is modified.
    """
    source, final = Path(source068).resolve(), Path(final070).resolve()
    if str(source) != binding['source_068']:
        raise ValueError('070 comparison source differs from pretraining binding')
    final_summary = _final_record(final, binding)
    result = dict(status='COMPLETED', comparison_scope='070-linear-loss-vs-frozen068',
        source_068=str(source), final070=str(final), model_forwards=0,
        population_and_masks_exact=True, old066_acceptance='REJECTED', test_opened=False,
        comparisons={}, initial_cache_reused=True)
    scene_reports = []
    for label, directory in (('vs_initial', 'initial'), ('vs_log_final', 'final')):
        rows = []
        for sid in binding['scene_ids']:
            if check:
                check()
            rows.append(load_scene(source/directory, final, sid))
            if label == 'vs_initial':
                _verify_route(final, sid)
        verify_source_summaries(rows, binding['endpoints'][directory], final_summary)
        objective = summarize(rows, binding['components'])
        component_split = _component_split(rows, objective, binding['components'])
        scene_reports.extend(dict(comparison=label, **row) for row in objective.pop('per_scene'))
        result['comparisons'][label] = dict(
            labels=dict(initial='068_fixed_0.95_initial' if directory == 'initial' else '068_log_final1000',
                        final='070_linear_final1000'),
            objectives=objective, largest_component_vs_remaining=component_split,
            physical=compare_endpoints(source/directory, final))
    if output is not None:
        output = Path(output).resolve()
        for protected in (source, Path(binding['source_system']).parent):
            if output == protected or protected in output.parents or output in protected.parents:
                raise ValueError('070 reports must not overwrite immutable sources')
        output.mkdir(parents=True, exist_ok=True)
        paths = [output/'objective-summary.json.gz', output/'scene-summary.json.gz']
        if any(path.exists() for path in paths):
            raise FileExistsError('070 comparison reports already exist; no overwrite')
        write_json_gzip(paths[0], result)
        write_json_gzip(paths[1], scene_reports)
        result['report_paths'] = [str(path) for path in paths]
    else:
        result['per_scene'] = scene_reports
    return result
