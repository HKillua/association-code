"""071/072 CPU closeout: actual-cache comparisons of frozen position methods.

No inference, training, archive access, threshold selection, or test access.
Every output group is checked before unique-target rows are evaluated.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.sequence_position import _weighted, REGIONS
from association.evaluation.support_gate_linear import _verify_route
from association.evaluation.support_gate_objectives import _read, _artifact, _structure, _check_artifact
from association.evaluation.support_gate_replication import SAMPLE_KEYS, verify_pairing
from association.evaluation.system_attribution import _same
from association.training.evidence_fusion import TRAIN_MODES
from association.training.support_gate import load_checkpoint as gate_checkpoint
from association.training.sequence_position import load_checkpoint as sequence_checkpoint
from association.utils.runtime import read_json, write_json, write_json_gzip

ROOT = Path(__file__).resolve().parents[2]
CONTROLS = ('anchor', 'mean_supported', 'attribute_hybrid', 'support_switch')
ARM_CONFIGS = {
    '071A_log': 'support-gate-replica-log-level1.json',
    '071B_linear': 'support-gate-replica-linear-level1.json',
    '071C_lstm': 'sequence-position-lstm-level1.json',
    '071D_transformer': 'sequence-position-transformer-level1.json',
    '070_linear': 'support-gate-linear-level1.json',
}
EXTENDED_CONFIGS = {
    '072B_lstm4000': 'sequence-position-lstm-extended-level1.json',
    '072C_transformer4000': 'sequence-position-transformer-extended-level1.json',
}
REPLICATION_CONFIGS = {
    '073A_gate4000_seed23': 'support-gate-linear-extended-level1.json',
    '073B_gate4000_seed21': 'support-gate-linear-extended-replica-level1.json',
    '073C_transformer4000_seed21': 'sequence-position-transformer-extended-replica-level1.json',
}
CONSTANT_METHOD = '072A_constant'
OPTIONAL_INPUTS = ('constant_run072A', 'lstm_run072B', 'transformer_run072C')
INPUT_KEYS = ('node_ids', 'identities', 'target_identity_ids', 'native_states', 'support',
              'nearest_seconds', 'exact_observation', 'query_seconds', 'learned_assoc_group_offsets',
              'learned_assoc_group_node_indices', 'group_anchor_node_indices')
METHODS = (*CONTROLS, 'fixed_prior095', *ARM_CONFIGS)


def _sequence_sampling(path):
    """Reconstruct exposure only from logged successful optimizer updates."""
    path = Path(path)
    run = read_json(path/'run.json')
    config = run['configuration']
    steps, size = config['training']['updates'], config['training']['batch_size']
    scenes = run.get('training_scene_ids', [])
    if (run.get('status') != 'COMPLETED' or run.get('optimizer_updates') != steps
            or run.get('test_opened') is not False or not scenes or len(set(scenes)) != len(scenes)):
        raise ValueError('sequence pairing requires completed actual updates and ordered training scenes')
    rows = [json.loads(line) for line in (path/'training.jsonl').read_text().splitlines()]
    if len(rows) != steps:
        raise ValueError('sequence pairing requires every actual successful batch')
    exposure = dict(successful_updates=0, sampled_anchors=0, valid_position_points=0, scenes={}, anchors={},
        modes={mode: dict(updates=0, anchors=0, valid_position_points=0) for mode in TRAIN_MODES})
    eligibility = dict(eligible_query_points=0, eligible_valid_position_points=0)
    for step, row in enumerate(rows, 1):
        mode = TRAIN_MODES[(step-1) % len(TRAIN_MODES)]
        if (row.get('step') != step or row.get('mode') != mode or len(row.get('batch', [])) != size
                or not np.isfinite(row.get('loss', np.nan)) or not np.isfinite(row.get('gradient_norm', np.nan))):
            raise ValueError('sequence successful update order/mode/batch is invalid')
        exposure['successful_updates'] += 1
        exposure['sampled_anchors'] += size
        exposure['modes'][mode]['updates'] += 1
        exposure['modes'][mode]['anchors'] += size
        for sample in row['batch']:
            if (set(sample) != SAMPLE_KEYS or sample['scene_id'] not in scenes
                    or type(sample['anchor_index']) is not int or sample['anchor_index'] < 0
                    or not isinstance(sample['node_id'], str) or not sample['node_id']
                    or not sample['members'] or sample['members'][0] != sample['anchor_index']
                    or len(set(sample['members'])) != len(sample['members'])
                    or any(type(x) is not int or x < 0 for x in sample['members'])):
                raise ValueError('sequence actual scene/anchor/member record is invalid')
            for key in (*eligibility, 'valid_position_points'):
                if type(sample[key]) is not int or not 0 <= sample[key] <= config['data']['length']:
                    raise ValueError('sequence sample count exceeds fixed query population')
            if sample['eligible_valid_position_points'] > min(sample['eligible_query_points'], sample['valid_position_points']):
                raise ValueError('sequence eligible valid points exceed sampled population')
            for key in eligibility:
                eligibility[key] += sample[key]
            exposure['valid_position_points'] += sample['valid_position_points']
            exposure['modes'][mode]['valid_position_points'] += sample['valid_position_points']
            for key, name in (('scenes', sample['scene_id']), ('anchors', sample['node_id'])):
                exposure[key][name] = exposure[key].get(name, 0) + 1
    # Gradient counts are properties of different backbones, not sampling.
    actual = run.get('exposure', {})
    if any(actual.get(key) != value for key, value in exposure.items()):
        raise ValueError('sequence successful batches do not reproduce recorded sampling exposure')
    return run, rows, dict(exposure, **eligibility)


def verify_sequence_sampling(lstm_fit, transformer_fit, gate_fit=None):
    a, ar, ae = _sequence_sampling(lstm_fit)
    b, br, be = _sequence_sampling(transformer_fit)
    ca, cb = deepcopy(a['configuration']), deepcopy(b['configuration'])
    if ca['model'].pop('backbone') != 'lstm' or cb['model'].pop('backbone') != 'transformer':
        raise ValueError('sequence pairing must compare the declared two backbones')
    if ca != cb or a['training_scene_ids'] != b['training_scene_ids']:
        raise ValueError('sequence configs may differ only in backbone; training scenes must agree')
    sampled = lambda rows: [{key: row[key] for key in ('step', 'mode', 'batch')} for row in rows]
    if sampled(ar) != sampled(br) or ae != be:
        raise ValueError('sequence actual successful batch pairing differs')
    cross_gate = None
    if gate_fit is not None:
        # Gate exposure has extra opportunity/gradient counters. Compare only
        # actual common batch fields; do not equate incompatible aggregates.
        gate_rows = [json.loads(line) for line in (Path(gate_fit)/'training.jsonl').read_text().splitlines()]
        cross_gate = sampled(ar) == sampled(gate_rows)
    return dict(status='PAIR_VERIFIED', actual_successful_batches_exact=True,
        ordered_training_scene_ids_exact=True, sampling_exposure_exact=True,
        initial_parameters_compared=False, matched_updates=len(ar), matched_anchors=ae['sampled_anchors'],
        seed=ca['training']['seed'], exposure=ae, gate_actual_batch_sequence_exact=cross_gate,
        interpretation='Backbones and initial parameters differ; only actual sampling is paired.')


def verify_extended_sampling(lstm_fit, transformer_fit, previous_lstm, previous_transformer):
    """Check a fresh longer budget and its actual1000-update sampling prefix."""
    pairing = verify_sequence_sampling(lstm_fit, transformer_fit)
    prefixes = {}
    sampled = lambda rows: [{key: row[key] for key in ('step', 'mode', 'batch')} for row in rows]
    for name, current, previous in (('lstm', lstm_fit, previous_lstm),
                                    ('transformer', transformer_fit, previous_transformer)):
        new, nr, _ = _sequence_sampling(current)
        old, ore, _ = _sequence_sampling(previous)
        config = deepcopy(old['configuration'])
        config['training']['updates'] = 4000
        if (new['configuration'] != config or len(nr) != 4000 or len(ore) != 1000
                or new['training_scene_ids'] != old['training_scene_ids']
                or sampled(nr[:1000]) != sampled(ore)):
            raise ValueError('072 sequence budget or actual071 sampling prefix differs')
        snapshots = []
        with torch.random.fork_rng(devices=[]):
            for directory, expected in ((current, new['configuration']), (previous, old['configuration'])):
                model, snapshot = sequence_checkpoint(Path(directory)/'step-0.pt', device='cpu')
                del model
                if snapshot['configuration'] != expected or snapshot['step'] != 0:
                    raise ValueError('072 fresh initial checkpoint/configuration differs')
                snapshots.append(snapshot)
        left, right = snapshots
        if (set(left['model']) != set(right['model']) or any(
                left['model'][key].shape != right['model'][key].shape
                or left['model'][key].dtype != right['model'][key].dtype
                or not torch.equal(left['model'][key], right['model'][key]) for key in left['model'])
                or left['numpy_rng'] != right['numpy_rng']):
            raise ValueError('072 fresh parameters or initial sampler RNG differ from071')
        prefixes[name] = dict(previous_fit=str(Path(previous).resolve()),
            initial_parameters_exact=True, initial_sampler_rng_exact=True,
            actual_first1000_batches_exact=True, matched_prefix_updates=1000)
    return dict(pairing, prefixes=prefixes, fresh_training=True,
        interpretation='Fresh4000-update budgets share initial parameters and actual first1000 samples with their own071 arms; this does not assert bitwise optimizer trajectories or convergence.')


def _metadata(path, arm):
    path = Path(path).resolve()
    run, summary, fit = [read_json(path/name) for name in ('run.json', 'summary.json', 'fit/run.json')]
    extended = arm in EXTENDED_CONFIGS or arm in REPLICATION_CONFIGS
    config = read_json(ROOT/'configs'/({**ARM_CONFIGS, **EXTENDED_CONFIGS, **REPLICATION_CONFIGS}[arm]))
    steps = 4000 if extended else 1000
    if config['training']['updates'] != steps:
        raise ValueError(f'comparison requires declared{steps}-update budgets')
    for record in (run, summary, fit):
        if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                or record.get('optimizer_updates') != steps or record.get('test_opened') is not False):
            raise ValueError('all actual run/configuration/updates must describe completed frozen jobs')
    if (run.get('source_system') != summary.get('source_system')
            or run.get('validation_selection') is not False or fit.get('final_checkpoint') != 'final.pt'):
        raise ValueError('completed job provenance or fixed-endpoint selection differs')
    expected_endpoints = ['final'] if arm == '070_linear' else ['initial', 'final']
    if run.get('completed_endpoints') != expected_endpoints:
        raise ValueError('full required endpoint chain has not completed')
    sequence = 'backbone' in config['model']
    loader = sequence_checkpoint if sequence else gate_checkpoint
    with torch.random.fork_rng(devices=[]):
        model, checkpoint = loader(path/'fit/final.pt', device='cpu')
    del model
    if (checkpoint['configuration'] != config or checkpoint['step'] != steps
            or checkpoint['actual_updates'] != steps):
        raise ValueError('actual final checkpoint/configuration/update mismatch')
    manifest = read_json(ROOT/'configs/data/mtad-level1-scene3000.json')
    if fit.get('training_scene_ids') != manifest['splits']['T_core']:
        raise ValueError('training and validation must retain complete declared Level1 populations')
    # Preserve the071 contract; the explicit072 extension additionally binds
    # the initial CPU endpoint instead of trusting completed_endpoints alone.
    endpoints = (('initial', 0, 'step-0.pt'), ('final', steps, 'final.pt')) if extended else (('final', steps, 'final.pt'),)
    for label, endpoint_step, filename in endpoints:
        final = read_json(path/label/'summary.json')
        for record in (read_json(path/label/'run.json'), final):
            endpoint = record.get('checkpoint', {})
            if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                    or record.get('source_system') != run['source_system'] or record.get('device') != 'cpu'
                    or record.get('optimizer_updates') != 0 or record.get('test_opened') is not False
                    or record.get('groups_recomputed') is not False or record.get('threshold_reselected') is not False
                    or endpoint.get('step') != endpoint_step or endpoint.get('actual_updates') != endpoint_step
                    or endpoint.get('configured_final_updates') != steps
                    or Path(endpoint.get('path', '')).resolve() != path/'fit'/filename):
                raise ValueError(label+' CPU output does not bind the actual frozen endpoint')
        if final['scene_ids'] != manifest['splits']['V_select']:
            raise ValueError('training and validation must retain complete declared Level1 populations')
        components = {sid: str(component) for component, group in final['components'].items() for sid in group['scene_ids']}
        if components != {sid: str(manifest['components'][sid]) for sid in final['scene_ids']}:
            raise ValueError('endpoint component registry differs from frozen manifest')
        for directory in ('predictions', 'system-predictions'):
            if {p.name for p in (path/label/directory).glob('*.npz')} != {f'states-{s}.npz' for s in final['scene_ids']}:
                raise ValueError(label+' cache inventory is incomplete or has extra scenes')
    return dict(path=path, method='learned_sequence' if sequence else 'learned_gate',
                source_system=run['source_system'], scene_ids=final['scene_ids'], components=components,
                summary=final)


def _constant_metadata(path, source071B):
    """Bind the scalar to its T-only grid fit and actual source training samples."""
    path, source071B = Path(path).resolve(), Path(source071B).resolve()
    run, summary, fit = [read_json(path/name) for name in ('run.json', 'summary.json', 'fit.json')]
    config = read_json(ROOT/'configs/constant-gate-level1.json')
    scope = '072-training-fitted-constant-gate'
    for record in (run, summary, fit):
        if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                or record.get('optimizer_updates') != 0 or record.get('test_opened') is not False
                or Path(record.get('source071B', '')).resolve() != source071B
                or record.get('source_system') != run.get('source_system')):
            raise ValueError('072 constant fit/run/source provenance differs')
    if (run.get('scope') != scope or summary.get('scope') != scope
            or run.get('completed_endpoints') != ['final'] or run.get('validation_selection') is not False
            or fit.get('train_only') is not True or fit.get('candidate_count') != 1001
            or fit.get('sampling_replay_verified') is not True
            or fit.get('successful_updates_replayed') != 1000 or fit.get('sampled_anchors') != 8000):
        raise ValueError('072 constant requires the completed T-only1001-grid fit, not V selection')
    source_log = source071B/'fit/training.jsonl'
    if Path(fit.get('source_training_log', '')).resolve() != source_log:
        raise ValueError('072 constant source must be the original071B successful training log')
    rows = [json.loads(line) for line in source_log.read_text().splitlines()]
    recorded = read_json(path/'fit-sampling.json')
    if (len(rows) != 1000 or sum(len(row['batch']) for row in rows) != 8000
            or recorded != [{key: row[key] for key in ('step', 'mode', 'batch')} for row in rows]
            or fit.get('training_scene_ids') != read_json(source071B/'fit/run.json')['training_scene_ids']):
        raise ValueError('072 constant actual replayed samples differ from071B')
    logits = np.asarray(fit.get('grid_logits'), np.float64)
    gates = np.asarray(fit.get('effective_gates'), np.float64)
    losses = np.asarray(fit.get('training_losses'), np.float64)
    if any(value.shape != (1001,) or not np.isfinite(value).all() for value in (logits, gates, losses)):
        raise ValueError('072 constant requires finite actual1001-grid logits/gates/training losses')
    nominal = np.linspace(0., 1., 1001)
    expected_logits = np.empty(1001, np.float32)
    expected_logits[0], expected_logits[-1] = -100., 100.
    expected_logits[1:-1] = np.log(nominal[1:-1]/(1.-nominal[1:-1]))
    _same(logits, expected_logits.astype(np.float64), '072 fixed training-only scalar grid')
    expected_gates = torch.sigmoid(torch.as_tensor(expected_logits)).numpy().astype(np.float64)
    _same(gates, expected_gates, '072 scalar effective float32 sigmoid gates')
    selected = fit.get('selected_index')
    if (type(selected) is not int or selected != int(np.argmin(losses))
            or fit.get('selected_logit') != float(logits[selected])
            or fit.get('selected_gate') != float(gates[selected])):
        raise ValueError('072 selected scalar must minimize the saved T-only objective with lower-index ties')
    final = read_json(path/'final/summary.json')
    for record in (read_json(path/'final/run.json'), final):
        if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                or record.get('source_system') != run['source_system'] or record.get('device') != 'cpu'
                or record.get('optimizer_updates') != 0 or record.get('test_opened') is not False
                or record.get('groups_recomputed') is not False or record.get('threshold_reselected') is not False
                or Path(record.get('fit_source', '')).resolve() != path/'fit.json'
                or record.get('selected_logit') != fit['selected_logit']
                or record.get('selected_gate') != fit['selected_gate'] or 'checkpoint' in record):
            raise ValueError('072 constant CPU endpoint does not bind the actual scalar fit')
    manifest = read_json(ROOT/'configs/data/mtad-level1-scene3000.json')
    if (fit['training_scene_ids'] != manifest['splits']['T_core']
            or final['scene_ids'] != manifest['splits']['V_select']):
        raise ValueError('072 constant must retain the full original T/V populations')
    components = {sid: str(component) for component, group in final['components'].items() for sid in group['scene_ids']}
    if components != {sid: str(manifest['components'][sid]) for sid in final['scene_ids']}:
        raise ValueError('072 constant component registry differs from frozen manifest')
    for directory in ('predictions', 'system-predictions'):
        if {p.name for p in (path/'final'/directory).glob('*.npz')} != {f'states-{s}.npz' for s in final['scene_ids']}:
            raise ValueError('072 constant final cache inventory differs')
    return dict(path=path, method='learned_gate', source_system=run['source_system'],
        scene_ids=final['scene_ids'], components=components, summary=final,
        selected_logit=fit['selected_logit'], selected_gate=fit['selected_gate'],
        fit_source=str(path/'fit.json'), train_only=True)


def _arrays(directory, sid):
    return tuple(_read(Path(directory)/sub/f'states-{sid}.npz')
                 for sub in ('predictions', 'system-predictions'))


def scene_arrays(arms, sid, initial_directory):
    """Both schemas share actual full-scene artifacts, not summary claims."""
    reference, system = _arrays(next(iter(arms.values()))['path']/'final', sid)
    base = _artifact(system, 'anchor')
    _, _, regions, population = _structure(reference, base)
    data = {'region_'+key: value for key, value in regions.items()}
    for control in CONTROLS:
        data[control+'_position_error'] = _check_artifact(reference, _artifact(system, control), base, control)
    initial, initial_system = _arrays(initial_directory, sid)
    _verify_route(initial_directory, sid, initial=True)
    prior = _artifact(initial_system, 'learned_gate')
    for key in INPUT_KEYS:
        _same(initial[key], reference[key], '071 shared initial input/'+key)
    data['fixed_prior095_position_error'] = _check_artifact(initial, prior, base, 'learned_gate')
    for name, arm in arms.items():
        fragment, saved = _arrays(arm['path']/'final', sid)
        for key in INPUT_KEYS:
            _same(fragment[key], reference[key], '071 cross-model physical input/'+key)
        for region in REGIONS:
            _same(fragment['region_'+region], data['region_'+region], '071 cross-model support stratum/'+region)
        for control in CONTROLS:
            current, fixed = _artifact(saved, control), _artifact(system, control)
            if set(current) != set(fixed):
                raise ValueError('ordinary complete artifact fields differ')
            for key in fixed:
                _same(current[key], fixed[key], '071 ordinary full artifact/'+control+'/'+key)
            _check_artifact(fragment, current, base, control)
        artifact = _artifact(saved, arm['method'])
        error = _check_artifact(fragment, artifact, base, arm['method'])
        _same(np.isfinite(error), np.isfinite(data['anchor_position_error']), '071 identical target masks')
        _same(artifact['group_states'][..., 2:], _artifact(saved, 'mean_supported')['group_states'][..., 2:],
              '071 all output-group speed/course remain Mean')
        if arm['method'] == 'learned_gate':
            _verify_route(arm['path']/'final', sid)
            if name == CONSTANT_METHOD:
                _same(fragment['group_logits'], np.full_like(fragment['group_logits'], arm['selected_logit']),
                      '072 every output group uses its T-fitted constant logit')
                _same(fragment['group_gate'], np.full_like(fragment['group_gate'], arm['selected_gate']),
                      '072 every output group uses its T-fitted constant gate')
        else:
            fixed = _artifact(saved, 'fixed_prior095')
            if set(fixed) != set(prior):
                raise ValueError('sequence fixed-prior artifact schema differs')
            for key in prior:
                _same(fixed[key], prior[key], '071 sequence shared .95 control/'+key)
            _check_artifact(fragment, fixed, base, 'fixed_prior095')
        data[name+'_position_error'] = error
    t = len(base['target_ids'])
    available = base['aligned_truth_valid'][:t, :, :2].all(-1)
    covered = available & base['aligned_prediction_valid'][:t, :, :2].all(-1)
    position_population = dict(available_target_points=int(available.sum()),
        covered_target_points=int(covered.sum()), missing_prediction_points=int((available & ~covered).sum()),
        available_target_trajectories=int(available.any(-1).sum()),
        covered_target_trajectories=int(covered.any(-1).sum()))
    return dict(scene_id=str(sid), population=population, position_population=position_population), data


def _pool(rows, arrays, methods=METHODS):
    position_population = {key: sum(row['position_population'][key] for row in rows)
                           for key in rows[0]['position_population']}
    available = position_population['available_target_points']
    position_population['point_coverage'] = position_population['covered_target_points']/available if available else None
    return dict(scene_count=len(rows), scene_ids=[r['scene_id'] for r in rows], position_population=position_population,
        population={key: sum(row['population'][key] for row in rows) for key in rows[0]['population']},
        position_weightings=_weighted(rows, arrays, methods=methods))


def run(args):
    inputs = dict(zip(ARM_CONFIGS, (args.log_run071A, args.linear_run071B, args.lstm_run071C,
                                  args.transformer_run071D, args.source_070)))
    inputs = {key: Path(value).resolve() for key, value in inputs.items()}
    optional = [getattr(args, key, None) for key in OPTIONAL_INPUTS]
    if any(optional) and not all(optional):
        raise ValueError('072 closeout requires all three optional runs together')
    extended = all(optional)
    if extended:
        inputs.update({key: Path(value).resolve() for key, value in
                       zip((CONSTANT_METHOD, *EXTENDED_CONFIGS), optional)})
    methods = (*METHODS, CONSTANT_METHOD, *EXTENDED_CONFIGS) if extended else METHODS
    output = Path(args.output).resolve()
    if args.cpu_threads != 4:
        raise ValueError('071 CPU closeout fixes CPU threads to4')
    torch.set_num_threads(4)
    if output.exists():
        raise FileExistsError('output already exists; no retry/overwrite')
    sources = [Path(read_json(path/'run.json')['source_system']).resolve().parent for path in inputs.values()]
    for source in (*inputs.values(), *sources):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('output must be separate from all immutable sources')
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='RUNNING', model_forwards=0, optimizer_updates=0, test_opened=False,
                 checkpoint_read_device='cpu', completed_scenes=0)
    write_json(output/'run.json', state)
    try:
        pairing = verify_pairing(inputs['071A_log']/'fit', inputs['071B_linear']/'fit')
        if pairing.get('status') != 'PAIR_VERIFIED':
            raise ValueError('actual A/B pairing must succeed before cross-model comparison')
        sequence_pairing = verify_sequence_sampling(inputs['071C_lstm']/'fit', inputs['071D_transformer']/'fit',
                                                    inputs['071A_log']/'fit')
        arms = {name: _metadata(path, name) for name, path in inputs.items() if name != CONSTANT_METHOD}
        extended_pairing = None
        if extended:
            extended_pairing = verify_extended_sampling(inputs['072B_lstm4000']/'fit',
                inputs['072C_transformer4000']/'fit', inputs['071C_lstm']/'fit', inputs['071D_transformer']/'fit')
            arms[CONSTANT_METHOD] = _constant_metadata(inputs[CONSTANT_METHOD], inputs['071B_linear'])
        first = arms['071A_log']
        for arm in arms.values():
            if any(arm[key] != first[key] for key in ('source_system', 'scene_ids', 'components')):
                raise ValueError('all arms require the same source, complete V population and components')
        rows, arrays = [], []
        for sid in first['scene_ids']:
            row, data = scene_arrays(arms, sid, inputs['071A_log']/'initial')
            row['component_id'] = first['components'][sid]
            rows.append(row); arrays.append(data)
            state['completed_scenes'] = len(rows)
            write_json(output/'run.json', state)
        summary = _pool(rows, arrays, methods)
        # Cross-check actual reconstructed complete-point metrics against every producer.
        for name, arm in arms.items():
            actual = summary['position_weightings']['all'][name]['point_equal']
            expected = arm['summary']['methods'][arm['method']]['full_target_metrics']['all']['position']
            if actual['unit_count'] != expected['intersection_count'] or any(
                    not np.isclose(actual[key], expected[key], rtol=1e-10, atol=1e-9) for key in ('mae', 'rmse')):
                raise ValueError('reconstructed state metrics differ from producer summary/'+name)
        components = {}
        for component in sorted(set(first['components'].values())):
            indices = [i for i, row in enumerate(rows) if row['component_id'] == component]
            components[component] = _pool([rows[i] for i in indices], [arrays[i] for i in indices], methods)
        largest = min(components, key=lambda key: (-components[key]['position_weightings']['all']['anchor']['point_equal']['unit_count'], key))
        rest = [i for i, row in enumerate(rows) if row['component_id'] != largest]
        summary.update(status='COMPLETED', methods=list(methods), pairing=pairing,
            sequence_pairing=sequence_pairing, components=components,
            largest_component_id=largest, remaining_components=_pool([rows[i] for i in rest], [arrays[i] for i in rest], methods) if rest else None,
            source_system=first['source_system'], inputs={key: str(path) for key, path in inputs.items()},
            model_forwards=0, optimizer_updates=0, test_opened=False, old066_acceptance='REJECTED',
            interpretation='Paired second-seed loss intervention plus bounded generic temporal baselines; reused V, not held-out superiority.',
            comparability='Gate acts only in support opportunity; sequence has full-Q 2D residual within base-point hemisphere (not constrained to Anchor–Mean path). Shared physical inputs/outputs, different capacity.',
            speed_heading='Exactly ordinary Mean for every learned output group; no learned speed/course claim.')
        if extended:
            summary.update(extended_sequence_pairing=extended_pairing,
                constant_fit=dict(fit_source=arms[CONSTANT_METHOD]['fit_source'], train_only=True,
                    selected_logit=arms[CONSTANT_METHOD]['selected_logit'], selected_gate=arms[CONSTANT_METHOD]['selected_gate']),
                interpretation='072 T-fitted global scalar control and fresh4000-update generic baselines alongside frozen071 evidence; reused V, not a new loss-pair replication or held-out superiority.')
        scene_rows = [dict(_pool([row], [data], methods), component_id=row['component_id']) for row, data in zip(rows, arrays)]
        write_json_gzip(output/'scene-summary.json.gz', scene_rows)
        write_json_gzip(output/'summary.json.gz', summary)
        table = [f'# {"072" if extended else "071"}冻结完整人口位置比较', '', '| 方法 | MAE(m) | RMSE(m) | P95(m) | P99(m) | 有效点 |', '|---|---:|---:|---:|---:|---:|']
        for method in methods:
            item = summary['position_weightings']['all'][method]['point_equal']
            table.append('| '+method+' | '+' | '.join(f'{item[key]:.4f}' for key in ('mae', 'rmse', 'p95', 'p99'))+f" | {item['unit_count']} |")
        table += ['', '固定同一056分组、首代表、目标人口及有效掩码。四权重、支撑层、逐场与共享身份组件详见JSON。',
                  '这是复用V上的有界pilot；一般LSTM/Transformer控制不等于作者方法复现，也不证明已收敛。']
        (output/'report.md').write_text('\n'.join(table)+'\n')
        state.update(status='COMPLETED', paired=True, output_population=summary['population'])
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='FAILED', paired=False, error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('log-run071A', 'linear-run071B', 'lstm-run071C', 'transformer-run071D', 'source-070', 'output'):
        parser.add_argument('--'+key, required=True)
    for key in ('constant-run072A', 'lstm-run072B', 'transformer-run072C'):
        parser.add_argument('--'+key)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    result = run(args)
    print(result['status']+': '+str(Path(args.output).resolve()), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
