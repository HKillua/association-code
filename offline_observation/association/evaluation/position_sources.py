"""Shared saved-position provenance and aggregate statistics.

Retains current source/cache checks without importing retired experiment CLIs.
No inference, training, archive read, threshold selection, or test-set access.
"""
from pathlib import Path
import json

import numpy as np
import torch

from association.evaluation.sequence_position import _weighted
from association.training.evidence_fusion import TRAIN_MODES
from association.training.support_gate import load_checkpoint as gate_checkpoint
from association.training.sequence_position import load_checkpoint as sequence_checkpoint
from association.utils.runtime import read_json

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

HARM_CONFIGS = {
    '074A_harm4000_seed23': 'sequence-position-transformer-relative-harm-level1.json',
    '074B_harm4000_seed21': 'sequence-position-transformer-relative-harm-replica-level1.json',
}

DROPOUT_CONFIGS = {'083_helper_dropout4000': 'sequence-position-transformer-helper-dropout-level1.json'}

INPUT_KEYS = ('node_ids', 'identities', 'target_identity_ids', 'native_states', 'support',
              'nearest_seconds', 'exact_observation', 'query_seconds', 'learned_assoc_group_offsets',
              'learned_assoc_group_node_indices', 'group_anchor_node_indices')

METHODS = (*CONTROLS, 'fixed_prior095', *ARM_CONFIGS)

SAMPLE_KEYS = {'scene_id', 'anchor_index', 'node_id', 'members', 'eligible_query_points',
               'eligible_valid_position_points', 'valid_position_points'}

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

def _metadata(path, arm):
    path = Path(path).resolve()
    run, summary, fit = [read_json(path/name) for name in ('run.json', 'summary.json', 'fit/run.json')]
    extended = arm in EXTENDED_CONFIGS or arm in REPLICATION_CONFIGS or arm in HARM_CONFIGS or arm in DROPOUT_CONFIGS
    config = read_json(ROOT/'configs'/({**ARM_CONFIGS, **EXTENDED_CONFIGS, **REPLICATION_CONFIGS, **HARM_CONFIGS, **DROPOUT_CONFIGS}[arm]))
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

def _pool(rows, arrays, methods=METHODS):
    position_population = {key: sum(row['position_population'][key] for row in rows)
                           for key in rows[0]['position_population']}
    available = position_population['available_target_points']
    position_population['point_coverage'] = position_population['covered_target_points']/available if available else None
    return dict(scene_count=len(rows), scene_ids=[r['scene_id'] for r in rows], position_population=position_population,
        population={key: sum(row['population'][key] for row in rows) for key in rows[0]['population']},
        position_weightings=_weighted(rows, arrays, methods=methods))
