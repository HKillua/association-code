"""Explicit configuration validation; no automatic budget/device changes."""
from __future__ import annotations

import math
from copy import deepcopy

from association.models.tmtaff import ModelConfig
from association.utils.runtime import read_json

SPLITS = ('T_core', 'T_head', 'V_select', 'T_test')


def _number(value, name, *, integer=False, positive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if integer and (not isinstance(value, int)):
        raise ValueError(f'{name} must be an integer')
    invalid_sign = value <= 0 if positive else value < 0
    if invalid_sign:
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')


def validate_config(config):
    config = deepcopy(config)
    if not isinstance(config.get('protocol'), str) or not config['protocol']:
        raise ValueError('protocol is required')
    model = ModelConfig(**config['model'])
    for name in ('input_dim', 'd_model', 'nhead', 'num_layers', 'dim_feedforward', 'length'):
        _number(getattr(model, name), f'model.{name}', integer=True)
    if model.input_dim != 4 or model.length < 2:
        raise ValueError('MTAD requires model.input_dim=4 and length>=2')
    _number(config['split_seed'], 'split_seed', integer=True, positive=False)
    if set(config['caps']) != set(SPLITS):
        raise ValueError('caps must name all four splits')
    for name, cap in config['caps'].items():
        _number(cap, f'caps.{name}', integer=True)
    seeds = config['seeds']
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be a nonempty unique list')
    for seed in seeds:
        _number(seed, 'seed', integer=True, positive=False)
    reference = config['reference']
    schedule = reference.get('schedule', 'legacy')
    if schedule not in ('legacy', 'staged_updates_v2'):
        raise ValueError('Unknown reference.schedule')
    for name in ('batch_tracks', 'pair_batch_size'):
        _number(reference[name], f'reference.{name}', integer=True)
    if schedule == 'legacy':
        for name in ('tcm_epochs', 'tam_epochs'):
            _number(reference[name], f'reference.{name}', integer=True)
        _number(reference['warmup_epochs'], 'warmup_epochs', integer=True, positive=False)
        if not isinstance(reference['joint_completion'], bool):
            raise ValueError('joint_completion must be boolean')
    else:
        if reference.get('purpose') not in ('probe', 'fit'):
            raise ValueError('staged_updates_v2 requires reference.purpose probe or fit')
        if reference['pair_batch_size'] < 2 or reference['pair_batch_size'] % 2:
            raise ValueError('staged_updates_v2 pair_batch_size must be even and >=2')
        for name in ('tcm_updates', 'tam_updates', 'validate_every_updates'):
            _number(reference[name], f'reference.{name}', integer=True)
        for name in ('joint_updates', 'completion_cache_mib'):
            _number(reference[name], f'reference.{name}', integer=True, positive=False)
        _number(reference.get('gradient_clip', 1.0), 'reference.gradient_clip')
    for name in ('learning_rate', 'max_hours'):
        _number(reference[name], f'reference.{name}')
    for name in ('weight_decay', 'completion_weight'):
        _number(reference[name], f'reference.{name}', positive=False)
    if schedule == 'staged_updates_v2' and reference['purpose'] == 'probe':
        for name, limit in (('max_hours', .5), ('tcm_updates', 8),
                            ('tam_updates', 4), ('joint_updates', 2)):
            if reference[name] > limit:
                raise ValueError(f'probe reference.{name} must be <= {limit}')
    return config


def load_config(path):
    return validate_config(read_json(path))


def validate_manifest_config(manifest, config):
    """Validate the actual split and preprocessing contract."""
    from association.data.archive import DEFAULT_DEVELOPMENT_SCENES, REGISTRATION_POLICY, UNIT_CONTRACT, validate_manifest
    validate_manifest(manifest)
    if manifest['seed'] != config['split_seed'] or manifest['caps'] != config['caps']:
        raise ValueError('Manifest seed/caps differ from the requested configuration')
    if manifest['registration_policy'] != REGISTRATION_POLICY or manifest['unit_contract'] != UNIT_CONTRACT:
        raise ValueError('Manifest preprocessing contract differs from this implementation')
    if config['protocol'] != 'synthetic-smoke-only-not-research-result':
        expected = set(DEFAULT_DEVELOPMENT_SCENES) & set(manifest['all_scenes'])
        if not expected.issubset(set(manifest['development_scenes'])):
            raise ValueError('Previously inspected development scenes are not all pinned')
    if any(not manifest['splits'][name] for name in SPLITS):
        raise ValueError('Every experiment split must contain selected scenes')
    return True
