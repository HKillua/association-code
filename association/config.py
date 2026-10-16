"""Explicit configuration validation; no automatic budget/device changes."""
from __future__ import annotations

import math
from copy import deepcopy

from .models import ModelConfig
from .runtime import read_json

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
    reference, heads, evaluation = config['reference'], config['heads'], config['evaluation']
    for name in ('tcm_epochs', 'tam_epochs', 'batch_tracks', 'pair_batch_size'):
        _number(reference[name], f'reference.{name}', integer=True)
    _number(reference['warmup_epochs'], 'warmup_epochs', integer=True, positive=False)
    if not isinstance(reference['joint_completion'], bool):
        raise ValueError('joint_completion must be boolean')
    for name in ('learning_rate', 'max_hours'):
        _number(reference[name], f'reference.{name}')
        _number(heads[name], f'heads.{name}')
    for name in ('weight_decay', 'completion_weight'):
        _number(reference[name], f'reference.{name}', positive=False)
    _number(heads['weight_decay'], 'heads.weight_decay', positive=False)
    for name in ('epochs', 'calibration_steps'):
        _number(heads[name], f'heads.{name}', integer=True)
    if not isinstance(heads['deltas'], list) or not heads['deltas'] or len(set(heads['deltas'])) != len(heads['deltas']):
        raise ValueError('deltas must be nonempty and unique')
    for delta in heads['deltas']:
        _number(delta, 'delta')
    for name in ('bootstrap_samples', 'minimum_components'):
        _number(evaluation[name], f'evaluation.{name}', integer=True)
    return config


def load_config(path):
    return validate_config(read_json(path))


def requested_seeds(config, seeds=None):
    selected = config['seeds'] if seeds is None else list(seeds)
    # Branch selection always uses the first predeclared seed. Prefix runs can
    # save compute; arbitrary subsets cannot silently redefine that decision.
    if not selected or selected != config['seeds'][:len(selected)]:
        raise ValueError('Requested seeds must be a nonempty prefix of config.seeds')
    return selected


def validate_manifest_config(manifest, config):
    """Refuse stale split/preprocessing contracts, even if each hash is valid."""
    from .data import (DEFAULT_DEVELOPMENT_SCENES, REGISTRATION_POLICY,
                       UNIT_CONTRACT, validate_manifest)
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
