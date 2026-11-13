"""Opt-in speed units for the four-state TMTAFF adaptation.

Raw readers remain untouched. This policy records the current Level1 evidence
(eight training scenes), not a claim that every reference row was audited.
"""
from __future__ import annotations

from collections.abc import Mapping

from association.data.archive import DataError

SPEED_UNIT_POLICY = 'measurement_kn_reference_mps_v1'
KNOT_TO_MPS = 1852.0 / 3600.0


def unit_metadata(policy=None):
    if policy is None:
        return None
    if policy != SPEED_UNIT_POLICY:
        raise DataError('Unknown speed_unit_policy; use measurement_kn_reference_mps_v1 explicitly')
    return dict(policy=policy, measurement_speed_unit='kn', state_speed_unit='mps',
                reference_speed_unit='mps', paper_speed_unit='kn',
                measurement_to_state_factor=KNOT_TO_MPS,
                basis='Level1 eight T_core scenes; reference generation code unverified')


def scene_speed_policy(scene):
    policy = scene.get('adapter_config', {}).get('speed_unit_policy')
    expected = unit_metadata(policy)
    if scene.get('physical_units') != expected:
        raise DataError('Scene physical_units disagree with its adapter speed_unit_policy')
    return policy


def normalizer_speed_policy(normalizer):
    if normalizer is None:
        return None
    if not isinstance(normalizer, Mapping):
        raise DataError('Normalizer must be a mapping with explicit unit metadata when opted in')
    policy = normalizer.get('adapter_config', {}).get('speed_unit_policy')
    if normalizer.get('physical_units') != unit_metadata(policy):
        raise DataError('Normalizer physical_units disagree with its adapter speed_unit_policy')
    return policy


def validate_target_units(scene):
    """Require declared m/s targets for opt-in scenes; never convert reference."""
    policy = scene_speed_policy(scene)
    expected = unit_metadata(policy)
    targets = scene.get('fusion_targets')
    if not isinstance(targets, Mapping):
        raise DataError('fusion_targets must explicitly declare aligned reference targets')
    for target in targets.values():
        if target.get('physical_units') != expected:
            raise DataError('Fusion target units do not match the physical input contract')
    return policy
