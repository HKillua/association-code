"""T-only fragment dropout and observable exposure audit for experiment 083.

Membership intervention accepts no identities or reference values. Labels enter
only the separate audit, after the random retained set has already been chosen.
"""
from __future__ import annotations
import copy
import math
import numpy as np

SCOPE = 'sequence-position-helper-dropout-v1'
KIND = 'helper_fragment_dropout'


def drop_helper_fragments(members, rng, probability):
    """Drop whole helper fragments; always retain the first node and its order.

    Boundary probabilities are allowed for unit tests, not production config.
    Draw once per original helper even at p=0; sampling uses a separate RNG.
    """
    nodes = list(members)
    if (not nodes or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0 for n in nodes)
            or len(set(nodes)) != len(nodes) or not isinstance(rng, np.random.Generator)
            or isinstance(probability, bool) or not isinstance(probability, (float, int))
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError('dropout requires ordered unique nodes, independent RNG and probability in [0,1]')
    dropped = rng.random(len(nodes)-1) < probability
    return [int(nodes[0])] + [int(node) for node, drop in zip(nodes[1:], dropped) if not drop], dropped.tolist()


def support_counts(support, members, valid_positions):
    count = np.asarray(support)[list(members)[1:]].sum(0)
    return {population: {name: int((selection & mask).sum())
        for name, selection in (('zero', count == 0), ('one', count == 1), ('multiple', count >= 2))}
        for population, mask in (('all_queries', np.ones_like(count, bool)),
                                  ('valid_position_queries', np.asarray(valid_positions, bool)))}


def dropout_audit(scene, original, retained, drop_mask, valid_positions):
    """Post-intervention label counts; never used to choose retained members."""
    identities = scene['identities']; anchor = original[0]
    def identity_counts(nodes):
        same = sum(identities[node] == identities[anchor] for node in nodes[1:])
        return dict(correct_helpers=int(same), wrong_helpers=len(nodes)-1-int(same))
    original_support = support_counts(scene['motion']['support'], original, valid_positions)
    retained_support = support_counts(scene['motion']['support'], retained, valid_positions)
    support = scene['motion']['support']
    eligible = ~support[anchor] & support[retained[1:]].any(0)
    return dict(original_members=list(original), retained_members=list(retained), drop_mask=list(drop_mask),
        dropped_helpers=int(sum(drop_mask)), original_helper_count=len(original)-1,
        retained_helper_count=len(retained)-1, original_identity=identity_counts(original),
        retained_identity=identity_counts(retained), original_support=original_support,
        retained_support=retained_support, retained_eligible_query_points=int(eligible.sum()),
        retained_eligible_valid_position_points=int((eligible & valid_positions).sum()))


def empty_exposure():
    return dict(successful_updates=0, sampled_anchors=0, original_helpers=0, retained_helpers=0,
        dropped_helpers=0, all_helpers_dropped_anchors=0, original_singleton_anchors=0,
        original_identity=dict(correct_helpers=0, wrong_helpers=0),
        retained_identity=dict(correct_helpers=0, wrong_helpers=0),
        original_support={p: {k: 0 for k in ('zero', 'one', 'multiple')}
                          for p in ('all_queries', 'valid_position_queries')},
        retained_support={p: {k: 0 for k in ('zero', 'one', 'multiple')}
                          for p in ('all_queries', 'valid_position_queries')})


def commit_exposure(exposure, batch):
    """Called only after optimizer.step succeeds; never counts failed attempts."""
    exposure['successful_updates'] += 1
    exposure['sampled_anchors'] += len(batch)
    for sample in batch:
        for key, source in (('original_helpers', 'original_helper_count'),
                            ('retained_helpers', 'retained_helper_count'), ('dropped_helpers', 'dropped_helpers')):
            exposure[key] += sample[source]
        exposure['all_helpers_dropped_anchors'] += int(sample['original_helper_count'] > 0 and sample['retained_helper_count'] == 0)
        exposure['original_singleton_anchors'] += int(sample['original_helper_count'] == 0)
        for key in ('original_identity', 'retained_identity'):
            for label, count in sample[key].items(): exposure[key][label] += count
        for key in ('original_support', 'retained_support'):
            for population, counts in sample[key].items():
                for label, count in counts.items(): exposure[key][population][label] += count


def validate_rng_state(state):
    """Require the actual PCG64 state shape used by default_rng, not a seed."""
    expected = np.random.default_rng(0).bit_generator.state
    if (not isinstance(state, dict) or set(state) != set(expected) or state.get('bit_generator') != 'PCG64'
            or not isinstance(state.get('state'), dict) or set(state['state']) != {'state', 'inc'}
            or any(type(state['state'].get(k)) is not int or not 0 <= state['state'][k] < 2**128 for k in ('state', 'inc'))
            or state['state']['inc'] % 2 != 1 or type(state.get('has_uint32')) is not int
            or state['has_uint32'] not in (0, 1) or type(state.get('uinteger')) is not int
            or not 0 <= state['uinteger'] < 2**32):
        raise ValueError('helper dropout checkpoint requires a complete valid PCG64 augmentation RNG state')
    rng = np.random.default_rng(0)
    try: rng.bit_generator.state = copy.deepcopy(state)
    except (TypeError, ValueError) as error:
        raise ValueError('invalid helper dropout augmentation RNG state') from error
