"""062 bindings to the actual056/057 endpoints and complete saved OLS replay."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from association.evaluation.evidence_fusion_combo import (
    REPLAY_TOLERANCE, read_scores, replay_check,
)
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.system_attribution import _same, assert_metrics
from association.training.evidence_association import load_checkpoint as load_association
from association.training.evidence_fusion import bind_ols_working_point, load_checkpoint
from tools.analysis.compare_association_duration import ARMS, INPUT_KEYS, _load_metadata
from tools.analysis.learned_group_oracle import _bind_source
from association.utils.runtime import read_json_or_gzip as read
from tools.experiments.level1_system_evaluate import verify_fusion_snapshot, _verify_association_metrics

FUSION_TASK = 'LEVEL1-FUSION-20261022-057-attempt-1'
CONFIG_ROOT = Path(__file__).resolve().parents[2]/'configs'
FUSION_CONFIGS = {
    'helper-quality': 'evidence-fusion-level1.json',
    'direct-only': 'evidence-fusion-level1-direct-only.json',
    'mixed-action': 'evidence-fusion-level1-mixed-action.json',
}


def _bind_association_checkpoint(source, manifest, metadata, record):
    training = source.parent/'association'
    summary, audit = read(training/'summary.json'), read(training/'data_audit.json')
    _, snapshot = load_association(training/'final.pt', 'cpu')
    if (snapshot.get('update') != 1000 or snapshot.get('config') != record['config']
            or snapshot.get('normalizer') != audit.get('normalizer')
            or snapshot.get('selection') != summary.get('final_selection')
            or summary.get('variant') != record['config'].get('variant')
            or summary.get('scope') != record['config'].get('scope')
            or summary.get('data') != record['config'].get('data')
            or audit.get('data') != record['config'].get('data')
            or record.get('selected') != summary.get('selected')
            or summary.get('status') != 'COMPLETED' or summary.get('optimizer_updates') != 1000
            or summary.get('motion_observation_policy') != 'canonical_all'
            or audit.get('motion_observation_policy') != 'canonical_all'
            or audit.get('training_scene_ids') != manifest['splits']['T_core']
            or audit.get('validation_scene_ids') != manifest['splits']['V_select']
            or audit.get('split_unit') != 'scene'
            or any(x.get(k) is not False for x in (summary, audit)
                   for k in ('test_opened', 'reference_read'))
            or summary.get('population') != audit.get('validation_population')
            or summary.get('population', {}).get('scenes') != len(metadata['scene_ids'])
            or summary.get('population', {}).get('nodes') != metadata['fragment_count']
            or summary.get('final_selection', {}).get('threshold') != metadata['association_threshold']):
        raise ValueError('056 checkpoint/config/normalizer/selection and original population disagree')
    _verify_association_metrics(metadata['association'], summary['final_result']['metrics'])
    scores = read_scores(training/'final-scores.npz')
    if list(scores) != metadata['scene_ids']:
        raise ValueError('056 final score scene order differs from complete V')
    pair_count = 0
    for sid in metadata['scene_ids']:
        row = scores[sid]
        with np.load(source/'predictions'/f'states-{sid}.npz', allow_pickle=False) as saved:
            for key in ('node_ids', 'sources'):
                _same(row[key], saved[key], '056 score cache/' + key)
        n = len(row['node_ids'])
        pairs = np.column_stack(np.triu_indices(n, 1))
        _same(row['pairs'], pairs, '056 all-node candidate pair order')
        if row['scores'].shape != (len(pairs),) or not np.isfinite(row['scores']).all():
            raise ValueError('056 scores must be finite on every original pair')
        pair_count += len(pairs)
    if pair_count != summary['population'].get('pairs'):
        raise ValueError('056 complete score pair population changed')
    return dict(checkpoint=str(training/'final.pt'), update=1000,
                checkpoint_metadata_verified=True, score_population_verified=True,
                scores_recomputed=False, pairs=pair_count)


def _bind_fusion_arm(source, arm, manifest, base, device):
    """Keep one source summary in memory, then discard its per-group trees."""
    source = Path(source).resolve()
    fit, system = source/arm/'fit', source/arm/'system'
    expected = bind_ols_working_point(read(CONFIG_ROOT/FUSION_CONFIGS[arm]), manifest,
                                     base['ols_working_point'])
    training, run, summary = read(fit/'run.json'), read(system/'run.json'), read(system/'summary.json')
    fusion, snapshot = load_checkpoint(fit/'final.pt', device=device)
    if (snapshot.get('configuration') != expected or training.get('test_opened') is not False
            or training.get('split_unit') != 'scene' or snapshot.get('step') != 1000):
        raise ValueError('057 fixed arm/objective/configuration/final1000 required: ' + arm)
    verify_fusion_snapshot(fusion, snapshot, training, manifest, expected['grouping'])
    ids = manifest['splits']['V_select']
    methods = sorted('ols_' + name for name in (*ARMS, 'learned'))
    for document in (run, summary):
        if (document.get('status') != 'COMPLETED' or document.get('dataset_level') != 'Level1'
                or document.get('split') != 'V_select' or document.get('test_opened') is not False
                or document.get('optimizer_updates') != 0 or document.get('association_endpoint') is not None
                or document.get('ordinary_controls') != list(ARMS)
                or document.get('method_names') != methods or document.get('fusion_checkpoint_step') != 1000):
            raise ValueError('057 original completed OLS four-arm final system required: ' + arm)
    if (run.get('motion_observation_policy') != 'canonical_all'
            or run.get('threshold_reselected') is not False or run.get('normalizer_refitted') is not False
            or run.get('completed_scenes') != len(ids) or run.get('total_scenes') != len(ids)
            or summary.get('scene_ids') != ids or summary.get('scene_count') != len(ids)
            or summary.get('fragment_count') != base['fragment_count']
            or summary.get('ols_working_point') != base['ols_working_point']
            or run.get('inputs') != summary.get('inputs')):
        raise ValueError('057 original full V/input protocol differs: ' + arm)
    inputs = run.get('inputs', {})
    for key, filename in (('fusion_checkpoint', 'final.pt'), ('fusion_run', 'run.json')):
        if not isinstance(inputs.get(key), str) or Path(inputs[key]).resolve() != fit/filename:
            raise ValueError('057 system must bind its original sibling fit/' + filename)
    if any(inputs.get('association_' + key) is not None
           for key in ('scores', 'summary', 'data_audit', 'checkpoint', 'run')):
        raise ValueError('057 source must be the original OLS-only fusion evaluation')
    rows = summary.get('per_scene', [])
    if [row.get('scene_id') for row in rows] != ids or any(
            sorted(row.get('unique_target', {})) != methods for row in rows):
        raise ValueError('057 per-scene order or method population changed')
    files = {f'states-{sid}.npz' for sid in ids}
    for directory in ('predictions', 'system-predictions'):
        if {p.name for p in (system/directory).glob('*.npz')} != files:
            raise ValueError('057 missing or extra cached scenes: ' + arm + '/' + directory)
    compact = dict(system=str(system), checkpoint=str(fit/'final.pt'), training_run=str(fit/'run.json'),
        step=1000, objective=expected['training']['utility_objective'], scene_ids=ids,
        component_ids={row['scene_id']: row['component_id'] for row in rows},
        expected={row['scene_id']: row['unique_target']['ols_learned']['full_target_metrics'] for row in rows},
        configuration=expected, original_device=training.get('device'),
        frozen_parameters_verified=True, test_opened=False)
    fusion.eval().requires_grad_(False)
    return fusion, compact


def bind_sources(source056, source057, manifest, device='cpu'):
    """Read-only source checks; no scoring, grouping, normalization or training."""
    source056, source057 = Path(source056).resolve(), Path(source057).resolve()
    splits = manifest.get('splits', {})
    if (manifest.get('dataset_level') != 'Level1' or manifest.get('split_unit') != 'scene'
            or len(splits.get('T_core', [])) != 2100 or len(splits.get('V_select', [])) != 300
            or any(len(splits[key]) != len(set(splits[key])) for key in ('T_core', 'V_select'))
            or set(splits['T_core']) & set(splits['V_select'])):
        raise ValueError('062 requires the fixed Level1 T2100/V300 scene population')
    if source057.name != FUSION_TASK:
        raise ValueError('062 requires original057 attempt-1 root')
    record = _bind_source(source056)
    base = _load_metadata(source056, manifest, 1000)
    base['component_ids'] = {row['scene_id']: row['component_id']
                             for row in read(source056/'summary.json')['per_scene']}
    base['checkpoint_binding'] = _bind_association_checkpoint(source056, manifest, base, record)
    models, metadata = {}, {}
    for arm in FUSION_CONFIGS:
        name = arm.replace('-', '_')
        models[name], metadata[name] = _bind_fusion_arm(source057, arm, manifest, base, device)
    components = [item['component_ids'] for item in metadata.values()]
    if any(item != base['component_ids'] for item in components):
        raise ValueError('056/057 arms disagree on shared-identity component registry')
    return base, models, metadata


def replay_fusion_source(fragment056, saved056, old057fragment, old057system,
                         predictionsOLS, routingOLS, *, expected=None):
    """Verify all old OLS output groups, including duplicate/non-retained groups.

    New learned-group predictions are not compared to the old OLS prediction.
    The supplied forward is on the original OLS groups solely to verify frozen
    checkpoint replay across devices. All categorical and ordinary arrays are exact.
    """
    for key in INPUT_KEYS:
        _same(old057fragment[key], fragment056[key], '056/057 input/' + key)
    for key in fragment056:
        if key.startswith(('ols_', 'region_')):
            _same(old057fragment[key], fragment056[key], '056/057 ordinary/' + key)
    for key in saved056:
        if key.startswith('ols_'):
            _same(old057system[key], saved056[key], '056/057 system/' + key)
    prefix = 'ols_routing_'
    for key in ('helper_anchor', 'helper_node', 'membership'):
        _same(routingOLS[key], old057fragment[prefix + key], '057 routing/' + key)
    adapted = dict(native_states=fragment056['native_states'])
    for key, value in (
            ('learned', old057fragment['ols_learned']),
            ('anchor_weights', old057fragment[prefix + 'anchor_weights']),
            ('membership', old057fragment['ols_membership']),
            ('mean_supported', old057fragment['ols_mean_supported'])):
        adapted['predicted_anchor_direct_' + key] = value
    result = replay_check(adapted, predictionsOLS, routingOLS)
    tolerance = dict(atol=REPLAY_TOLERANCE['anchor_weight_atol'],
                     rtol=REPLAY_TOLERANCE['anchor_weight_rtol'])
    for key in ('helper_weights', 'utility'):
        current, previous = np.asarray(routingOLS[key]), old057fragment[prefix + key]
        close = (current.shape == previous.shape and np.isfinite(current).all()
                 and np.isfinite(previous).all() and np.allclose(current, previous, **tolerance))
        result['checks'][key + '_close'] = bool(close)
        result['max_' + key + '_difference'] = (float(np.max(abs(current - previous), initial=0.))
                                               if current.shape == previous.shape else None)
    result['passed'] = bool(all(result['checks'].values()))
    if not result['passed']:
        raise ValueError('057 frozen OLS inference exceeds predeclared replay tolerance: ' + str(result))
    # Reconstruct the old learned artifact from its complete saved prediction,
    # not just its retained target rows. This detects duplicate-group corruption.
    offsets, flat = (fragment056['ols_' + key] for key in ('group_offsets', 'group_node_indices'))
    groups = [flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    targets = fragment056['target_identity_ids']
    reference = {key[len('ols_anchor_'):]: value for key, value in saved056.items()
                 if key.startswith('ols_anchor_')}
    report = evaluate_full_scene(
        [dict(node_indices=members, states=old057fragment['ols_learned'][members[0]]) for members in groups],
        node_identities=fragment056['identities'], target_identities=targets,
        target_states=reference['aligned_truth'][:len(targets)],
        target_valid=reference['aligned_truth_valid'][:len(targets)], node_observed=fragment056['support'],
        split='V_select', grouping_kind='predicted', correspondence_policy='anchor_identity_first',
        return_artifact=True, node_input_support=dict(support=fragment056['support'],
                                                    nearest_seconds=fragment056['nearest_seconds']))
    actual = report.pop('artifact')
    cached = {key[len('ols_learned_'):]: value for key, value in old057system.items()
              if key.startswith('ols_learned_')}
    if set(actual) != set(cached):
        raise ValueError('057 learned system artifact fields changed')
    for key in actual:
        _same(actual[key], cached[key], '057 complete learned system/' + key)
    if expected is not None:
        assert_metrics(report['full_target_metrics'], expected, '057 original learned physical metrics')
    result.update(complete_inputs_and_ordinary_arrays_exact=True,
                  complete_old_learned_artifact_exact=True)
    return result
