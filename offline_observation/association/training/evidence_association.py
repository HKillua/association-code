"""Bounded training of a small residual score on frozen native OLS evidence."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from association.data.motion_evidence import DATA, prepare_motion_scenes, validate_motion_observation_policy
from association.evaluation.association import evaluate_scenes, select_threshold
from association.evaluation.grouping import group_predicted_tracks
from association.evaluation.physical import group_membership_diagnostics
from association.models.evidence_association import (
    EvidenceAssociation, FEATURE_NAMES, FIXED_OLS_SCORE, VARIANTS, features_from_motion)
from association.training.common import _update
from association.training.sampling import BalancedScenePairSampler
from association.training.paired_sampling import PairedNegativeIntervention, validate_sampling_config
from association.training.association_prefix import PrefixReference
from association.utils import runtime


SCOPE = 'native-ols-evidence-association-v1'
PRECISION = ('native feature calculation/base float64; feature cache float32; training normalizer '
             'streaming float64; normalized network input/parameters float32; summed score float64')


def validate_config(config):
    if (config.get('scope') != SCOPE or config.get('variant') not in VARIANTS
            or config.get('data') != DATA):
        raise ValueError('explicit native OLS evidence scope, variant and frozen data settings required')
    validate_motion_observation_policy(config.get('motion_observation_policy', 'selected_cap128'))
    if isinstance(config.get('seed'), bool) or not isinstance(config.get('seed'), int) or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    training = config['training']
    for key in ('updates', 'batch_size', 'validate_every', 'inference_batch_size'):
        if isinstance(training[key], bool) or not isinstance(training[key], int) or training[key] < 1:
            raise ValueError(f'{key} must be positive integer')
    if training['batch_size'] % 2:
        raise ValueError('balanced sampler requires even batch_size')
    if 'sampling' in config:
        validate_sampling_config(config['sampling'])
        if training['batch_size'] != 256:
            raise ValueError('paired negative intervention requires the frozen batch_size 256')
    for key in ('learning_rate', 'gradient_clip', 'advisory_hours'):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive finite')
    return training


def _validate_partitions(train, validation):
    """Keep scene separation in both protocols; shared-A isolation is explicit."""
    if (not train or not validation or any(s['split'] != 'T_core' for s in train)
            or any(s['split'] != 'V_select' for s in validation)):
        raise ValueError('nonempty T_core/V_select required; T_test sealed')
    units = {s.get('split_unit', 'shared_a_component') for s in [*train, *validation]}
    if len(units) != 1 or not units.issubset({'shared_a_component', 'scene'}):
        raise ValueError('train/validation must use one explicit supported split_unit')
    unit = next(iter(units))
    ids = [str(s['scene_id']) for s in [*train, *validation]]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate or training/validation scene_id overlap')
    if unit == 'shared_a_component' and (
            {str(s['component_id']) for s in train} & {str(s['component_id']) for s in validation}):
        raise ValueError('training/validation component_id overlap')
    return unit


def prepare_features(scenes, split, check=lambda: None, pair_chunk_size=4096):
    """Freeze all original pairs; truth enters metadata but not feature calls."""
    if isinstance(pair_chunk_size, bool) or not isinstance(pair_chunk_size, int) or pair_chunk_size < 1:
        raise ValueError('pair chunk size must be positive')
    result = []
    for scene in scenes:
        check()
        n = len(scene['node_ids'])
        pairs = np.asarray(scene['pairs'])
        if (scene['split'] != split or pairs.shape != (n*(n-1)//2, 2)
                or pairs.dtype.kind not in 'iu' or (pairs < 0).any() or (pairs >= n).any()
                or (pairs[:, 0] >= pairs[:, 1]).any()
                or (len(pairs) > 1 and ((pairs[1:, 0] < pairs[:-1, 0])
                    | ((pairs[1:, 0] == pairs[:-1, 0]) & (pairs[1:, 1] <= pairs[:-1, 1]))).any())):
            raise ValueError('explicit split and complete original unordered candidate population required')
        features = np.empty((len(pairs), len(FEATURE_NAMES)), np.float32)
        base = np.empty(len(pairs), np.float64)
        joint = np.empty(len(pairs), np.int64)
        support = scene['motion']['support']
        max_roundoff = 0.
        for start in range(0, len(pairs), pair_chunk_size):
            check()
            stop = min(start+pair_chunk_size, len(pairs))
            block, base[start:stop] = features_from_motion(scene['motion'], pairs[start:stop])
            features[start:stop] = block
            selected_pairs = pairs[start:stop]
            joint[start:stop] = (support[selected_pairs[:, 0]] & support[selected_pairs[:, 1]]).sum(1)
            max_roundoff = max(max_roundoff, float(np.max(np.abs(
                features[start:stop].astype(np.float64)-block))))
        identities = np.asarray(scene['identities'])
        labels = np.asarray(scene['labels'])
        if identities.shape != (n,) or labels.shape != (len(pairs),):
            raise ValueError('original identity labels and candidate order disagree')
        for start in range(0, len(pairs), pair_chunk_size):
            selected_pairs = pairs[start:start+pair_chunk_size]
            if not np.array_equal(labels[start:start+pair_chunk_size],
                    identities[selected_pairs[:, 0]] == identities[selected_pairs[:, 1]]):
                raise ValueError('original identity labels and candidate order disagree')
        result.append(dict(**{key: scene[key] for key in (
            'scene_id', 'component_id', 'split', 'node_ids', 'sources', 'identities', 'pairs', 'labels')},
            features=features, base_scores=base,
            feature_cache_max_abs_roundoff=max_roundoff,
            joint_support_queries=joint))
        if 'split_unit' in scene:
            result[-1]['split_unit'] = scene['split_unit']
        if 'motion_observation_policy' in scene:
            result[-1]['motion_observation_policy'] = validate_motion_observation_policy(scene['motion_observation_policy'])
    return result


def fit_normalizer(train_scenes):
    if not train_scenes or any(scene['split'] != 'T_core' for scene in train_scenes):
        raise ValueError('normalizer must use T_core only')
    count = 0
    mean = np.zeros(len(FEATURE_NAMES), np.float64)
    m2 = np.zeros_like(mean)
    for scene in train_scenes:
        for start in range(0, len(scene['features']), 4096):
            values = scene['features'][start:start+4096].astype(np.float64)
            size = len(values)
            local_mean = values.mean(0)
            local_m2 = np.square(values-local_mean).sum(0)
            total = count+size
            delta = local_mean-mean
            m2 += local_m2+delta*delta*(count*size/total)
            mean += delta*(size/total)
            count = total
    if not count:
        raise ValueError('training has no pairs')
    std = np.sqrt(np.maximum(m2/count, 0.))
    return dict(feature_names=list(FEATURE_NAMES), mean=mean.tolist(),
                std=np.where(std < 1e-6, 1., std).tolist(), pair_count=count,
                fitted_split='T_core', weighting='one weight per original training pair',
                algorithm='4096-pair chunk Welford, float64 accumulation of float32 cached features')


def normalized_features(features, normalizer):
    if (normalizer.get('feature_names') != list(FEATURE_NAMES)
            or normalizer.get('fitted_split') != 'T_core'):
        raise ValueError('normalizer feature order or training role mismatch')
    mean, std = np.asarray(normalizer['mean']), np.asarray(normalizer['std'])
    if (mean.shape != (len(FEATURE_NAMES),) or std.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('invalid frozen training feature normalizer')
    result = ((features-mean)/std).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError('feature normalization exceeded finite float32')
    return result


def _auc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, np.float64)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends-counts+1+ends)/2
    return float((ranks[inverse][labels].sum()-positive*(positive+1)/2)/(positive*negative))


def _subset(records, joint):
    result = []
    for record in records:
        mask = (record['joint_support_queries'] > 0) == joint
        result.append({**record, **{key: record[key][mask] for key in
            ('scores', 'labels', 'pairs', 'joint_support_queries')}})
    return result


def grouping_summary(records, threshold):
    """Use the actual partition rule; identities are read only for diagnostics."""
    rows = []
    for record in records:
        groups = group_predicted_tracks(len(record['node_ids']), record['pairs'], record['scores'],
            threshold=threshold, grouping_policy='anchor_direct', node_order='input')
        membership = np.empty(len(record['node_ids']), dtype=np.int64)
        bad_anchors = bad_helpers = mixed = 0
        truth = np.asarray(record['identities'])
        for gid, group in enumerate(groups):
            members = np.asarray(group)
            membership[members] = gid
            mixed += int(len(np.unique(truth[members])) > 1)
            wrong = truth[members, None] != truth[None, members]
            bad_anchors += int(wrong.any(1).sum())
            bad_helpers += int(wrong.sum())
        rows.append(dict(scene_id=record['scene_id'], group_count=len(groups), mixed_group_count=mixed,
            anchors_with_wrong_helper=bad_anchors, directed_wrong_helper_relations=bad_helpers,
            membership=membership.tolist(),
            partition=group_membership_diagnostics(membership.tolist(), truth.tolist())))
    keys = ('group_count', 'mixed_group_count', 'anchors_with_wrong_helper',
            'directed_wrong_helper_relations')
    return dict(policy='anchor_direct/input; every fragment is an anchor in helper diagnostics',
                **{key: sum(row[key] for row in rows) for key in keys}, per_scene=rows)


def summarize_records(records, threshold):
    labels = np.concatenate([record['labels'] for record in records])
    scores = np.concatenate([record['scores'] for record in records])
    return dict(metrics=evaluate_scenes(records, threshold), pooled_auc=_auc(labels, scores),
        support={name: evaluate_scenes(_subset(records, has_support), threshold)
                 for name, has_support in (('joint_q_support', True), ('no_joint_q_support', False))},
        grouping=grouping_summary(records, threshold))


@torch.no_grad()
def score_scenes(model, scenes, normalizer, device, batch_size, check=lambda: None):
    model.eval()
    records = []
    for scene in scenes:
        if scene.get('motion_observation_policy', 'selected_cap128') != getattr(
                model, 'motion_observation_policy', 'selected_cap128'):
            raise ValueError('association model and scene motion_observation_policy differ')
        chunks = []
        for start in range(0, len(scene['pairs']), batch_size):
            check()
            stop = start+batch_size
            values = torch.as_tensor(normalized_features(scene['features'][start:stop], normalizer), device=device)
            base = torch.as_tensor(scene['base_scores'][start:stop], dtype=torch.float64, device=device)
            chunks.append(model(values, base).cpu().numpy())
        scores = np.concatenate(chunks) if chunks else np.empty(0, np.float64)
        if not np.isfinite(scores).all():
            raise FloatingPointError('association scores must be finite on every pair')
        records.append({**{key: scene[key] for key in (
            'scene_id', 'component_id', 'node_ids', 'sources', 'identities',
            'pairs', 'labels', 'joint_support_queries')}, 'scores': scores})
    return records


def _save_scores(path, records):
    values = {'scene_ids': np.asarray([row['scene_id'] for row in records], dtype=str)}
    for row in records:
        for key in ('pairs', 'labels', 'scores', 'node_ids', 'sources', 'joint_support_queries'):
            values[f"{row['scene_id']}_{key}"] = np.asarray(row[key])
    np.savez_compressed(path, **values)


def _save_checkpoint(path, model, optimizer, config, normalizer, update, selection):
    snapshot = dict(scope=SCOPE, feature_names=list(FEATURE_NAMES), config=deepcopy(config),
        motion_observation_policy=config.get('motion_observation_policy', 'selected_cap128'),
        normalizer=deepcopy(normalizer), update=update, selection=deepcopy(selection),
        state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        optimizer_state=deepcopy(optimizer.state_dict()), precision=PRECISION,
        score_semantics='association score, not a calibrated probability')
    temp = Path(str(path)+'.tmp')
    torch.save(snapshot, temp)
    temp.replace(path)


def load_checkpoint(path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get('scope') != SCOPE or saved.get('feature_names') != list(FEATURE_NAMES):
        raise ValueError('checkpoint scope or feature order mismatch')
    validate_config(saved['config'])
    policy = saved['config'].get('motion_observation_policy', 'selected_cap128')
    if saved.get('motion_observation_policy', 'selected_cap128') != policy:
        raise ValueError('checkpoint motion_observation_policy disagrees with configuration')
    normalized_features(np.zeros((0, len(FEATURE_NAMES))), saved['normalizer'])
    model = EvidenceAssociation(saved['config']['variant']).to(device)
    model.motion_observation_policy = policy
    model.load_state_dict(saved['state_dict'], strict=True)
    expected = np.ones(len(FEATURE_NAMES), np.float32)
    if model.variant == 'geometry_only':
        expected[7:] = 0.
    if not np.array_equal(model.feature_mask.cpu().numpy(), expected):
        raise ValueError('checkpoint feature mask disagrees with declared variant')
    model.eval()
    return model, saved


def fit_evidence_association(config, output_dir, device, *, archive_path=None, manifest=None,
                             train_scenes=None, validation_scenes=None, synthetic=False,
                             command=None, prefix_reference=None):
    """One fit; exactly fixed updates, step-zero selection, advisory time only."""
    settings = validate_config(config)
    policy = config.get('motion_observation_policy', 'selected_cap128')
    device = runtime.device_from_string(str(device))
    run = runtime.Run(output_dir, dict(kind=SCOPE, config=config, command=command,
        provenance=runtime.provenance(device), precision=PRECISION, synthetic=synthetic,
        reference_read=False, test_opened=False, optimizer_updates=0,
        score_semantics='association score, not a calibrated probability'), max_hours=None)
    started = time.monotonic()
    advisory_written = False

    def check():
        nonlocal advisory_written
        if not advisory_written and time.monotonic()-started >= settings['advisory_hours']*3600:
            runtime.write_json(run.path/'advisory.json', dict(
                elapsed_seconds=time.monotonic()-started, fixed_updates=settings['updates'],
                action='suggested duration exceeded; notify user, continue original fixed updates; no automatic retry'))
            advisory_written = True

    actual_update = 0
    sampler = None
    intervention = None
    prefix = None
    prefix_result = None

    def persist_paired_sampling_audit():
        if intervention is None:
            return None
        coverage = intervention.audit()
        hard_per_update = 64 if config['sampling']['mode'] == 'hard_negative' else 0
        if (coverage['completed_batch_count'] != actual_update
                or coverage['hard_source_slots'] != actual_update * hard_per_update):
            raise AssertionError('paired sampling counts disagree with successful optimizer updates')
        coverage.update(actual_optimizer_updates=actual_update,
                        configured_updates=settings['updates'], batch_size=settings['batch_size'])
        runtime.write_json(run.path/'paired_sampling_audit.json', coverage)
        return {key: value for key, value in coverage.items() if key != 'scenes'} | {
            'per_scene_artifact': 'paired_sampling_audit.json'}

    def persist_sampling_audit():
        if sampler is None:
            return None
        coverage = sampler.coverage()
        if (coverage['completed_batch_count'] != actual_update
                or coverage['sampled_pair_count'] != actual_update * settings['batch_size']
                or coverage['sampled_positive_count'] != actual_update * settings['batch_size'] // 2
                or coverage['sampled_negative_count'] != actual_update * settings['batch_size'] // 2):
            raise AssertionError('training sampling counts disagree with successful optimizer updates')
        coverage.update(split='T_core', actual_optimizer_updates=actual_update,
                        configured_updates=settings['updates'], batch_size=settings['batch_size'])
        runtime.write_json(run.path/'sampling_audit.json', coverage)
        persist_paired_sampling_audit()
        return {key: value for key, value in coverage.items() if key != 'scenes'} | {
            'per_scene_artifact': 'sampling_audit.json'}

    try:
        if prefix_reference is not None:
            prefix = PrefixReference(prefix_reference, config, expected_precision=PRECISION, synthetic=synthetic)
            run.metadata['prefix_reference'] = str(prefix.path)
            runtime.write_json(run.path/'run.json', run.metadata)
        runtime.seed_everything(config['seed'])
        if synthetic:
            if archive_path is not None or manifest is not None or train_scenes is None or validation_scenes is None:
                raise ValueError('synthetic check requires supplied scenes and no native archive/manifest')
        else:
            if archive_path is None or manifest is None or train_scenes is not None or validation_scenes is not None:
                raise ValueError('native fit must prepare T_core/V_select from explicit archive/manifest')
            train_scenes = prepare_motion_scenes(archive_path, manifest, 'T_core', check,
                motion_observation_policy=policy)
            validation_scenes = prepare_motion_scenes(archive_path, manifest, 'V_select', check,
                motion_observation_policy=policy)
        if any(scene.get('motion_observation_policy', 'selected_cap128') != policy
               for scene in [*train_scenes, *validation_scenes]):
            raise ValueError('training configuration and prepared motion_observation_policy differ')
        split_unit = _validate_partitions(train_scenes, validation_scenes)
        raw_preparation_seconds = time.monotonic()-started
        train = prepare_features(train_scenes, 'T_core', check)
        validation = prepare_features(validation_scenes, 'V_select', check)
        feature_preparation_seconds = time.monotonic()-started-raw_preparation_seconds
        normalizer = fit_normalizer(train)
        sampler = BalancedScenePairSampler(train, config['seed'])
        if 'sampling' in config:
            intervention = PairedNegativeIntervention(train, config['sampling'])
        preparation_seconds = time.monotonic()-started
        audit = dict(split_unit=split_unit, training_scene_ids=[scene['scene_id'] for scene in train],
            validation_scene_ids=[scene['scene_id'] for scene in validation],
            normalizer=normalizer, sampler=sampler.describe(), data=DATA, motion_observation_policy=policy,
            inference_uses_identity=False, reference_read=False, test_opened=False,
            candidate_policy='all original unordered i<j pairs, no source/support filtering',
            training_labels='identity equality only for sampling and BCE; never a feature',
            validation_population=dict(scenes=len(validation),
                components=len({scene['component_id'] for scene in validation}),
                nodes=sum(len(scene['node_ids']) for scene in validation),
                pairs=sum(len(scene['pairs']) for scene in validation),
                positives=sum(int(np.sum(scene['labels'])) for scene in validation)),
            preparation_seconds=preparation_seconds)
        if intervention is not None:
            audit['paired_sampling_config'] = deepcopy(config['sampling'])
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        audit.update(raw_preparation_seconds=raw_preparation_seconds,
            feature_preparation_seconds=feature_preparation_seconds,
            feature_cache_dtype='float32', geometric_base_dtype='float64', feature_pair_chunk_size=4096,
            cached_feature_bytes=sum(scene['features'].nbytes for scene in train+validation),
            cached_base_score_bytes=sum(scene['base_scores'].nbytes for scene in train+validation),
            feature_cache_max_abs_roundoff=max(
                (scene['feature_cache_max_abs_roundoff'] for scene in train+validation), default=0.),
            preparation_peak_rss_bytes=int(rss if sys.platform == 'darwin' else rss*1024))
        runtime.write_json(run.path/'data_audit.json', audit)
        model = EvidenceAssociation(config['variant']).to(device)
        model.motion_observation_policy = policy
        optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
        if prefix is not None:
            runtime.write_json(run.path/'prefix-binding.json', prefix.bind(audit, validation, model, optimizer))
        best = None
        history = []
        training_seconds = validation_seconds = 0.
        final_records = None
        level1 = (manifest or {}).get('dataset_level') == 'Level1'
        fixed_ols_threshold = None if level1 else FIXED_OLS_SCORE
        fixed_ols_threshold_source = ('step-zero full V pure OLS working point' if level1 else
                                      'legacy 14.2 km OLS threshold; not a Level1 working point')
        for update in range(settings['updates']+1):
            check()
            row = dict(update=update)
            if update:
                step_started = time.monotonic()
                groups = sampler.sample(settings['batch_size'])
                if intervention is not None:
                    groups, sampling_record = intervention.apply(groups)
                features = np.concatenate([train[index]['features'][indices] for index, indices in groups])
                bases = np.concatenate([train[index]['base_scores'][indices] for index, indices in groups])
                labels = np.concatenate([train[index]['labels'][indices] for index, indices in groups])
                model.train()
                score = model(torch.as_tensor(normalized_features(features, normalizer), device=device),
                              torch.as_tensor(bases, dtype=torch.float64, device=device))
                loss = F.binary_cross_entropy_with_logits(score, torch.as_tensor(labels, dtype=torch.float64,
                                                                                  device=device))
                loss_value, grad_norm = _update(loss, optimizer, model.parameters(), settings['gradient_clip'])
                actual_update = update
                sampler.record_completed_batch(groups)
                if intervention is not None:
                    intervention.record_completed_batch(groups, sampling_record)
                row.update(loss=loss_value, gradient_norm=grad_norm,
                           train_seconds=time.monotonic()-step_started)
                training_seconds += row['train_seconds']
                if update == min(50, settings['updates']):
                    row['remaining_training_seconds_estimate'] = (
                        training_seconds/update*(settings['updates']-update))
                    row['timing_note'] = 'actual inclusive first steps; estimate only, excludes future evaluation'
            if update == 0 or update % settings['validate_every'] == 0 or update == settings['updates']:
                validation_started = time.monotonic()
                final_records = score_scenes(model, validation, normalizer, device,
                                            settings['inference_batch_size'], check)
                selection = select_threshold(final_records)
                row['selection'] = selection
                if best is None or selection['macro_f1'] > best['selection']['macro_f1']+1e-12:
                    best = dict(update=update, selection=selection)
                    _save_checkpoint(run.path/'selected.pt', model, optimizer, config, normalizer, update, selection)
                    _save_scores(run.path/'selected-scores.npz', final_records)
                if update == 0:
                    if any(not np.array_equal(record['scores'], scene['base_scores'])
                           for record, scene in zip(final_records, validation)):
                        raise AssertionError('zero residual did not preserve exact OLS scores')
                    _save_scores(run.path/'baseline-scores.npz', final_records)
                    if level1:
                        fixed_ols_threshold = selection['threshold']
                    runtime.write_json(run.path/'baseline.json', dict(selection=selection,
                        motion_observation_policy=policy,
                        fixed_ols_score_threshold=fixed_ols_threshold,
                        fixed_ols_threshold_source=fixed_ols_threshold_source,
                        selected_threshold=summarize_records(final_records, selection['threshold']),
                        fixed_ols_threshold=summarize_records(final_records, fixed_ols_threshold)))
                    runtime.write_json(run.path/'ols-working-point.json', dict(
                        scope='ols-working-point-v1', dataset_level=(manifest or {}).get('dataset_level'),
                        synthetic=synthetic, motion_observation_policy=policy, data=dict(DATA),
                        training_scene_ids=[s['scene_id'] for s in train],
                        validation_scene_ids=[s['scene_id'] for s in validation],
                        score_semantics='-log1p(rms_haversine_m/1000)',
                        threshold=selection['threshold'], selection_split='V_select',
                        selection_rule='pure OLS full V scene macro-F1, before any network update'))
                row['validation_seconds'] = time.monotonic()-validation_started
                validation_seconds += row['validation_seconds']
                persist_sampling_audit()
                if prefix is not None and update == prefix.update:
                    stem = f'prefix-step-{update}'
                    _save_checkpoint(run.path/(stem+'.pt'), model, optimizer, config, normalizer, update, selection)
                    _save_scores(run.path/(stem+'-scores.npz'), final_records)
                    prefix_sampling = runtime.read_json(run.path/'sampling_audit.json')
                    runtime.write_json(run.path/(stem+'-sampling_audit.json'), prefix_sampling)
                    runtime.write_json(run.path/(stem+'.json'), dict(update=update, selection=selection,
                        result=summarize_records(final_records, selection['threshold']),
                        checkpoint=stem+'.pt', scores=stem+'-scores.npz'))
                    prefix_result = prefix.compare(model, optimizer, final_records, selection, prefix_sampling)
                    runtime.write_json(run.path/'prefix-comparison.json', prefix_result)
            history.append(row)
            runtime.write_json(run.path/'history.json', history)
            runtime.write_json(run.path/'progress.json', dict(
                actual_optimizer_updates=actual_update, fixed_updates=settings['updates'],
                last_completed_update=update, selected=best,
                elapsed_seconds=time.monotonic()-started, advisory_issued=advisory_written))
        final_selection = select_threshold(final_records)
        _save_checkpoint(run.path/'final.pt', model, optimizer, config, normalizer,
                         actual_update, final_selection)
        _save_scores(run.path/'final-scores.npz', final_records)
        selected_model, selected_checkpoint = load_checkpoint(run.path/'selected.pt', device)
        selected_records = score_scenes(selected_model, validation, normalizer, device,
                                       settings['inference_batch_size'], check)
        report = dict(status='COMPLETED', scope=SCOPE, variant=config['variant'], split_unit=split_unit,
            motion_observation_policy=policy,
            synthetic=synthetic, data=DATA, precision=PRECISION,
            score_semantics='association score, not a calibrated probability',
            feature_names=list(FEATURE_NAMES), optimizer_updates=actual_update,
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            selection_rule='full V scene macro-F1; >= threshold; earliest update wins score ties including step 0',
            fixed_ols_score_threshold=fixed_ols_threshold,
            fixed_ols_threshold_source=fixed_ols_threshold_source,
            training_sampling=persist_sampling_audit(),
            population=audit['validation_population'], selected=best,
            selected_result=summarize_records(selected_records, best['selection']['threshold']),
            selected_fixed_ols_threshold=summarize_records(selected_records, fixed_ols_threshold),
            final_result=summarize_records(final_records, final_selection['threshold']),
            final_selection=final_selection,
            final_fixed_ols_threshold=summarize_records(final_records, fixed_ols_threshold),
            preparation_seconds=preparation_seconds, training_seconds=training_seconds,
            validation_seconds=validation_seconds,
            elapsed_seconds=time.monotonic()-started, reference_read=False, test_opened=False,
            limitations=['V reused for development checkpoint/threshold selection; not held-out evidence',
                         'frozen ordinary OLS completion, not learned completion or full system',
                         'balanced training score is not a calibrated posterior probability'])
        if intervention is not None:
            report['paired_sampling'] = persist_paired_sampling_audit()
        if prefix is not None:
            if prefix_result is None:
                raise AssertionError('fixed prefix comparison was not completed')
            report['prefix_comparison'] = {key: prefix_result[key] for key in
                ('comparison_status', 'causal_attribution_valid', 'bitwise_state_and_scores_equal',
                 'prefix_update', 'source_directory')}
            report['prefix_comparison']['artifact'] = 'prefix-comparison.json'
        if selected_checkpoint['update'] != best['update']:
            raise AssertionError('restored selected checkpoint differs from selected update')
        runtime.write_json(run.path/'summary.json', report)
        run.finish(optimizer_updates=actual_update, selected=best)
        return report
    except BaseException as error:
        audit_error = None
        try:
            persist_sampling_audit()
        except Exception as audit_failure:
            audit_error = f'{type(audit_failure).__name__}: {audit_failure}'
        status = 'INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED'
        run.finish(status, optimizer_updates=actual_update, error=f'{type(error).__name__}: {error}',
                   **({'sampling_audit_error': audit_error} if audit_error else {}))
        raise
