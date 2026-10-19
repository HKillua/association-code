"""Complete development inference on registered or native-time physical scenes.

One TCM pass feeds both association and fusion. References enter only the
evaluation call after predictions; this module does not register native tracks.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from association.evaluation.scene import evaluate_full_scene, simple_fusion_outputs, single_source_outputs, INPUT_SUPPORT_REGIONS, INPUT_DISTANCE_REGIONS
from association.evaluation.grouping import group_predicted_tracks, group_tracks_by_identity
from association.evaluation.support import build_input_support
from association.evaluation.association import evaluate_scenes, select_threshold
from association.training.common import normalize_scene, stable_evaluation_parameters


def aggregate_fusion_reports(reports):
    """Pool error numerators and coverage counts, never average scene RMSEs."""
    result = _aggregate_regions(reports, 'full_target_metrics', ('all', 'observed', 'missing'))
    for key, names in (('input_support_metrics', INPUT_SUPPORT_REGIONS),
                       ('input_distance_metrics', INPUT_DISTANCE_REGIONS)):
        if any(key in report for report in reports):
            if not all(key in report for report in reports):
                raise ValueError('cannot pool scenes with incomplete input-support diagnostics')
            result[key] = _aggregate_regions(reports, key, names)
    return result


def _aggregate_regions(reports, metric_key, regions):
    result = {}
    for region in regions:
        result[region] = {}
        for feature in ('position', 'speed', 'heading'):
            rows = [report[metric_key][region][feature] for report in reports]
            count_keys = ('available_target_count', 'available_prediction_count',
                          'intersection_count', 'missing_prediction_count',
                          'prediction_without_target_count', 'available_target_trajectory_count',
                          'covered_target_trajectory_count', 'missing_target_trajectory_count')
            pooled = {key: sum(row[key] for row in rows) for key in count_keys}
            n = pooled['intersection_count']
            for key in ('mae', 'mse'):
                pooled[key] = sum(row[key] * row['intersection_count'] for row in rows
                                  if row['intersection_count']) / n if n else None
            pooled['rmse'] = math.sqrt(pooled['mse']) if n else None
            for ratio, numerator, denominator in (
                    ('coverage', 'intersection_count', 'available_target_count'),
                    ('trajectory_coverage', 'covered_target_trajectory_count', 'available_target_trajectory_count')):
                pooled[ratio] = pooled[numerator] / pooled[denominator] if pooled[denominator] else None
            pooled['unit'] = {'position': 'metres', 'speed': 'native_speed', 'heading': 'degrees'}[feature]
            result[region][feature] = pooled
    return result


def scene_input_support(scene):
    """Visible encoder inputs in physical seconds; no target/reference access."""
    if scene.get('input_adapter') == 'native_time_v1':
        return build_input_support(scene['grid'], scene['native_times'], scene['native_observed'])
    return build_input_support(scene['grid'], scene['observation_times'], scene['observed'],
                               node_support=scene['observed'])


def _registered(scene, length):
    if scene.get('split') != 'V_select':
        raise ValueError('complete development evaluation requires V_select; T_test is sealed')
    timed = scene.get('input_adapter') == 'native_time_v1'
    if timed:
        from association.data.timed import validate_timed_scene
        validate_timed_scene(scene)
    values = np.asarray(scene['values'])
    observed = np.asarray(scene['observed'])
    grid = np.asarray(scene['grid'], dtype=np.float64)
    actual = np.asarray(scene['observation_times'], dtype=np.float64)
    if (values.ndim != 3 or values.shape[1:] != (length, 4)
            or observed.shape != values.shape[:2] or observed.dtype != bool
            or grid.shape != (length,) or actual.shape != observed.shape
            or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any()):
        raise ValueError('invalid registered scene shape, mask or physical grid')
    if len(values) and ((not timed and (~observed.any(axis=1)).any()) or not np.isfinite(values[observed]).all()):
        raise ValueError('every valid fragment must supply finite observations')
    if not np.allclose(actual[observed], np.broadcast_to(grid, observed.shape)[observed], rtol=0, atol=1e-8):
        raise ValueError('observation time differs from physical grid; legacy scene64 registration is not accepted')
    pairs = np.asarray(scene['pairs'])
    expected_pairs = np.column_stack(np.triu_indices(len(values), 1))
    if pairs.dtype.kind not in 'iu' or not np.array_equal(pairs, expected_pairs):
        raise ValueError('full-scene evaluation requires all canonical unordered candidate pairs')
    identities, labels = np.asarray(scene['identities']), np.asarray(scene['labels'])
    if (identities.shape != (len(values),) or labels.shape != (len(pairs),)
            or not np.array_equal(labels, identities[pairs[:, 0]] == identities[pairs[:, 1]])):
        raise ValueError('full-scene labels must match the true identities for every candidate')
    targets = scene.get('fusion_targets')
    if not isinstance(targets, dict):
        raise ValueError('fusion_targets must explicitly declare aligned references, possibly an empty dict')
    for target in targets.values():
        if (not isinstance(target, dict) or 'query_times' not in target
                or not np.array_equal(np.asarray(target['query_times'], dtype=np.float64), grid)):
            raise ValueError('every fusion target query_times must exactly match the physical scene grid')
    return values, observed


def _inference_device(device, model_device, fusion_device):
    """Resolve generic CUDA to the model's actual device without a GPU call."""
    requested, model_device, fusion_device = map(torch.device, (device, model_device, fusion_device))
    if requested.type == 'cuda' and requested.index is None and model_device.type == 'cuda':
        requested = model_device
    if model_device != requested or fusion_device != requested:
        raise ValueError('models must be on the requested device')
    return requested


@stable_evaluation_parameters
@torch.inference_mode()
def evaluate_complete(model, fusion, normalizer, scenes, *, device,
                      batch_tracks, pair_batch_size, select_on_validation,
                      threshold, singleton_policy, check=None, progress=None,
                      prediction_writer=None):
    """Return full-V association plus predicted/oracle/simple/single-source fusion.

    ``singleton_policy`` is explicitly 'missing' or 'observed'. The latter
    passes through the singleton's aligned physical observations only; it does
    not decode a completed latent as a physical trajectory. References never
    pick an output, repair a predicted group or feed recursive fusion. The
    oracle, simple-fusion and single-source rows use true-identity grouping and
    are diagnostics, not independent association algorithms.

    Model modes are set to eval and left there. Training must explicitly restore
    its current stage. Cached latents live on CPU only for this call. Threshold
    fitting uses all supplied V scenes together, never per scene or on T_test.
    Optional progress receives small stage/scene records, never arrays; a caller
    can preserve the last completed boundary even if evaluation is interrupted.
    An optional prediction_writer receives one scene's actual metric arrays;
    export I/O is within the evaluation budget and never alters the summaries.
    """
    for value in (batch_tracks, pair_batch_size):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('inference batch sizes must be positive integers')
    if singleton_policy not in ('missing', 'observed'):
        raise ValueError('singleton_policy must explicitly be missing or observed')
    if not isinstance(select_on_validation, bool):
        raise ValueError('select_on_validation must be boolean')
    if select_on_validation and threshold is not None:
        raise ValueError('provide threshold=None when fitting on V_select')
    if (model.config.length, model.config.d_model) != (fusion.config.length, fusion.config.d_model):
        raise ValueError('association and fusion dimensions must agree')
    scenes = list(scenes)
    if not scenes or len({str(s['scene_id']) for s in scenes}) != len(scenes):
        raise ValueError('a nonempty list of distinct validation scenes is required')
    adapter = getattr(model, 'input_adapter', 'registered_v1')
    if any(scene.get('input_adapter', 'registered_v1') != adapter for scene in scenes):
        raise ValueError('all validation scenes must match the model input adapter')
    for scene in scenes:
        _registered(scene, model.config.length)
    device = _inference_device(device, next(model.parameters()).device, next(fusion.parameters()).device)
    dtype = next(model.parameters()).dtype
    if next(fusion.parameters()).dtype != dtype:
        raise ValueError('association and fusion dtype must agree')
    started = time.monotonic()
    def report_progress(stage, completed_scenes, scene_id=None, **details):
        if progress is not None:
            progress(dict(stage=stage, completed_scenes=completed_scenes,
                          total_scenes=len(scenes), scene_id=scene_id,
                          elapsed_seconds=time.monotonic()-started, **details))
    model.eval()
    fusion.eval()
    records, cached = [], []
    for scene_index, scene in enumerate(scenes):
        report_progress('association_scoring', scene_index, str(scene['scene_id']))
        if check:
            check()
        x, mask = normalize_scene(scene, normalizer)
        completed, encoded = [], []
        for offset in range(0, len(x), batch_tracks):
            if check:
                check()
            latent = model.complete(torch.as_tensor(x[offset:offset+batch_tracks], device=device, dtype=dtype),
                                    torch.as_tensor(mask[offset:offset+batch_tracks], device=device))
            embedding = model.tam_encoder(latent).flatten(1)
            if not torch.isfinite(latent).all() or not torch.isfinite(embedding).all():
                raise FloatingPointError('nonfinite completed or association features')
            completed.append(latent.cpu())
            encoded.append(embedding.cpu())
        latents = torch.cat(completed) if completed else torch.empty(0, model.config.length, model.config.d_model, dtype=dtype)
        embeddings = torch.cat(encoded) if encoded else torch.empty(0, model.embedding_dim, dtype=dtype)
        pairs = np.asarray(scene['pairs'])
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu':
            raise ValueError('pairs must be integer [E,2]')
        if len(pairs) and ((pairs < 0).any() or (pairs >= len(x)).any() or (pairs[:, 0] >= pairs[:, 1]).any()):
            raise ValueError('invalid candidate pair')
        scores = []
        for offset in range(0, len(pairs), pair_batch_size):
            if check:
                check()
            nodes, inverse = np.unique(pairs[offset:offset+pair_batch_size], return_inverse=True)
            value = model.score_pairs(embeddings[nodes].to(device), inverse.reshape(-1, 2))
            scores.append(value.cpu().numpy())
        record = {key: scene[key] for key in ('scene_id', 'component_id', 'node_ids', 'identities', 'sources', 'intervals', 'labels')}
        record.update(pairs=pairs, scores=np.concatenate(scores) if scores else np.empty(0))
        records.append(record)
        cached.append(latents)
        report_progress('association_scoring', scene_index+1)
    selection = select_threshold(records) if select_on_validation else None
    if selection is not None:
        threshold = selection['threshold']
    association = evaluate_scenes(records, threshold)
    sources = sorted({str(source) for scene in scenes for source in scene['sources']})
    methods = {key: [] for key in ['predicted', 'oracle', 'simple_fusion_oracle']
               + ['single_source:' + source for source in sources]}
    per_scene = []
    fusion_steps = 0
    for scene_index, (scene, record, latent) in enumerate(zip(scenes, records, cached)):
        report_progress('physical_fusion', scene_index, str(scene['scene_id']),
                        fusion_forward_count=fusion_steps)
        targets = scene['fusion_targets']
        target_ids = list(targets)
        # Empty scenes still belong to the declared development population.
        # Nonempty nodes without target rows are rejected by evaluate_full_scene.
        truth = (np.stack([targets[identity]['states'] for identity in target_ids])
                 if target_ids else np.empty((0, model.config.length, 4)))
        valid = (np.stack([targets[identity]['valid'] for identity in target_ids])
                 if target_ids else np.empty((0, model.config.length, 4), dtype=bool))
        common = dict(node_identities=scene['identities'], target_identities=target_ids,
                      target_states=truth, target_valid=valid, node_observed=scene['observed'],
                      split=scene['split'], correspondence_policy='anchor_identity_first',
                      return_artifact=prediction_writer is not None)
        support = scene_input_support(scene)
        outputs = {}
        groupings = {
            'predicted': group_predicted_tracks(len(latent), record['pairs'], record['scores'],
                threshold=threshold, grouping_policy='anchor_direct', node_order='input'),
            'oracle': group_tracks_by_identity(scene['identities'], node_order='input'),
        }
        for kind, groups in groupings.items():
            generated = []
            for members in groups:
                if check:
                    check()
                row = dict(node_indices=members, states=None)
                if len(members) == 1 and singleton_policy == 'observed':
                    row.update(states=np.asarray(scene['values'])[members[0]],
                               prediction_valid=np.repeat(np.asarray(scene['observed'])[members[0], :, None], 4, axis=1))
                elif len(members) > 1:
                    memory = latent[members[0]:members[0]+1].to(device)
                    for node in members[1:]:
                        if check:
                            check()
                        fused = fusion(memory, latent[node:node+1].to(device))
                        memory = fused['fused_features']
                        fusion_steps += 1
                    row['states'] = fused['states'][0].cpu().numpy()
                generated.append(row)
            outputs[kind] = evaluate_full_scene(generated, grouping_kind=kind,
                                                node_input_support=support, **common)
        node_valid = np.repeat(np.asarray(scene['observed'])[..., None], 4, axis=-1)
        simple = simple_fusion_outputs(scene['values'], node_valid, scene['identities'], scene['sources'])
        outputs['simple_fusion_oracle'] = evaluate_full_scene(simple, grouping_kind='oracle', **common)
        for source in sources:
            single = single_source_outputs(scene['values'], node_valid, scene['identities'], scene['sources'], source=source)
            outputs['single_source:' + source] = evaluate_full_scene(single, grouping_kind='oracle', **common)
        if prediction_writer is not None:
            artifacts = {key: report.pop('artifact') for key, report in outputs.items()}
            if check:
                check()
            prediction_writer(scene_index, scene, artifacts)
            if check:
                check()
            del artifacts
        for key, report in outputs.items():
            methods[key].append(report)
        per_scene.append(dict(scene_id=str(scene['scene_id']), methods=outputs))
        report_progress('physical_fusion', scene_index+1, fusion_forward_count=fusion_steps)
    if check:
        check()
    fusion_report = {key: aggregate_fusion_reports(reports) for key, reports in methods.items()}
    if check:
        check()
    report_progress('completed', len(scenes), fusion_forward_count=fusion_steps)
    return dict(association=association, selection=selection, threshold=threshold, records=records,
                fusion=fusion_report,
                per_scene=per_scene, singleton_policy=singleton_policy, feedback_policy='fc_memory',
                grouping_policy='anchor_direct', correspondence_policy='anchor_identity_first',
                elapsed_seconds=time.monotonic()-started, fusion_forward_count=fusion_steps,
                test_opened=False, scope='V_select development; oracle, simple-fusion and single-source are diagnostics')


@stable_evaluation_parameters
@torch.inference_mode()
def evaluate_tcm_completion(model, normalizer, scenes, *, device='cpu', check=None, progress=None):
    """Diagnose the saved native TCM in its own frozen feature space, without TAM.

    Native teacher-forced steps and actual full-query rollout are separate. The
    state-hold control re-embeds a visible state at the requested target time;
    it never holds the old time encoding. Inside support both query methods use
    the existing observation interpolation, which is not a learned prediction.
    All four reference attributes must be valid for a query feature target.
    """
    from association.models.timed import continuous_time_encoding
    scenes = list(scenes)
    if (not scenes or len({str(s['scene_id']) for s in scenes}) != len(scenes)
            or getattr(model, 'input_adapter', None) != 'native_time_v1'):
        raise ValueError('native TCM and distinct nonempty V scene population required')
    parameter = next(model.parameters())
    device = _inference_device(device, parameter.device, parameter.device)
    dtype, d = parameter.dtype, model.config.d_model
    model.eval()
    started, per_fragment, per_scene = time.monotonic(), [], []
    region_names = ('all', 'inside_support', 'before_support', 'after_support')
    count_keys = ('query_count', 'evaluable_query_count', 'missing_reference_query_count', 'feature_elements')

    def metric(error, mask, valid):
        selected = np.asarray(mask, bool) & valid
        count, total = int(selected.sum()), int(np.asarray(mask).sum())
        square_sum = float(error[selected].sum())
        return dict(query_count=total, evaluable_query_count=count,
                    missing_reference_query_count=total-count, feature_elements=count*d,
                    squared_error_sum=square_sum, feature_mse=square_sum/(count*d) if count else None,
                    reference_coverage=count/total if total else None)

    def pool(rows):
        totals = {key: sum(row[key] for row in rows) for key in count_keys}
        square_sum = sum(row['squared_error_sum'] for row in rows)
        return dict(**totals, squared_error_sum=square_sum,
                    feature_mse=square_sum/totals['feature_elements'] if totals['feature_elements'] else None,
                    reference_coverage=totals['evaluable_query_count']/totals['query_count'] if totals['query_count'] else None)

    def embed(states, times):
        return torch.relu(model.input_embedding(states + continuous_time_encoding(times, 4)))

    def errors(prediction, target):
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise FloatingPointError('nonfinite TCM completion feature')
        return (prediction.double()-target.double()).square().sum(-1).cpu().numpy()

    for scene_index, scene in enumerate(scenes):
        if check:
            check()
        if scene.get('input_adapter') != 'native_time_v1':
            raise ValueError('TCM diagnostic requires native-time scenes')
        _registered(scene, model.config.length)
        packed, observed = normalize_scene(scene, normalizer)
        support = scene_input_support(scene)
        begin = len(per_fragment)
        for node, identity in enumerate(scene['identities']):
            if check:
                check()
            x = torch.as_tensor(packed[node:node+1], device=device, dtype=dtype)
            mask = torch.as_tensor(observed[node:node+1], device=device)
            count = int(scene['native_observed'][node].sum())
            native, times = x[:, :count, :4], x[:, :count, 4]
            embedded = model.embed_inputs(x, mask)[:, :count]
            one_step = {name: {} for name in ('model', 'state_hold')}
            for direction, sequence, states, tau in (
                    ('forward', embedded, native, times),
                    ('reverse', embedded.flip(1), native.flip(1), times.flip(1))):
                target = sequence[:, 1:]
                prediction = model._teacher_predictions(sequence[:, :-1], tau[:, 1:])
                held = embed(states[:, :-1], tau[:, 1:])
                full = np.ones(count-1, bool)
                for name, value in (('model', prediction), ('state_hold', held)):
                    one_step[name][direction] = metric(errors(value, target)[0], full, full)
            for result in one_step.values():
                result['all'] = pool([result['forward'], result['reverse']])
            completed = model.complete(x, mask)[0]
            query_tau = x[0, model.max_observations:, 4]
            before = scene['grid'] < scene['native_times'][node, 0]
            after = scene['grid'] > scene['native_times'][node, count-1]
            held_states = x[0, model.max_observations:, :4].clone()
            held_states[torch.as_tensor(before, device=device)] = native[0, 0]
            held_states[torch.as_tensor(after, device=device)] = native[0, -1]
            held = embed(held_states, query_tau)
            # Reference is read only after all model and control predictions.
            target_row = scene['fusion_targets'].get(identity)
            valid = (np.asarray(target_row['valid'], bool).all(-1) if target_row is not None
                     else np.zeros(model.config.length, bool))
            target_states = np.zeros((model.config.length, 4), dtype=np.float64)
            if target_row is not None:
                target_states[valid] = ((np.asarray(target_row['states'])[valid]-np.asarray(normalizer['mean']))
                                        / np.asarray(normalizer['std']))
            target = embed(torch.as_tensor(target_states, device=device, dtype=dtype), query_tau)
            masks = dict(zip(region_names, (np.ones(model.config.length, bool), support['support'][node], before, after)))
            nearest = support['nearest_seconds'][node]
            distances = dict(zip(INPUT_DISTANCE_REGIONS, (nearest <= 300, (nearest > 300) & (nearest <= 600),
                (nearest > 600) & (nearest <= 1800), (nearest > 1800) & np.isfinite(nearest), ~np.isfinite(nearest))))
            query = {}
            for name, prediction in (('model', completed), ('state_hold', held)):
                error = errors(prediction, target)
                query[name] = dict(regions={key: metric(error, region, valid) for key, region in masks.items()},
                    input_distance_metrics={key: metric(error, region, valid) for key, region in distances.items()})
            per_fragment.append(dict(scene_id=str(scene['scene_id']), node_id=str(scene['node_ids'][node]),
                node_index=node, identity=str(identity), native_count=count,
                reference_declared=target_row is not None, has_complete_reference=bool(valid.any()),
                one_step=one_step, query=query))
        per_scene.append(dict(scene_id=str(scene['scene_id']), fragment_count=len(per_fragment)-begin,
            no_complete_reference_fragment_count=sum(not row['has_complete_reference'] for row in per_fragment[begin:]),
            reference_targets_without_input=[str(identity) for identity in scene['fusion_targets']
                                             if identity not in set(scene['identities'])]))
        if progress:
            progress(dict(stage='tcm_completion', completed_scenes=scene_index+1, total_scenes=len(scenes),
                          completed_fragments=len(per_fragment), elapsed_seconds=time.monotonic()-started))
    if check:
        check()
    return dict(scene_count=len(scenes), fragment_count=len(per_fragment), per_scene=per_scene, per_fragment=per_fragment,
        one_step={name: {direction: pool([row['one_step'][name][direction] for row in per_fragment])
                        for direction in ('all', 'forward', 'reverse')} for name in ('model', 'state_hold')},
        query={name: {kind: {region: pool([row['query'][name][kind][region] for row in per_fragment])
                            for region in names} for kind, names in (('regions', region_names),
                            ('input_distance_metrics', INPUT_DISTANCE_REGIONS))} for name in ('model', 'state_hold')},
        no_complete_reference_fragment_count=sum(not row['has_complete_reference'] for row in per_fragment),
        scope='V_select frozen TCM feature diagnostics; no physical decoding, association, training or selection',
        target_policy='query target requires all four reference attributes; unchanged masks and frozen training normalization',
        control='visible state re-embedded at next/query tau; endpoint hold outside, shared observation interpolation inside support',
        interpretation='inside-support query output is observation interpolation, not learned completion; feature MSE is not physical error',
        elapsed_seconds=time.monotonic()-started, test_opened=False)
