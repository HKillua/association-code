"""Full-population development evaluation for the native cooperative model.

Encodings are cached on CPU. Only an encoding microbatch, the nodes of a pair
microbatch, or one decoded group reside on the requested inference device.
Physical references enter metrics only, after each group's prediction.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .complete_evaluation import _inference_device, _registered, aggregate_fusion_reports
from .full_scene_evaluation import evaluate_full_scene, simple_fusion_outputs, single_source_outputs
from .fusion_inference import group_predicted_tracks, group_tracks_by_identity
from .metrics import evaluate_scenes, select_threshold
from .reference import normalize_scene, stable_evaluation_parameters


_METADATA = ('scene_id', 'component_id', 'node_ids', 'identities', 'sources', 'intervals', 'labels')
_ENCODED = ('memory', 'observed', 'times', 'queries', 'pooled')
_ROUTES = ('plain', 'identity', 'utility', 'support', 'learned')
_V2_ROUTES = ('plain', 'scalar-u', 'attribute-u', 'learned-attribute')


def _is_v2(model):
    return getattr(model.config, 'model_variant', 'legacy') == 'support_residual_v2'


def _validate(model, scenes, device, batch_tracks, pair_batch_size, use_utility_identity):
    for value in (batch_tracks, pair_batch_size):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('inference batch sizes must be positive integers')
    if not isinstance(use_utility_identity, bool):
        raise ValueError('use_utility_identity must be boolean')
    scenes = list(scenes)
    if not scenes or len({str(scene['scene_id']) for scene in scenes}) != len(scenes):
        raise ValueError('a nonempty list of distinct validation scenes is required')
    if getattr(model, 'input_adapter', None) != 'native_time_v1':
        raise ValueError('cooperative evaluation requires a native-time model')
    for scene in scenes:
        if scene.get('input_adapter') != 'native_time_v1':
            raise ValueError('cooperative evaluation requires native-time scenes')
        _registered(scene, model.config.length)
    parameter = next(model.parameters())
    # Like generic CUDA, generic MPS refers to the model's selected device.
    requested = torch.device(device)
    if requested.type == 'mps' and requested.index is None and parameter.device.type == 'mps':
        device = parameter.device
    device = _inference_device(device, parameter.device, parameter.device)
    return scenes, device, parameter.dtype


def _record(scene, scores):
    result = {key: scene[key] for key in _METADATA}
    result.update(pairs=np.asarray(scene['pairs']).copy(), scores=np.asarray(scores).copy())
    return result


def _fixed_records(scenes, records):
    records = list(records)
    if len(records) != len(scenes):
        raise ValueError('fixed records must preserve the entire validation scene population')
    result = []
    for scene, record in zip(scenes, records):
        if not isinstance(record, dict):
            raise ValueError('every fixed record must be a dictionary')
        for key in _METADATA:
            actual = np.asarray(record.get(key))
            if key == 'intervals' and isinstance(record.get(key), list) and not record[key]:
                actual = np.empty((0, 2))
            if key not in record or not np.array_equal(actual, np.asarray(scene[key])):
                raise ValueError(f'fixed record {key} differs from the validation scene')
        if 'pairs' not in record or 'scores' not in record:
            raise ValueError('fixed records require all candidate pairs and scores')
        pairs = np.asarray(record['pairs'])
        # Empty canonical arrays become [] in JSON; preserve that serialization.
        if isinstance(record['pairs'], list) and not record['pairs']:
            pairs = np.empty((0, 2), dtype=np.int64)
        if pairs.dtype.kind not in 'iu' or not np.array_equal(pairs, scene['pairs']):
            raise ValueError('fixed record candidate pairs differ or omit canonical edges')
        scores = np.asarray(record['scores'])
        if scores.shape != (len(pairs),) or scores.dtype.kind not in 'iuf' or not np.isfinite(scores).all():
            raise ValueError('fixed record scores must be finite numeric [E] values')
        result.append(_record(scene, scores))
    return result


def _take(encoded, nodes, device):
    indices = torch.as_tensor(nodes, dtype=torch.long)
    keys = _ENCODED + (('native_positions',) if 'native_positions' in encoded else ())
    return {key: encoded[key].index_select(0, indices).to(device) for key in keys}


def _finite_tensor(value, shape, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} has an invalid shape')
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'nonfinite {name}')


def _score_and_cache(model, normalizer, scenes, device, dtype, batch_tracks,
                     pair_batch_size, use_utility_identity, fixed, check, progress=None):
    # Encoding and physical decoding must use the same frozen training moments.
    for key, buffer in (('mean', model.state_mean), ('std', model.state_std)):
        expected = torch.as_tensor(normalizer[key], dtype=torch.float32).to(dtype=buffer.dtype)
        if not torch.equal(buffer.detach().cpu(), expected):
            raise ValueError('evaluation normalizer differs from model physical-state moments')
    if _is_v2(model):
        from .cooperative_inputs import normalize_cooperative_scene, validate_cooperative_representation
        representation = validate_cooperative_representation(normalizer, model.config)
        for key in ('log_speed_mean', 'log_speed_std', 'speed_bias'):
            buffer = getattr(model, key)
            expected = torch.as_tensor(representation[key], dtype=torch.float32).to(dtype=buffer.dtype)
            if not torch.equal(buffer.detach().cpu(), expected):
                raise ValueError('evaluation representation differs from frozen model buffers')
    encoded_keys = _ENCODED + (('native_positions',) if _is_v2(model) else ())
    records, cached = [], []
    stage = 'encoding_fixed_identity' if fixed is not None else 'association_scoring'
    def report(completed, scene_id=None):
        if progress is not None:
            progress(dict(stage=stage, completed_scenes=completed,
                          total_scenes=len(scenes), scene_id=scene_id))
    report(0)
    for scene_index, scene in enumerate(scenes):
        report(scene_index, str(scene['scene_id']))
        if check:
            check()
        values, observed = (normalize_cooperative_scene(scene, normalizer, model.config) if _is_v2(model)
                            else normalize_scene(scene, normalizer))
        chunks = {key: [] for key in encoded_keys}
        for offset in range(0, len(values), batch_tracks):
            if check:
                check()
            stop = offset + batch_tracks
            encoded = model.encode(torch.as_tensor(values[offset:stop], device=device, dtype=dtype),
                                   torch.as_tensor(observed[offset:stop], device=device))
            n = len(values[offset:stop])
            m, q, d = model.config.max_observations, model.config.length, model.config.d_model
            if not isinstance(encoded, dict) or any(key not in encoded for key in encoded_keys):
                raise ValueError('encode must return the complete native encoding dictionary')
            for key, shape in (('memory', (n, m, d)), ('queries', (n, q)), ('pooled', (n, d))):
                _finite_tensor(encoded[key], shape, key)
            if (encoded['observed'].shape != (n, m) or encoded['observed'].dtype != torch.bool
                    or encoded['times'].shape != (n, m)):
                raise ValueError('encoded native mask/times have invalid shapes')
            if not torch.isfinite(encoded['times'][encoded['observed']]).all():
                raise FloatingPointError('nonfinite observed native times')
            if _is_v2(model):
                _finite_tensor(encoded['native_positions'], (n, m, 2), 'native_positions')
            for key in encoded_keys:
                chunks[key].append(encoded[key].detach().cpu())
        # Empty input scenes retain targets and metric denominators, with no model call.
        cached_scene = {key: torch.cat(rows, dim=0) for key, rows in chunks.items()} if len(values) else None
        if cached_scene is not None:
            queries = cached_scene['queries']
            if not torch.equal(queries, queries[:1].expand_as(queries)):
                raise ValueError('all native tracks in a scene must share physical query times')
        cached.append(cached_scene)
        if fixed is not None:
            records.append(fixed[scene_index])
            report(scene_index + 1, str(scene['scene_id']))
            continue
        pairs = np.asarray(scene['pairs'])
        scores = []
        for offset in range(0, len(pairs), pair_batch_size):
            if check:
                check()
            batch = pairs[offset:offset+pair_batch_size]
            nodes, inverse = np.unique(batch, return_inverse=True)
            logits = model.score_pairs(_take(cached_scene, nodes, device), inverse.reshape(-1, 2),
                                       use_utility=use_utility_identity)
            _finite_tensor(logits, (len(batch),), 'association scores')
            scores.append(logits.detach().cpu().numpy())
        records.append(_record(scene, np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)))
        report(scene_index + 1, str(scene['scene_id']))
    return records, cached


@stable_evaluation_parameters
@torch.inference_mode()
def score_cooperative(model, normalizer, scenes, device, batch_tracks=64,
                      pair_batch_size=1024, check=None, *, use_utility_identity=False,
                      progress=None):
    """Score all canonical V_select candidates with CPU-cached native encodings.

    Same records/threshold convention as ``score_reference``. Model remains in
    eval mode; parameter freeze flags are restored to their entry values.
    """
    scenes, device, dtype = _validate(model, scenes, device, batch_tracks, pair_batch_size,
                                      use_utility_identity)
    model.eval()
    records, _ = _score_and_cache(model, normalizer, scenes, device, dtype, batch_tracks,
                                  pair_batch_size, use_utility_identity, None, check, progress)
    return records


def _anchor_logits(record, members, node_count, device, dtype):
    # Canonical np.triu_indices order permits direct indexing without an N x N table.
    anchor = members[0]
    helpers = np.asarray(members[1:], dtype=np.int64)
    indices = anchor * (2 * node_count - anchor - 1) // 2 + helpers - anchor - 1
    return torch.as_tensor(record['scores'][indices], device=device, dtype=dtype)


def _anchor_utility(model, cached, members, device, dtype, pair_batch_size, check):
    values = []
    for offset in range(1, len(members), pair_batch_size):
        if check:
            check()
        helpers = members[offset:offset+pair_batch_size]
        pairs = np.column_stack((np.full(len(helpers), members[0]), helpers))
        nodes, inverse = np.unique(pairs, return_inverse=True)
        utility = model.utility(_take(cached, nodes, device), inverse.reshape(-1, 2))
        _finite_tensor(utility, (len(helpers), 2, model.config.length, 4), 'directional utility')
        values.append(utility[:, 0].detach().cpu())
    return (torch.cat(values).to(device) if values else
            torch.empty((0, model.config.length, 4), device=device, dtype=dtype))


@stable_evaluation_parameters
@torch.inference_mode()
def evaluate_cooperative(model, normalizer, scenes, *, device, batch_tracks,
                         pair_batch_size, select_on_validation, threshold, routing='plain',
                         use_utility_identity=False, fixed_records=None, check=None, progress=None,
                         prediction_writer=None):
    """Return association and predicted/oracle/simple/single-source fusion metrics.

    ``fixed_records`` must describe exactly these scenes, nodes and canonical
    candidates, in the same order. Their scores and the caller's explicit
    threshold determine groups unchanged; threshold=None means all reject.
    Fixed records forbid threshold refitting. Reference identity is used only
    for the separately labelled oracle diagnostics and final correspondence.

    Every predicted/oracle group, including a singleton, is decoded to physical
    states. The visible/missing partition is the union of ORIGINAL native time
    support across all fragments of each true identity, shared by all methods;
    generated availability never changes that partition. All declared targets,
    including identities without input nodes, remain in metric denominators.
    Optional progress receives small stage/scene records only; callback failures
    propagate, and it never receives reference arrays or model tensors.
    An optional prediction_writer receives full-population physical artifacts
    after scoring; it never supplies training inputs or changes the metrics.
    """
    if routing not in (_V2_ROUTES if _is_v2(model) else _ROUTES):
        raise ValueError('unknown cooperative routing policy')
    if not isinstance(select_on_validation, bool):
        raise ValueError('select_on_validation must be boolean')
    if select_on_validation and threshold is not None:
        raise ValueError('provide threshold=None when fitting on V_select')
    if fixed_records is not None and select_on_validation:
        raise ValueError('fixed identity records prohibit threshold refitting')
    if threshold is not None and (isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, (int, float, np.integer, np.floating))
            or not math.isfinite(float(threshold))):
        raise ValueError('threshold must be finite or None (all reject)')
    scenes, device, dtype = _validate(model, scenes, device, batch_tracks, pair_batch_size,
                                      use_utility_identity)
    fixed = _fixed_records(scenes, fixed_records) if fixed_records is not None else None
    started = time.monotonic()
    def notify(item):
        if progress is not None:
            progress(dict(item, elapsed_seconds=time.monotonic()-started))
    model.eval()
    records, cached = _score_and_cache(model, normalizer, scenes, device, dtype, batch_tracks,
                                       pair_batch_size, use_utility_identity, fixed, check, notify)
    notify(dict(stage='association_metrics', completed_scenes=len(scenes),
                total_scenes=len(scenes), scene_id=None))
    selection = select_threshold(records) if select_on_validation else None
    if selection is not None:
        threshold = selection['threshold']
    association = evaluate_scenes(records, threshold)
    sources = sorted({str(source) for scene in scenes for source in scene['sources']})
    methods = {key: [] for key in ['predicted', 'oracle', 'simple_fusion_oracle']
               + ['single_source:' + source for source in sources]}
    per_scene, decode_count = [], 0
    notify(dict(stage='physical_fusion', completed_scenes=0, total_scenes=len(scenes),
                scene_id=None, fusion_forward_count=0))
    for scene_index, (scene, record, encoded) in enumerate(zip(scenes, records, cached)):
        notify(dict(stage='physical_fusion', completed_scenes=scene_index, total_scenes=len(scenes),
                    scene_id=str(scene['scene_id']), fusion_forward_count=decode_count))
        node_count = len(scene['node_ids'])
        groups = {
            'predicted': group_predicted_tracks(node_count, record['pairs'], record['scores'],
                threshold=threshold, grouping_policy='anchor_direct', node_order='input'),
            'oracle': group_tracks_by_identity(scene['identities'], node_order='input'),
        }
        predictions = {}
        for kind, partition in groups.items():
            predictions[kind] = []
            for members in partition:
                if check:
                    check()
                logits = (_anchor_logits(record, members, node_count, device, dtype)
                          if routing in ('identity', 'utility', 'scalar-u', 'attribute-u', 'learned-attribute') else None)
                utility = (_anchor_utility(model, encoded, members, device, dtype, pair_batch_size, check)
                           if routing in ('utility', 'scalar-u', 'attribute-u') else None)
                states = model.decode_group(_take(encoded, members, device), list(range(len(members))),
                                            routing=routing, identity_logits=logits, utility=utility)
                _finite_tensor(states, (model.config.length, 4), 'decoded physical states')
                predictions[kind].append(dict(node_indices=members, states=states.detach().cpu().numpy()))
                decode_count += 1
        # Reference values become visible only after predicted/oracle model calls.
        targets = scene['fusion_targets']
        target_ids = list(targets)
        truth = (np.stack([targets[identity]['states'] for identity in target_ids])
                 if target_ids else np.empty((0, model.config.length, 4)))
        valid = (np.stack([targets[identity]['valid'] for identity in target_ids])
                 if target_ids else np.empty((0, model.config.length, 4), dtype=bool))
        common = dict(node_identities=scene['identities'], target_identities=target_ids,
                      target_states=truth, target_valid=valid, node_observed=scene['observed'],
                      split=scene['split'], correspondence_policy='anchor_identity_first')
        if prediction_writer is not None:
            common['return_artifact'] = True
        outputs = {kind: evaluate_full_scene(rows, grouping_kind=kind, **common)
                   for kind, rows in predictions.items()}
        node_valid = np.repeat(np.asarray(scene['observed'])[..., None], 4, axis=-1)
        simple = simple_fusion_outputs(scene['values'], node_valid, scene['identities'], scene['sources'])
        outputs['simple_fusion_oracle'] = evaluate_full_scene(simple, grouping_kind='oracle', **common)
        for source in sources:
            single = single_source_outputs(scene['values'], node_valid, scene['identities'],
                                           scene['sources'], source=source)
            outputs['single_source:' + source] = evaluate_full_scene(single, grouping_kind='oracle', **common)
        if prediction_writer is not None:
            artifacts = {key: report.pop('artifact') for key, report in outputs.items()}
            notify(dict(stage='writing_predictions', completed_scenes=scene_index, total_scenes=len(scenes),
                        scene_id=str(scene['scene_id']), fusion_forward_count=decode_count))
            if check:
                check()
            prediction_writer(scene_index, scene, artifacts)
            if check:
                check()
        for key, report in outputs.items():
            methods[key].append(report)
        per_scene.append(dict(scene_id=str(scene['scene_id']), methods=outputs,
                              predicted_groups=groups['predicted'], oracle_groups=groups['oracle']))
        notify(dict(stage='physical_fusion', completed_scenes=scene_index+1, total_scenes=len(scenes),
                    scene_id=str(scene['scene_id']), fusion_forward_count=decode_count))
    if check:
        check()
    result = dict(association=association, selection=selection, threshold=threshold, records=records,
                fusion={key: aggregate_fusion_reports(reports) for key, reports in methods.items()},
                per_scene=per_scene, routing=routing, singleton_policy='model_decode',
                grouping_policy='anchor_direct', correspondence_policy='anchor_identity_first',
                identity_score_source='fixed_records' if fixed is not None else 'cooperative_model',
                use_utility_identity=use_utility_identity if fixed is None else False,
                region_definition='union of original native time support over all fragments of each true identity',
                elapsed_seconds=time.monotonic()-started, fusion_forward_count=decode_count,
                test_opened=False, scope='V_select development; oracle, simple-fusion and single-source are diagnostics')
    if _is_v2(model):
        result.update(model_variant=model.config.model_variant,
                      representation_version=normalizer['cooperative_representation']['version'])
    notify(dict(stage='completed', completed_scenes=len(scenes), total_scenes=len(scenes),
                scene_id=None, fusion_forward_count=decode_count))
    return result
