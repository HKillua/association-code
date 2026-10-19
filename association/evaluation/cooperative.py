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

from association.evaluation.tmtaff import _inference_device, _registered, _aggregate_regions, aggregate_fusion_reports, scene_input_support
from association.evaluation.scene import evaluate_full_scene, simple_fusion_outputs, single_source_outputs, INPUT_DISTANCE_REGIONS
from association.evaluation.physical import evaluate_aligned_fusion
from association.evaluation.grouping import group_predicted_tracks, group_tracks_by_identity
from association.evaluation.association import evaluate_scenes, select_threshold
from association.training.common import normalize_scene, stable_evaluation_parameters


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
    keys = _ENCODED + tuple(key for key in ('native_positions', 'native_headings') if key in encoded)
    return {key: encoded[key].index_select(0, indices).to(device) for key in keys}


def _finite_tensor(value, shape, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} has an invalid shape')
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'nonfinite {name}')


def _validate_normalizer(model, normalizer):
    # Encoding and physical decoding must use the same frozen training moments.
    for key, buffer in (('mean', model.state_mean), ('std', model.state_std)):
        expected = torch.as_tensor(normalizer[key], dtype=torch.float32).to(dtype=buffer.dtype)
        if not torch.equal(buffer.detach().cpu(), expected):
            raise ValueError('evaluation normalizer differs from model physical-state moments')
    if _is_v2(model):
        from association.data.cooperative import validate_cooperative_representation
        representation = validate_cooperative_representation(normalizer, model.config)
        for key in ('log_speed_mean', 'log_speed_std', 'speed_bias'):
            buffer = getattr(model, key)
            expected = torch.as_tensor(representation[key], dtype=torch.float32).to(dtype=buffer.dtype)
            if not torch.equal(buffer.detach().cpu(), expected):
                raise ValueError('evaluation representation differs from frozen model buffers')


def _score_and_cache(model, normalizer, scenes, device, dtype, batch_tracks,
                     pair_batch_size, use_utility_identity, fixed, check, progress=None):
    _validate_normalizer(model, normalizer)
    if _is_v2(model):
        from association.data.cooperative import normalize_cooperative_scene
    encoded_keys = _ENCODED + (('native_positions',) if _is_v2(model) else ())
    if model.config.heading_output_policy == 'observation_residual':
        encoded_keys += ('native_headings',)
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
            if 'native_headings' in encoded_keys:
                _finite_tensor(encoded['native_headings'], (n, m), 'native_headings')
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
        support = scene_input_support(scene)
        outputs = {kind: evaluate_full_scene(rows, grouping_kind=kind, node_input_support=support, **common)
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


@stable_evaluation_parameters
@torch.inference_mode()
def evaluate_cooperative_completion(model, normalizer, scenes, *, device='cpu', check=None,
                                    progress=None, prediction_writer=None):
    """Evaluate every native fragment independently, without association or helpers.

    Only the fragment's visible inputs and query times enter encode/decode. Its
    identity selects reference rows afterwards; repeated identities retain every
    fragment. Position-base controls have no speed or heading prediction. Metrics
    pool point numerators; per-fragment rows and scene reports retain coverage.
    This is V_select development evaluation of an existing v2 teacher, not a
    pure-completion training claim or a new artificial endpoint-hiding protocol.
    An optional writer receives ``(scene, arrays)`` after metrics are computed;
    it observes the existing predictions and never causes another model call.
    ``position_base`` has four columns, with unavailable speed/course as NaN.
    """
    from association.data.cooperative import normalize_cooperative_scene
    if not _is_v2(model):
        raise ValueError('independent completion requires support_residual_v2 and its position base')
    scenes, device, dtype = _validate(model, scenes, device, 1, 1, False)
    _validate_normalizer(model, normalizer)
    model.eval()
    started = time.monotonic()
    regions = ('all', 'inside_support', 'before_support', 'after_support',
               'exact_observation', 'inside_nonobserved')
    per_scene, per_fragment = [], []
    method_reports = {name: [] for name in ('model', 'position_base')}
    total_fragments = sum(len(scene['node_ids']) for scene in scenes)
    decoded = 0
    for scene_index, scene in enumerate(scenes):
        if check:
            check()
        values, observed = normalize_cooperative_scene(scene, normalizer, model.config)
        n, q = len(values), model.config.length
        predictions = np.empty((n, q, 4), dtype=np.float32)
        base = np.full((n, q, 4), np.nan, dtype=np.float32)
        for node in range(n):
            if check:
                check()
            encoded = model.encode(torch.as_tensor(values[node:node+1], device=device, dtype=dtype),
                                   torch.as_tensor(observed[node:node+1], device=device))
            states = model.decode_group(encoded, [0], routing='plain')
            positions = model.position_base(encoded, [0])
            _finite_tensor(states, (q, 4), 'single-fragment physical states')
            _finite_tensor(positions, (1, q, 2), 'single-fragment position base')
            predictions[node] = states.detach().cpu().numpy()
            base[node, :, :2] = positions[0].detach().cpu().numpy()
            decoded += 1
        # Reference identity, values and masks are used only after model calls.
        truth = np.full((n, q, 4), np.nan)
        valid = np.zeros((n, q, 4), dtype=bool)
        targets = scene['fusion_targets']
        for node, identity in enumerate(scene['identities']):
            if identity in targets:
                truth[node] = targets[identity]['states']
                valid[node] = targets[identity]['valid']
        support = scene_input_support(scene)
        inside, nearest = support['support'], support['nearest_seconds']
        before, after = np.zeros((n, q), bool), np.zeros((n, q), bool)
        for node in range(n):
            times = np.asarray(scene['native_times'])[node, scene['native_observed'][node]]
            before[node] = scene['grid'] < times.min()
            after[node] = scene['grid'] > times.max()
        exact = nearest == 0
        masks = dict(zip(regions, (np.ones((n, q), bool), inside, before, after,
                                  exact, inside & ~exact)))
        distance_masks = (nearest <= 300, (nearest > 300) & (nearest <= 600),
                          (nearest > 600) & (nearest <= 1800),
                          (nearest > 1800) & np.isfinite(nearest), ~np.isfinite(nearest))
        model_valid = np.ones((n, q, 4), bool)
        base_valid = np.zeros_like(model_valid)
        base_valid[..., :2] = True
        fragment_start = len(per_fragment)
        for node in range(n):
            per_fragment.append(dict(scene_id=str(scene['scene_id']), node_index=node,
                node_id=str(scene['node_ids'][node]), identity=str(scene['identities'][node]),
                source=str(scene['sources'][node]), query_count=q,
                reference_declared=scene['identities'][node] in targets,
                has_supervision=bool(valid[node].any()),
                region_query_counts={key: int(mask[node].sum()) for key, mask in masks.items()},
                methods={}))
        methods = {}
        for name, prediction, available in (('model', predictions, model_valid),
                                             ('position_base', base, base_valid)):
            def metric(mask, rows=slice(None)):
                return evaluate_aligned_fusion(prediction[rows], truth[rows], valid[rows],
                    prediction_valid=available[rows], region_mask=mask[rows])
            report = dict(regions={key: metric(mask) for key, mask in masks.items()},
                          input_distance_metrics={key: metric(mask) for key, mask in
                                                  zip(INPUT_DISTANCE_REGIONS, distance_masks)})
            methods[name] = report
            method_reports[name].append(report)
            for node in range(n):
                per_fragment[fragment_start+node]['methods'][name] = metric(masks['all'], slice(node, node+1))
        per_scene.append(dict(scene_id=str(scene['scene_id']), fragment_count=n,
            reference_unavailable_fragment_count=int((~valid.any(axis=(1, 2))).sum()),
            reference_targets_without_input=[str(identity) for identity in targets
                                             if identity not in set(scene['identities'])],
            methods=methods))
        if prediction_writer is not None:
            if check:
                check()
            prediction_writer(scene, dict(prediction=predictions, position_base=base,
                target=truth, target_valid=valid, region_masks=masks,
                nearest_seconds=nearest))
            if check:
                check()
        if progress is not None:
            progress(dict(stage='single_fragment_completion', completed_scenes=scene_index+1,
                total_scenes=len(scenes), completed_fragments=decoded, total_fragments=total_fragments,
                scene_id=str(scene['scene_id']), elapsed_seconds=time.monotonic()-started))
    if check:
        check()
    return dict(scene_count=len(scenes), fragment_count=total_fragments,
        reference_unavailable_fragment_count=sum(row['reference_unavailable_fragment_count'] for row in per_scene),
        methods={name: dict(regions=_aggregate_regions(reports, 'regions', regions),
            input_distance_metrics=_aggregate_regions(reports, 'input_distance_metrics', INPUT_DISTANCE_REGIONS))
            for name, reports in method_reports.items()},
        per_scene=per_scene, per_fragment=per_fragment, completion_forward_count=decoded,
        position_control='visible native position interpolation within support; endpoint hold outside; no speed/heading control',
        region_definition='per-fragment retained native time span; exact_observation requires identical physical query time; inside_nonobserved includes interpolation and internal gaps',
        error_scope='point-pooled metrics over all fragments, retaining repeated identities and no-reference rows; not target-weighted',
        scope='V_select single-fragment development; no artificial hiding, helpers or identity scores',
        elapsed_seconds=time.monotonic()-started, test_opened=False)


@torch.inference_mode()
def completion_objective_report(prediction, target, valid, objective, region_masks):
    """Measure frozen self predictions with the original FP32 FusionLoss.

    Attribute means use each region's actual valid counts. Missing attributes
    are reported as None, not zero. Counts of one for empty columns only satisfy
    FusionLoss's microbatch denominator contract; their zero internal sums are
    never reported as measured losses. The learned regularizer is listed once,
    separately from data terms, and no objective state or gradient is changed.
    This is a self-only development measurement, not the sampled training loss
    or evidence of self/cross/identity gradient conflict.
    """
    from association.models.fusion import FusionLoss

    if not isinstance(objective, FusionLoss):
        raise ValueError('completion objective report requires FusionLoss')
    if objective.log_variance.dtype != torch.float32:
        raise ValueError('completion objective report requires FP32 objective weights')
    device = objective.log_variance.device
    prediction = torch.as_tensor(prediction, dtype=torch.float32, device=device)
    target = torch.as_tensor(target, dtype=torch.float32, device=device)
    valid = torch.as_tensor(valid, device=device)
    if (prediction.ndim != 3 or prediction.shape[-1] != 4
            or target.shape != prediction.shape or valid.shape != prediction.shape
            or valid.dtype != torch.bool):
        raise ValueError('completion objective inputs require matching [N,Q,4] arrays and boolean valid mask')
    log_variance = objective.log_variance.detach()
    weights = .5 * torch.exp(-log_variance)
    regularizer = torch.nn.functional.softplus(log_variance)
    if not all(torch.isfinite(value).all() for value in (log_variance, weights, regularizer)):
        raise ValueError('completion objective weights must be finite')
    reports = {}
    for name, region in region_masks.items():
        region = torch.as_tensor(region, device=device)
        if region.dtype != torch.bool or region.shape != prediction.shape[:2]:
            raise ValueError('completion objective region masks must be boolean [N,Q] arrays')
        selected = valid & region[..., None]
        counts = selected.sum((0, 1))
        stats = objective(prediction, target, selected,
            normalization_counts=counts.clamp_min(1), regularizer_weight=0.)
        attributes = {}
        for column, attribute in enumerate(objective.feature_names):
            count = int(counts[column].item())
            raw = stats['losses'][attribute]
            attributes[attribute] = dict(valid_count=count,
                raw_data_loss=float(raw.item()) if count else None,
                weighted_contribution=float((weights[column]*raw).item()) if count else None)
        reports[str(name)] = dict(attributes=attributes)
    return dict(feature_order=list(objective.feature_names), dtype='float32',
        position_loss_mode=objective.position_loss_mode,
        longitude_loss_mode=objective.longitude_loss_mode,
        log_variance=log_variance.cpu().tolist(), data_weights=weights.cpu().tolist(),
        regularizer_by_attribute=regularizer.cpu().tolist(),
        regularizer_total=float(regularizer.sum().item()), regions=reports,
        scope='frozen self predictions; per-region attribute means; regularizer excluded from data contributions')
