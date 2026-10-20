"""Fixed-motion fusion training and all-fragment validation, without test access."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from association.data.motion_evidence import build_motion_evidence
from association.evaluation.fusion_controls import ordinary_fusion, metrics, rms_haversine_scores
from association.evaluation.grouping import group_predicted_tracks, group_tracks_by_identity
from association.models.evidence_fusion import EvidenceFusion, evidence_fusion_loss, fusion_inputs
from association.utils.runtime import write_json

FIXED_DISTANCE_THRESHOLD_M = 14214.358367922307
GROUPINGS = ('oracle_identity', 'predicted_anchor_direct')
TRAIN_MODES = ('clean', 'clean_plus_wrong', 'wrong_only')
ARMS = ('anchor', 'nearest_observation', 'mean_supported', 'learned')


def prepare_evidence_scenes(scenes, check=None):
    """Separate target-free motion construction from labels and training targets."""
    result = []
    for scene in scenes:
        if check:
            check()
        if scene['split'] not in ('T_core', 'V_select'):
            raise ValueError('only T_core and V_select are permitted; T_test remains sealed')
        motion = build_motion_evidence(scene['native_values'], scene['native_times'],
            scene['native_observed'], scene['grid'], scene['sources'])
        n = len(motion['states'])
        pairs = np.column_stack(np.triu_indices(n, 1))
        scores = np.empty(len(pairs), np.float64)
        # Keep every pair while bounding [pairs,Q,2] temporary geometry arrays.
        for start in range(0, len(pairs), 4096):
            if check:
                check()
            scores[start:start+4096] = rms_haversine_scores(
                motion['states'][..., :2], pairs[start:start+4096])
        distances = np.full((n, n), np.inf)
        distances[pairs[:, 0], pairs[:, 1]] = -scores
        distances[pairs[:, 1], pairs[:, 0]] = -scores
        predicted = group_predicted_tracks(n, pairs, scores, threshold=-FIXED_DISTANCE_THRESHOLD_M,
            grouping_policy='anchor_direct', node_order='input')
        identities = np.asarray(scene['identities'], str)
        targets, masks = [], []
        for identity in identities:
            target = scene['fusion_targets'][str(identity)]
            targets.append(target['states'])
            masks.append(target['valid'])
        q = motion['states'].shape[1]
        result.append(dict(scene_id=scene['scene_id'], component_id=scene['component_id'],
            split=scene['split'], node_ids=np.asarray(scene['node_ids'], str), identities=identities,
            sources=np.asarray(scene['sources'], str), motion=motion, distances=distances,
            target=np.asarray(targets, np.float64).reshape(n, q, 4),
            valid=np.asarray(masks, bool).reshape(n, q, 4),
            groups=dict(oracle_identity=group_tracks_by_identity(identities, node_order='input'),
                        predicted_anchor_direct=predicted)))
    return result


def training_members(scene, anchor, mode):
    """Identity labels construct training corruption, but are never model features."""
    if scene['split'] != 'T_core' or mode not in TRAIN_MODES:
        raise ValueError('training sample construction is restricted to T_core')
    identities = scene['identities']
    clean = [i for i in range(len(identities)) if i != anchor and identities[i] == identities[anchor]]
    wrong = np.flatnonzero(identities != identities[anchor])
    # The hard negative uses only the already frozen OLS distance; ties input-order.
    wrong = ([] if not len(wrong) else [int(wrong[np.argmin(scene['distances'][anchor, wrong])])])
    return [anchor]+(clean if mode != 'wrong_only' else [])+(wrong if mode != 'clean' else [])


def _scene_regions(scene):
    motion = scene['motion']
    return dict(all=np.ones_like(motion['support']), inside_support=motion['support'],
        outside_support=~motion['support'], exact_observation=motion['exact_observation'],
        inside_nonobserved=motion['support'] & ~motion['exact_observation'],
        far_gt_1800s=motion['nearest_seconds'] > 1800.)


@torch.no_grad()
def evaluate_evidence_fusion(model, scenes, *, device, prediction_dir=None, check=None):
    """All fragments remain anchors; grouping changes no target mask or population."""
    was_training = model.training
    model.eval()
    rows, per_scene, per_anchor = [], [], []
    destination = Path(prediction_dir) if prediction_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
    try:
        for scene in scenes:
            if check:
                check()
            if scene['split'] != 'V_select':
                raise ValueError('monitoring requires V_select scenes')
            motion, target, valid = scene['motion'], scene['target'], scene['valid']
            n, q = target.shape[:2]
            regions = _scene_regions(scene)
            outputs, mixed_groups = {}, {}
            payload = dict(node_ids=scene['node_ids'], identities=scene['identities'], sources=scene['sources'],
                query_seconds=motion['query_seconds'], target=target, target_valid=valid,
                native_states=motion['states'], nearest_seconds=motion['nearest_seconds'],
                **{'region_'+key: value for key, value in regions.items()})
            local = dict(scene_id=scene['scene_id'], component_id=scene['component_id'],
                         fragments=n, query_count=n*q, methods={}, routing={})
            for grouping in GROUPINGS:
                groups = scene['groups'][grouping]
                controls = ordinary_fusion(motion['states'], motion['nearest_seconds'], motion['support'], groups)
                predictions = {key: controls['predictions'][key] for key in ARMS[:-1]}
                predictions['learned'] = motion['states'].copy()
                mixed = np.zeros(n, bool)
                own_weights = np.ones((n, q, 3))
                all_rejected = np.ones((n, q, 3), bool)
                zero_helpers = np.zeros((n, q, 3), np.int64)
                helper_counts = np.zeros(n, np.int64)
                for group in groups:
                    mixed[group] = len(set(scene['identities'][group])) > 1
                    for anchor in group:
                        members = [anchor]+[i for i in group if i != anchor]
                        result = model(*fusion_inputs(motion, members, device=device))
                        predictions['learned'][anchor] = result['states'].cpu().numpy()
                        weights = result['weights'].cpu().numpy()
                        own_weights[anchor] = weights[0]
                        zero_helpers[anchor] = (weights[1:] == 0).sum(0)
                        all_rejected[anchor] = (weights[1:] == 0).all(0)
                        helper_counts[anchor] = len(members)-1
                eligible = helper_counts > 0
                denominator = int(helper_counts.sum())*q
                local['routing'][grouping] = dict(
                    helper_query_count=denominator,
                    rejected_helper_queries_by_attribute=zero_helpers.sum((0, 1)).tolist(),
                    all_helpers_rejected_by_attribute=all_rejected[eligible].sum((0, 1)).tolist(),
                    eligible_anchor_query_count=int(eligible.sum())*q,
                    singleton_anchors=int((~eligible).sum()),
                    mixed_group_anchors=int(mixed.sum()), groups=groups)
                local['methods'][grouping] = {arm: metrics(value, target, valid, regions['all'])
                                              for arm, value in predictions.items()}
                outputs[grouping], mixed_groups[grouping] = predictions, mixed
                payload[grouping+'_membership'] = controls['membership']
                payload[grouping+'_anchor_weights'] = own_weights
                payload[grouping+'_zero_helper_counts'] = zero_helpers
                payload[grouping+'_helper_counts'] = helper_counts
                for arm, prediction in predictions.items():
                    payload[grouping+'_'+arm] = prediction
            if destination is not None:
                np.savez_compressed(destination/('states-'+scene['scene_id']+'.npz'), **payload)
                for anchor in range(n):
                    selection = slice(anchor, anchor+1)
                    per_anchor.append(dict(scene_id=scene['scene_id'], node_id=scene['node_ids'][anchor],
                        identity=scene['identities'][anchor], source=scene['sources'][anchor],
                        methods={grouping: {arm: metrics(prediction[selection], target[selection],
                            valid[selection], regions['all'][selection]) for arm, prediction in methods.items()}
                            for grouping, methods in outputs.items()}))
            rows.append(dict(scene=scene, regions=regions, predictions=outputs, mixed=mixed_groups))
            per_scene.append(local)
        if not rows:
            raise ValueError('complete validation population is required')
        target = np.concatenate([row['scene']['target'] for row in rows])
        valid = np.concatenate([row['scene']['valid'] for row in rows])
        sources = np.concatenate([row['scene']['sources'] for row in rows])
        regions = {name: np.concatenate([row['regions'][name] for row in rows]) for name in rows[0]['regions']}
        overall, routing = {}, {}
        for grouping in GROUPINGS:
            mixed = np.broadcast_to(np.concatenate([row['mixed'][grouping] for row in rows])[:, None], valid.shape[:2])
            region_masks = dict(regions, mixed_group=mixed, pure_group=~mixed)
            overall[grouping] = {}
            for arm in ARMS:
                prediction = np.concatenate([row['predictions'][grouping][arm] for row in rows])
                overall[grouping][arm] = dict(
                    regions={name: metrics(prediction, target, valid, region) for name, region in region_masks.items()},
                    by_source={str(source): {name: metrics(prediction[sources == source],
                        target[sources == source], valid[sources == source], region[sources == source])
                        for name, region in regions.items()} for source in sorted(set(sources))})
            entries = [scene['routing'][grouping] for scene in per_scene]
            denominator = sum(entry['helper_query_count'] for entry in entries)
            rejected = np.sum([entry['rejected_helper_queries_by_attribute'] for entry in entries], axis=0)
            eligible = sum(entry['eligible_anchor_query_count'] for entry in entries)
            all_rejected = np.sum([entry['all_helpers_rejected_by_attribute'] for entry in entries], axis=0)
            routing[grouping] = dict(helper_query_count=denominator,
                rejection_fraction_by_attribute=(rejected/denominator).tolist() if denominator else [None]*3,
                eligible_anchor_query_count=eligible,
                all_helpers_rejected_fraction_by_attribute=(all_rejected/eligible).tolist() if eligible else [None]*3)
        return dict(scene_ids=[scene['scene_id'] for scene in scenes], scene_count=len(scenes),
            component_count=len({scene['component_id'] for scene in scenes}), fragment_count=len(target),
            query_count=len(target)*target.shape[1], target_valid_by_attribute=valid.sum((0, 1)).tolist(),
            fixed_distance_threshold_m=FIXED_DISTANCE_THRESHOLD_M, grouping_policy='anchor_direct/input',
            coverage='every original fragment anchor; not unique identity output',
            overall=overall, routing=routing, per_scene=per_scene, per_anchor=per_anchor,
            limitations=['V is repeatedly exposed development data; no independent confirmation',
                'oracle identity grouping is diagnostic only', '9001/reference reuse limits denoising claims',
                'point-pooled fragment weighting repeats identities with multiple fragments'],
            test_opened=False)
    finally:
        model.train(was_training)


def save_checkpoint(path, model, optimizer, step, config, rng):
    torch.save(dict(scope='native-evidence-fusion-v1', configuration=config, step=step,
                    model=model.state_dict(), optimizer=optimizer.state_dict(),
                    numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None), path)


def load_checkpoint(path, *, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['scope'] != 'native-evidence-fusion-v1':
        raise ValueError('expected an evidence-fusion checkpoint')
    model = EvidenceFusion(**state['configuration']['model']).to(device)
    model.load_state_dict(state['model'], strict=True)
    return model, state


def train_evidence_fusion(train, validation, config, output_dir, *, device='cpu',
                          check=None, initial_elapsed_seconds=0.):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    training = config['training']
    torch.manual_seed(training['seed'])
    rng = np.random.default_rng(training['seed'])
    model = EvidenceFusion(**config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training['learning_rate'])
    state = dict(status='RUNNING', optimizer_updates=0, test_opened=False, device=str(device),
        configuration=config, training_scene_ids=[scene['scene_id'] for scene in train],
        validation_scene_ids=[scene['scene_id'] for scene in validation],
        selection_policy='fixed final step; intermediate full V is monitoring only',
        sampling='uniform scene then uniform anchor, independently for each batch member; repeats possible',
        modes=list(TRAIN_MODES), fixed_distance_threshold_m=FIXED_DISTANCE_THRESHOLD_M,
        training_samples={mode: dict(anchors=0, with_helper=0, with_wrong_helper=0,
                                    no_wrong_fallback=0) for mode in TRAIN_MODES},
        physical_dtype='float64 native motion and geometry; float32 MLP', advisory_reached=False)
    write_json(output/'run.json', state)
    try:
        save_checkpoint(output/'step-0.pt', model, optimizer, 0, config, rng)
        train = prepare_evidence_scenes(train, check=check)
        validation = prepare_evidence_scenes(validation, check=check)
        if (not train or not validation or any(scene['split'] != 'T_core' for scene in train)
                or any(scene['split'] != 'V_select' for scene in validation)
                or {scene['component_id'] for scene in train} & {scene['component_id'] for scene in validation}):
            raise ValueError('nonempty, component-isolated T_core/V_select are required')
        if any(not len(scene['identities']) for scene in train):
            raise ValueError('uniform-scene training has an empty scene; do not silently filter population')
        report = evaluate_evidence_fusion(model, validation, device=device,
                                        prediction_dir=output/'predictions-step-0', check=check)
        write_json(output/'evaluation-step-0.json', report)
        update_seconds = 0.
        for step in range(1, training['updates']+1):
            if check:
                check()
            model.train()
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            logs = dict(total=0., utility=0., direct=0.)
            mode = TRAIN_MODES[(step-1) % len(TRAIN_MODES)]
            for _ in range(training['batch_size']):
                scene = train[int(rng.integers(len(train)))]
                anchor = int(rng.integers(len(scene['identities'])))
                members = training_members(scene, anchor, mode)
                counters = state['training_samples'][mode]
                counters['anchors'] += 1
                counters['with_helper'] += int(len(members) > 1)
                has_wrong = bool(np.any(scene['identities'][members] != scene['identities'][anchor]))
                counters['with_wrong_helper'] += int(has_wrong)
                counters['no_wrong_fallback'] += int(mode != 'clean' and not has_wrong)
                base, features, exact = fusion_inputs(scene['motion'], members, device=device)
                result = model(base, features, exact)
                loss = evidence_fusion_loss(result, base,
                    torch.as_tensor(scene['target'][anchor], device=device),
                    torch.as_tensor(scene['valid'][anchor], device=device))
                if not torch.isfinite(loss['total']):
                    raise FloatingPointError('nonfinite evidence fusion loss')
                (loss['total']/training['batch_size']).backward()
                for key in logs:
                    logs[key] += float(loss[key].detach())/training['batch_size']
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'],
                                                    error_if_nonfinite=True)
            optimizer.step()
            update_seconds += time.monotonic()-step_started
            elapsed = initial_elapsed_seconds+time.monotonic()-started
            if elapsed >= training['advisory_seconds'] and not state['advisory_reached']:
                print('ADVISORY: suggested duration exceeded; continuing the same fixed updates, no retry.',
                      flush=True)
            state.update(optimizer_updates=step, elapsed_seconds=elapsed,
                         advisory_reached=elapsed >= training['advisory_seconds'],
                         mean_update_seconds=update_seconds/step,
                         remaining_update_seconds_estimate=(training['updates']-step)*update_seconds/step)
            with (output/'training.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(step=step, mode=mode, gradient_norm=float(gradient),
                                             elapsed_seconds=elapsed, **logs), allow_nan=False)+'\n')
            write_json(output/'run.json', state)
            if step == 50:
                print(f'fusion step 50: {update_seconds/step:.3f} seconds/update; '
                      f'fixed remaining updates ~{state["remaining_update_seconds_estimate"]:.1f}s, '
                      'excluding future full-V evaluations; advisory only.', flush=True)
            if step % training['evaluate_every'] == 0 or step == training['updates']:
                final = step == training['updates']
                report = evaluate_evidence_fusion(model, validation, device=device,
                    prediction_dir=output/'predictions-final' if final else None, check=check)
                write_json(output/f'evaluation-step-{step}.json', report)
                print(f'fusion step {step}: total={logs["total"]:.6g}; full V complete', flush=True)
        save_checkpoint(output/'final.pt', model, optimizer, training['updates'], config, rng)
        state.update(status='COMPLETED', elapsed_seconds=initial_elapsed_seconds+time.monotonic()-started,
                     final_evaluation=f'evaluation-step-{training["updates"]}.json')
        write_json(output/'run.json', state)
        return state
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                     error=f'{type(error).__name__}: {error}',
                     elapsed_seconds=initial_elapsed_seconds+time.monotonic()-started)
        write_json(output/'run.json', state)
        try:
            save_checkpoint(output/'interrupted.pt', model, optimizer, state['optimizer_updates'], config, rng)
        except BaseException as save_error:
            state['recovery_checkpoint_error'] = f'{type(save_error).__name__}: {save_error}'
            write_json(output/'run.json', state)
        raise
