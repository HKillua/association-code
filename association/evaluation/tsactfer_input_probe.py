"""079 bounded T-only frozen-forward qualification, not a new TSA benchmark.

Same positive views enter every condition. Negatives are another identity's right
view, as in contrastive training; its crop is made with its own positive partner.
The two scene domains use paired anchors and equal 1:1 prevalence. Eligibility is
reported separately, never used to quietly remove difficult negative candidates.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F

from association.data.archive import MTADArchive
from association.data.legacy_dataset import validate_archive
from association.data.native import _record
from association.evaluation.tsactfer_score_diagnostics import _ranking, _quantiles
from association.utils.runtime import read_json
from baselines.tsactfer_data import build_positive_pools, sample_batch
from baselines.tsactfer_level1 import configuration, compact_native, pack_compact, final_model

SCOPE = '079-tsa-frozen-training-input-probe-v1'
DEFAULT = dict(scope=SCOPE, seed=20261024, scene_count=32, pairs_per_head_scene=32,
               inference_batch_tracks=128, advisory_seconds=3600)


def validate_config(config):
    if set(config) != set(DEFAULT) or config.get('scope') != SCOPE:
        raise ValueError('unexpected bounded probe configuration')
    for key in ('seed', 'scene_count', 'pairs_per_head_scene', 'inference_batch_tracks', 'advisory_seconds'):
        if type(config[key]) is not int or config[key] < (0 if key == 'seed' else 1):
            raise ValueError('invalid probe '+key)
    if not 2 <= config['scene_count'] <= 32 or not 2 <= config['pairs_per_head_scene'] <= 32:
        raise ValueError('probe fixes a finite cap of32 scenes and32 pairs perheadscene')
    return deepcopy(config)


def select_roster(manifest, config):
    validate_config(config)
    train = list(map(str, manifest['splits']['T_core']))
    if (len(set(train)) != len(train) or len(train) < config['scene_count']
            or set(train) & (set(manifest['splits'].get('V_select', [])) |
                              set(manifest['splits'].get('T_test', [])))):
        raise ValueError('distinct disjoint T_core roster required')
    indices = np.sort(np.random.default_rng(config['seed']).choice(
        len(train), config['scene_count'], replace=False))
    return [train[int(i)] for i in indices]


def bind_source(source, manifest, expected_config):
    """Bind final1000 with complete successful-call audit; never reselect V."""
    from pathlib import Path
    source = Path(source)
    configuration(expected_config)
    run, summary = read_json(source/'run.json'), read_json(source/'summary.json')
    endpoint = read_json(source/'final-endpoint.json')
    steps = expected_config['training']['updates']
    for item in (run, summary):
        if (item.get('status') != 'COMPLETED' or item.get('scope') != expected_config['scope']
                or item.get('configuration') != expected_config or item.get('optimizer_updates') != steps
                or item.get('test_opened') is not False or item.get('endpoint') != 'final'):
            raise ValueError('079 requires completed074D actual final')
    if (run.get('phase') != 'completed' or run.get('checkpoint_selection') is not False
            or run.get('retries') != 0 or run.get('old066_acceptance') != 'REJECTED'
            or endpoint.get('status') != 'COMPLETED' or endpoint.get('configuration') != expected_config
            or endpoint.get('step') != steps or endpoint.get('actual_updates') != steps
            or endpoint.get('checkpoint') != str(source/'fit/final.pt')
            or endpoint.get('scene_ids') != manifest['splits']['V_select']
            or endpoint.get('test_opened') is not False
            or endpoint.get('scores_from_actual_final_checkpoint') is not True):
        raise ValueError('079 frozen source final/population binding differs')
    normalizer = endpoint['normalizer']
    if (normalizer.get('training_scene_ids') != manifest['splits']['T_core']
            or normalizer.get('training_splits') != ['T_core']):
        raise ValueError('079 requires frozen full T_core normalizer')
    model, saved = final_model(source/'fit', expected_config, normalizer)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, normalizer, dict(source=str(source), checkpoint=str(source/'fit/final.pt'),
        endpoint_steps=saved['step'], fit_exposure_verified=True, normalizer_refit=False,
        source_final_bound=True, validation_metadata_read=True, validation_arrays_read=False)


def prepare_inputs(archives, manifest, roster, normalizer, tsa_config, check=lambda: None):
    if (len(set(roster)) != len(roster) or not set(roster) <= set(manifest['splits']['T_core'])
            or normalizer['training_scene_ids'] != manifest['splits']['T_core']):
        raise ValueError('079 input roster must be frozen T_core subset')
    packed, packing = [], []
    with MTADArchive(archives) as archive:
        validate_archive(archive, manifest)
        for sid in roster:
            check()
            native = _record(archive.load_scene(sid), manifest['components'][sid], 'T_core',
                tsa_config['data']['max_points'], split_unit=manifest.get('split_unit'))
            item = pack_compact(compact_native(native, tsa_config['data']['max_points']), normalizer, tsa_config)
            packed.append(item)
            packing.append(item['packing_audit'])
            del native
        reads = list(archive.members_read)
    if (any(row['kind'] not in ('labels', 'measurements') or row['scene_id'] not in roster for row in reads)
            or {row['scene_id'] for row in reads} != set(roster)
            or len(reads) != 2*len(roster)):
        raise ValueError('079 archive read population differs from bounded T_only request')
    return packed, dict(scene_ids=roster, archive_members_read=reads, packing=packing,
                        reference_read=False, validation_read=False, test_opened=False)


def _embeddings(model, arrays, task, batch_size, device, check):
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(arrays['positions']), batch_size):
            check()
            inputs = [torch.as_tensor(arrays[key][start:start+batch_size], device=device)
                      for key in ('positions', 'time_ids', 'observed')]
            chunks.append(F.normalize(model(*inputs, task=task.lower()), dim=-1).cpu().numpy())
    vectors = np.concatenate(chunks)
    if not np.isfinite(vectors).all():
        raise FloatingPointError('nonfinite frozen embeddings')
    return vectors


def _natural_eligible(left, right, task):
    a, b = left['left_interval'], right['right_interval']
    same = left['left_source'] == right['right_source']
    gap = max(a[0], b[0])-min(a[1], b[1])
    return bool(same and 0 <= gap < 1200 and a[1]-a[0] > 120 and b[1]-b[0] > 120) if task == 'ITSA' else bool(not same and -gap > 120)


def build_views(model, scenes, tsa_config, config, *, device='cpu', check=lambda: None):
    """Cap by fixed RNG, one eligible pair per identity per scene; no replacement.

    sample_batch itself makes every crop (same augmentation code as074D). Scenes
    with <2 eligible identities cannot support the paired same-scene comparison;
    they remain in the preparation audit and are not replaced with another scene.
    """
    if (len({str(s['scene_id']) for s in scenes}) != len(scenes)
            or any(s['split'] != 'T_core' for s in scenes)):
        raise ValueError('079 view population must be distinct T_core scenes')
    model.eval()
    model.to(device)
    result, audits = {}, []
    for task_index, task in enumerate(('ITSA', 'MSTSA')):
        selection = np.random.default_rng(np.random.SeedSequence([config['seed'], task_index, 1]))
        crop_rng = np.random.default_rng(np.random.SeedSequence([config['seed'], task_index, 2]))
        rows, vectors = [], {'full': [], 'train_crop': []}
        for scene in scenes:
            check()
            pools, stats = build_positive_pools([scene])
            available = sorted(pools[task])
            count = min(config['pairs_per_head_scene'], len(available))
            audit = dict(scene_id=str(scene['scene_id']), task=task,
                eligible_identities=len(available), eligible_pairs=stats['eligible_pair_counts'][task],
                selected_identities=count if count >= 2 else 0,
                excluded_reason='fewer_than_two_eligible_identities' if count < 2 else None)
            audits.append(audit)
            if count < 2:
                continue
            identities = [available[int(i)] for i in selection.choice(len(available), count, replace=False)]
            forced = {identity: [pools[task][identity][int(selection.integers(len(pools[task][identity])))]]
                      for identity in identities}
            batch = sample_batch([scene], {task: forced}, task, count, crop_rng,
                min_crop_fraction=tsa_config['data']['min_crop_fraction'])
            refs = batch['pairs']
            node_indices = np.r_[refs[:, 1], refs[:, 2]]
            full = {key: np.asarray(scene[key])[node_indices] for key in ('positions', 'time_ids', 'observed')}
            for mode, arrays in (('full', full), ('train_crop', batch)):
                encoded = _embeddings(model, arrays, task, config['inference_batch_tracks'], device, check)
                vectors[mode].append(np.stack((encoded[:count], encoded[count:]), axis=1))
            for index, (_, left, right) in enumerate(refs):
                if scene['identities'][left] != scene['identities'][right]:
                    raise ValueError('079 positive pair identity mismatch')
                rows.append(dict(scene_id=str(scene['scene_id']), component_id=str(scene['component_id']),
                    identity=str(scene['identities'][left]), left_node=str(scene['node_ids'][left]),
                    right_node=str(scene['node_ids'][right]), left_index=int(left), right_index=int(right),
                    left_source=str(scene['sources'][left]), right_source=str(scene['sources'][right]),
                    left_interval=np.asarray(scene['intervals'][left]).tolist(),
                    right_interval=np.asarray(scene['intervals'][right]).tolist(), crop=batch['crop_windows'][index],
                    full_tokens=int(full['observed'][[index, index+count]].sum()),
                    crop_tokens=int(batch['observed'][[index, index+count]].sum())))
                for side, position in (('left', index), ('right', index+count)):
                    selected_times = batch['times'][position, batch['observed'][position]]
                    rows[-1]['crop_'+side+'_interval'] = [float(selected_times[0]), float(selected_times[-1])]
                    rows[-1]['crop_'+side+'_tokens'] = len(selected_times)
                cropped_metadata = dict(rows[-1], left_interval=rows[-1]['crop_left_interval'],
                                        right_interval=rows[-1]['crop_right_interval'])
                rows[-1]['crop_retains_task_eligibility'] = _natural_eligible(cropped_metadata, cropped_metadata, task)
                if not _natural_eligible(rows[-1], rows[-1], task):
                    raise ValueError('079 positive pair must meet original temporal eligibility')
            audit['augmentation'] = batch['augmentation']
        result[task] = dict(rows=rows, vectors={mode: np.concatenate(parts) if parts else
            np.empty((0, 2, model.config.d_model), np.float32) for mode, parts in vectors.items()})
    return result, audits


def _metrics(positive, negative):
    positive, negative = np.asarray(positive), np.asarray(negative)
    scores = np.r_[positive, negative]
    labels = np.r_[np.ones(len(positive), bool), np.zeros(len(negative), bool)]
    if not np.isfinite(scores).all():
        raise ValueError('079 cannot report nonfinite scores')
    auc, ap = _ranking(scores, labels)
    return dict(positive_count=len(positive), negative_count=len(negative),
        prevalence=len(positive)/len(scores) if len(scores) else None,
        roc_auc=auc, average_precision=ap, positive_scores=_quantiles(positive),
        negative_scores=_quantiles(negative))


def evaluate_views(views, config):
    results, saved = {}, {}
    for task_index, task in enumerate(('ITSA', 'MSTSA')):
        rows, vectors = views[task]['rows'], views[task]['vectors']
        rng = np.random.default_rng(np.random.SeedSequence([config['seed'], task_index, 3]))
        chosen, excluded = [], []
        for i, left in enumerate(rows):
            pools = {}
            for domain in ('same_scene', 'cross_scene'):
                candidates = [j for j, right in enumerate(rows)
                    if right['identity'] != left['identity']
                    and ((right['scene_id'] == left['scene_id']) == (domain == 'same_scene'))
                    and ((right['right_source'] == left['left_source']) == (task == 'ITSA'))]
                pools[domain] = candidates
            if not all(pools.values()):
                excluded.append(dict(view_index=i, scene_id=left['scene_id'],
                    reason=[domain+'_no_other_identity_matching_head_source' for domain in pools if not pools[domain]]))
                continue
            chosen.append((i, int(rng.choice(pools['same_scene'])), int(rng.choice(pools['cross_scene']))))
        mapping = np.asarray(chosen, np.int64).reshape(-1, 3)
        conditions, differences = {}, {}
        arrays = dict(mapping=mapping)
        for domain_index, domain in enumerate(('same_scene', 'cross_scene'), 1):
            anchors, negatives = mapping[:, 0], mapping[:, domain_index]
            eligibility = np.array([_natural_eligible(rows[int(i)], rows[int(j)], task)
                                    for i, j in zip(anchors, negatives)], dtype=bool)
            arrays[domain+'_negative_natural_eligible'] = eligibility
            mode_scores = {}
            for mode in ('full', 'train_crop'):
                encoded = vectors[mode]
                positive = np.einsum('ij,ij->i', encoded[anchors, 0], encoded[anchors, 1])
                negative = np.einsum('ij,ij->i', encoded[anchors, 0], encoded[negatives, 1])
                mode_scores[mode] = (positive, negative)
                key = domain+'__'+mode
                arrays[key+'_positive'] = positive
                arrays[key+'_negative'] = negative
                by_scene = {}
                for sid in sorted({rows[int(i)]['scene_id'] for i in anchors}):
                    mask = np.array([rows[int(i)]['scene_id'] == sid for i in anchors])
                    by_scene[sid] = _metrics(positive[mask], negative[mask])
                conditions[key] = dict(pooled=_metrics(positive, negative), scenes=by_scene,
                    natural_eligible_negative_count=int(eligibility.sum()),
                    natural_eligible_paired_subset=_metrics(positive[eligibility], negative[eligibility]),
                    note='Eligibility uses original intervals, not cropped duration; cross-scene clocks remain original per-scene clocks.')
            full_p, full_n = mode_scores['full']; crop_p, crop_n = mode_scores['train_crop']
            differences[domain] = dict(positive_crop_minus_full=_quantiles(crop_p-full_p),
                negative_crop_minus_full=_quantiles(crop_n-full_n),
                margin_crop_minus_full=_quantiles((crop_p-crop_n)-(full_p-full_n)),
                mean_margin_full=float(np.mean(full_p-full_n)) if len(anchors) else None,
                mean_margin_crop=float(np.mean(crop_p-crop_n)) if len(anchors) else None)
        for mode in ('full', 'train_crop'):
            if not np.array_equal(arrays['same_scene__'+mode+'_positive'], arrays['cross_scene__'+mode+'_positive']):
                raise ValueError('079 positive scores changed across domains')
        results[task] = dict(selected_positive_views=len(rows), paired_anchors=len(mapping),
            cropped_positive_eligible_count=sum(row['crop_retains_task_eligibility'] for row in rows),
            excluded_anchors=excluded, conditions=conditions, paired_crop_differences=differences,
            available=len(mapping)>0, global_identity_negative_exclusion=True,
            unique_same_negative_views=len(set(mapping[:, 1].tolist())),
            unique_cross_negative_views=len(set(mapping[:, 2].tolist())))
        saved[task] = dict(rows=rows, arrays=arrays)
    return results, saved


def probe(model, scenes, tsa_config, config, *, device='cpu', check=lambda: None):
    validate_config(config)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    views, sampling = build_views(model, scenes, tsa_config, config, device=device, check=check)
    result, saved = evaluate_views(views, config)
    for name, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), before[name]):
            raise ValueError('079 frozen model state changed')
    return dict(scope=SCOPE, tasks=result, sampling=sampling, frozen_weights_unchanged=True,
        model_training=False, optimizer_updates=0, threshold_selected=False, groups_recomputed=False,
        limitation='Training subset with balanced controlled negatives, not validation/full-candidate AP or independent generalization. Crops use positive context; cross-scene clocks are not aligned. No causal conclusion about original InfoNCE gradient difficulty.'), saved
