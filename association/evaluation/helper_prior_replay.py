"""089: replay successful T-only helper dropout inputs, without learned models.

No checkpoint, network, optimizer, validation payload or predicted group is read.
This measures changed physical starting states; it does not identify the cause
of the completed model's V error. Loss coefficients are not gradient weights.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from association.data.archive import MTADArchive, validate_manifest
from association.data.native import iter_native_split_from_archive
from association.data.timed import prepare_timed_scene
from association.data.motion_evidence import scene_motion_evidence
from association.data.reference_states import iter_reference_states
from association.data.targets import build_scene_fusion_targets
from association.evaluation.fusion_controls import attribute_errors, rms_haversine_scores
from association.evaluation.helper_exposure_audit import read_json
from association.models.sequence_position import sequence_inputs
from association.training.evidence_fusion import TRAIN_MODES, training_members
from association.training.helper_dropout import (drop_helper_fragments, dropout_audit,
    empty_exposure, commit_exposure, SCOPE, KIND)

REGIONS = ('anchor_inside_support', 'anchor_outside_helper_supported', 'group_no_support')
BUCKETS = ('zero', 'one', 'two_or_more')
LIMITS = [
    'Physical T starting-state changes do not apportion the final083 validation regression causally.',
    'Point frequencies, sample-equal loss coefficients and scene-equal descriptions have different denominators.',
    'Loss coefficients are averaging weights, not gradients, Adam changes or clipping contributions.',
    'Support means a time interval, not accurate measurements; labels only rebuild historical T corruption and truth diagnostics.',
    'Original initial tensor equality is inherited from the prior paired checker; no checkpoint is opened here.',
    'Logs retain mask counts, not complete original per-query masks: reconstructed masks are checked against counts and alignment, not an absent saved mask array.',
]


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def rows(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))) for line in handle]


def archive_paths(value):
    """Keep multipart order; resolve aliases without accepting a new source."""
    require(type(value) is list and bool(value) and
            all(type(p) is str and bool(p) for p in value),
            'nonempty original archive path list required')
    return [str(Path(p).resolve()) for p in value]


def bind_sources(dropout_run, linear_run, manifest, alignment, expected_config, *, archives):
    """Validate full source metadata/logs; no checkpoint or prediction reads."""
    validate_manifest(manifest)
    require(manifest.get('dataset_level') == 'Level1', '089 requires Level1')
    ids = manifest['splits']['T_core']
    require(bool(ids) and set(ids).issubset(alignment), 'T-only alignment required')
    for sid in ids:
        value = alignment[sid]
        require(set(value) == {'time_offset_seconds', 'max_gap_seconds'} and
                all(type(v) in (int, float) and np.isfinite(v) for v in value.values()) and
                value['max_gap_seconds'] >= 0, 'finite explicit T reference alignment required')
    cfg = deepcopy(expected_config)
    require(cfg.get('scope') == SCOPE and cfg.get('position_loss') == 'linear' and
            cfg.get('train_splits') == ['T_core'] and cfg.get('motion_observation_policy') == 'canonical_all' and
            cfg.get('augmentation') == dict(kind=KIND, probability=.5, seed=cfg['training']['seed']+83000),
            'frozen083 dropout configuration required')
    expected_linear = deepcopy(cfg); expected_linear.pop('augmentation'); expected_linear['scope'] = 'sequence-position-v1'
    updates, batch_size = cfg['training']['updates'], cfg['training']['batch_size']
    require(type(updates) is int and updates > 0 and type(batch_size) is int and batch_size > 0,
            'positive successful-update budget required')
    roots = [Path(dropout_run).resolve(), Path(linear_run).resolve()]
    requested_archives = archive_paths(archives)
    records, logs, outer = [], [], []
    for root, expected, profile in zip(roots, (cfg, expected_linear), ('helper-dropout4000', 'extended4000')):
        run, execution, fit = [read_json(root/name) for name in ('run.json', 'execution.json', 'fit/run.json')]
        require(run.get('status') == fit.get('status') == 'COMPLETED' and
                run.get('optimizer_updates') == fit.get('optimizer_updates') == updates and
                run.get('test_opened') is fit.get('test_opened') is False and
                run.get('completed_endpoints') == ['initial', 'final'] and run.get('retries') == 0,
                'source must be completed fixed endpoint, no test or retries')
        require(all(same(record.get('configuration'), expected) for record in (run, execution, fit)) and
                fit.get('scope') == expected['scope'] and fit.get('motion_observation_policy') == 'canonical_all' and
                fit.get('training_scene_ids') == ids and fit.get('parameter_count') == 105154 and
                fit.get('final_checkpoint') == 'final.pt', 'source configuration/population/parameter mismatch')
        require(run.get('profile') == profile and all(run.get(k) is False for k in
            ('validation_selection', 'association_recomputed', 'groups_recomputed', 'threshold_reselected')),
            'source experiment profile or frozen protocol mismatch')
        args = execution.get('arguments', {})
        require(archive_paths(args.get('archive')) == requested_archives,
                'requested archive paths/order differ from original execution')
        require(args.get('profile') == profile and Path(args.get('output', '')).resolve() == root and
                args.get('source_system') == run.get('source_system') and execution.get('test_opened') is False,
                'execution source/output binding mismatch')
        for key, wanted in (('manifest', manifest), ('alignment', alignment), ('config', expected)):
            require(same(read_json(args[key]), wanted), 'original execution '+key+' differs or unavailable')
        current = rows(root/'fit/training.jsonl')
        require(len(current) == updates, 'complete successful training log required')
        logs.append(current); records.append(fit); outer.append(run)
    require(outer[0]['source_system'] == outer[1]['source_system'] and
            Path(outer[0].get('paired_control', '')).resolve() == roots[1], '083 must pair with this072C/056 source')
    pair = records[0].get('paired_control', {})
    require(pair.get('status') == 'COMPLETED' and pair.get('actual_successful_batches_matched') == updates and
            all(pair.get(k) is True for k in ('initial_parameters_exact', 'initial_sampler_rng_exact',
                'initial_torch_rng_exact', 'sampling_exposure_exact')) and
            Path(pair.get('fit', '')).resolve() == roots[1]/'fit', 'original paired provenance is incomplete')
    require(records[0].get('augmentation_applied_only_to_training') is True and
            same(records[0].get('augmentation'), cfg['augmentation']), 'source augmentation policy mismatch')
    preparation = read_json(roots[0]/'preparation.json')
    require(preparation.get('training_scene_ids') == ids and
            preparation.get('validation_scene_ids') == manifest['splits']['V_select'] and
            same(preparation.get('data_config'), cfg['data']) and
            preparation.get('motion_observation_policy') == 'canonical_all' and
            preparation.get('test_opened') is False, 'source preparation population/input contract mismatch')
    counts = {str(s['scene_id']): s['node_count'] for s in preparation['scenes'] if s['split'] == 'T_core'}
    require(list(counts) == ids and all(type(n) is int and n > 0 for n in counts.values()),
            'preparation must retain every nonempty T scene in order')
    rng, augment_rng = np.random.default_rng(cfg['training']['seed']), np.random.default_rng(cfg['augmentation']['seed'])
    plans, exposure = defaultdict(list), empty_exposure()
    sampled_scenes, sampled_nodes = Counter(), Counter()
    mode_exposure = {m: dict(updates=0, anchors=0, valid_position_points=0) for m in TRAIN_MODES}
    valid_count = 0
    for step, (new, old) in enumerate(zip(*logs), 1):
        mode = TRAIN_MODES[(step-1) % len(TRAIN_MODES)]
        require(new.get('step') == old.get('step') == step and new.get('mode') == old.get('mode') == mode and
                same(new.get('batch'), old.get('batch')) and len(new.get('batch', [])) == batch_size and
                len(new.get('augmentation_batch', [])) == batch_size, 'original batch sequence/count differs')
        require(same(new.get('augmentation_rng_before'), augment_rng.bit_generator.state), 'augmentation RNG before mismatch')
        batch_valid = 0
        for sample, audit in zip(new['batch'], new['augmentation_batch']):
            sid = ids[int(rng.integers(len(ids)))]; anchor = int(rng.integers(counts[sid]))
            require(sample['scene_id'] == sid and type(sample['anchor_index']) is int and sample['anchor_index'] == anchor,
                    'original sampler replay differs')
            members = sample['members']
            require(members[0] == anchor and all(type(x) is int and 0 <= x < counts[sid] for x in members), 'invalid original members')
            retained, mask = drop_helper_fragments(members, augment_rng, .5)
            require(audit.get('retained_members') == retained and audit.get('drop_mask') == mask and
                    audit.get('original_members') == members, 'augmentation sample replay differs')
            valid = sample['valid_position_points']
            require(type(valid) is int and 0 <= valid <= cfg['data']['length'], 'invalid source mask count')
            sampled_scenes[sid] += 1; sampled_nodes[sample['node_id']] += 1; batch_valid += valid
            plans[sid].append(dict(step=step, mode=mode, sample=sample, augmentation=audit))
        require(same(new.get('augmentation_rng_after'), augment_rng.bit_generator.state), 'augmentation RNG after mismatch')
        require(new['objectives']['valid_position_points'] == old['objectives']['valid_position_points'] == batch_valid,
                'successful batch target denominator differs')
        valid_count += batch_valid
        mode_exposure[mode]['updates'] += 1; mode_exposure[mode]['anchors'] += batch_size
        mode_exposure[mode]['valid_position_points'] += batch_valid
        commit_exposure(exposure, new['augmentation_batch'])
    require(same(records[0]['augmentation_exposure'], exposure), 'augmentation exposure differs from logs')
    expected_exposure = dict(successful_updates=updates, sampled_anchors=updates*batch_size,
        valid_position_points=valid_count, scenes=dict(sampled_scenes), anchors=dict(sampled_nodes), modes=mode_exposure)
    for record in records:
        require(same({k: record['exposure'][k] for k in expected_exposure}, expected_exposure),
                'successful sampling exposure mismatch')
    return dict(configuration=cfg, plans=dict(plans), preparation=preparation, node_counts=counts,
        verification=dict(status='LOG_REPLAY_VERIFIED', successful_updates=updates, sample_count=updates*batch_size,
            valid_position_points=valid_count, original_sampler_replayed=True, augmentation_rng_replayed=True,
            original_batch_pairs_exact=True, original_initial_tensor_check='inherited prior pair proof; no checkpoint read',
            augmentation_exposure=exposure, sources=[str(p) for p in roots],
            archive_paths=requested_archives, archive_binding='both original executions; ordered resolved paths'))


def iter_training_inputs(archives, manifest, alignment, config, *, bound, check=None):
    """Stream only T native inputs/reference; do not call mixed T/V loaders."""
    ids = manifest['splits']['T_core']; data = config['data']
    reports = {s['scene_id']: s for s in bound['preparation']['scenes'] if s['split'] == 'T_core'}
    reference_reports = {r['scene_id']: r for r in bound['preparation']['reference_members_read']
                         if r['split'] == 'T_core'}
    require(list(reports) == ids and set(reference_reports) == set(ids), 'original T preparation records required')
    observation_members = {(r['kind'], r['scene_id'], r['member'])
                           for r in bound['preparation']['observation_members_read'] if r['scene_id'] in ids}
    checked_members = 0
    with MTADArchive(archives) as archive, closing(iter_reference_states(
            archives, manifest, split='T_core', scene_ids=ids)) as references:
        for native, reference in zip(iter_native_split_from_archive(archive, manifest, 'T_core',
                max_points=data['max_observations'], check=check), references):
            sid = native['scene_id']
            require(reference['scene_id'] == sid, 'T native/reference order mismatch')
            if check: check()
            scene = prepare_timed_scene(native, **{k: data[k] for k in ('length', 'max_observations', 'time_unit_seconds')})
            labels = archive.read_labels(sid)
            identities = list(dict.fromkeys('-'.join(r['mmsi'].split('-')[:2]) for r in labels))
            scene['motion'] = scene_motion_evidence(scene, motion_observation_policy='canonical_all')
            scene['motion_observation_policy'] = 'canonical_all'
            scene['fusion_targets'] = build_scene_fusion_targets(scene, reference, target_identity_ids=identities,
                longitude_policy=data['reference_longitude_policy'], **alignment[sid])
            require(same(reference_reports[sid], dict(scene_id=sid, split='T_core', member=reference['member'],
                raw_row_count=len(reference['rows']), issue_count=len(reference['issues']))),
                'rebuilt T reference member audit differs')
            targets = scene['fusion_targets']; valid_counts = np.zeros(4, dtype=np.int64)
            for target in targets.values(): valid_counts += target['valid'].sum(0)
            fields = dict(component_id=scene['component_id'], node_count=len(scene['native_values']),
                native_point_count=int(scene['native_observed'].sum()), candidate_count=len(scene['pairs']),
                positive_count=int(scene['labels'].sum()), target_count=len(targets),
                target_without_input_count=len(set(targets)-set(scene['identities'])),
                target_without_reference_count=sum(t['report']['missing_identity'] for t in targets.values()),
                reference_valid_points_per_feature=valid_counts.tolist(), input_audit=scene['timed_input_audit'],
                quarantine_audit=scene['audit'], motion_native_point_count=int(scene['motion']['native_count'].sum()),
                reference_alignment=dict(**alignment[sid], longitude_policy=data['reference_longitude_policy'],
                    policy='aligned_reference_time = raw_reference_time - time_offset_seconds',
                    query_grid_policy=scene['timed_input_audit']['grid_policy'],
                    clock_mapping_source='explicit caller per-scene alignment'))
            for key, value in fields.items():
                require(same(reports[sid].get(key), value), 'rebuilt T preparation differs: '+sid+'/'+key)
            for member in archive.members_read[checked_members:]:
                require(member['scene_id'] == sid and
                        (member['kind'], sid, member['member']) in observation_members,
                        'rebuilt observation member absent from original preparation')
            checked_members = len(archive.members_read)
            scene.pop('tracks', None)
            archive._label_cache.pop(sid, None)
            yield scene


def training_view(scene):
    """Reconstruct historical training membership distances, without any grouping."""
    require(scene['split'] == 'T_core', 'only T inputs may enter prior replay')
    motion = scene['motion']; n = len(scene['identities']); q = motion['states'].shape[1]
    pairs = np.column_stack(np.triu_indices(n, 1)); distances = np.full((n, n), np.inf)
    for start in range(0, len(pairs), 4096):
        subset = pairs[start:start+4096]
        value = -rms_haversine_scores(motion['states'][..., :2], subset)
        distances[subset[:, 0], subset[:, 1]] = value
        distances[subset[:, 1], subset[:, 0]] = value
    targets = [scene['fusion_targets'][str(identity)] for identity in scene['identities']]
    return dict(scene_id=str(scene['scene_id']), split='T_core', identities=np.asarray(scene['identities'], str),
        node_ids=np.asarray(scene['node_ids'], str), motion=motion, distances=distances,
        target=np.asarray([t['states'] for t in targets], np.float64).reshape(n, q, 4),
        valid=np.asarray([t['valid'] for t in targets], bool).reshape(n, q, 4))


@dataclass
class Totals:
    all_query_count: int = 0
    all_query_displacement_sum_m: float = 0.
    point_count: int = 0
    full_sum_m: float = 0.
    dropped_sum_m: float = 0.
    full_square_sum_m2: float = 0.
    dropped_square_sum_m2: float = 0.
    displacement_sum_m: float = 0.
    harm_sum_m: float = 0.
    improvement_sum_m: float = 0.
    coefficient_mass: float = 0.
    full_coefficient_m: float = 0.
    dropped_coefficient_m: float = 0.
    displacement_coefficient_m: float = 0.
    harm_coefficient_m: float = 0.
    improvement_coefficient_m: float = 0.

    def add_geometry(self, displacement):
        require(np.isfinite(displacement).all(), 'finite all-query prior geometry required')
        self.all_query_count += len(displacement)
        self.all_query_displacement_sum_m += float(displacement.sum())

    def add(self, full, dropped, displacement, coefficient):
        require(len(full) == len(dropped) == len(displacement) and all(np.isfinite(v).all() for v in
                (full, dropped, displacement)) and coefficient >= 0, 'finite paired prior errors required')
        harm, improvement = np.maximum(dropped-full, 0), np.maximum(full-dropped, 0)
        self.point_count += len(full)
        for name, value in (('full', full), ('dropped', dropped), ('displacement', displacement),
                            ('harm', harm), ('improvement', improvement)):
            total = float(value.sum())
            setattr(self, name+'_sum_m', getattr(self, name+'_sum_m')+total)
            setattr(self, name+'_coefficient_m', getattr(self, name+'_coefficient_m')+total*coefficient)
        self.full_square_sum_m2 += float(np.square(full).sum())
        self.dropped_square_sum_m2 += float(np.square(dropped).sum())
        self.coefficient_mass += len(full)*coefficient

    def describe(self, total_samples):
        row = dict(vars(self))
        row.update(point_mae_full_m=self.full_sum_m/self.point_count if self.point_count else None,
            all_query_displacement_mean_m=self.all_query_displacement_sum_m/self.all_query_count
                if self.all_query_count else None,
            point_mae_dropped_m=self.dropped_sum_m/self.point_count if self.point_count else None,
            point_rmse_full_m=float(np.sqrt(self.full_square_sum_m2/self.point_count)) if self.point_count else None,
            point_rmse_dropped_m=float(np.sqrt(self.dropped_square_sum_m2/self.point_count)) if self.point_count else None,
            sample_equal_linear_loss_full=self.full_coefficient_m/1000.,
            sample_equal_linear_loss_dropped=self.dropped_coefficient_m/1000.,
            loss_global_denominator_samples=total_samples,
            signed_absolute_error_change_m=self.dropped_sum_m-self.full_sum_m,
            signed_loss_coefficient_change_m=self.dropped_coefficient_m-self.full_coefficient_m)
        return row


def audit_inputs(bound, scenes, *, on_scene=None):
    """Verify reconstructed inputs and accumulate disjoint region/transition cells."""
    cfg = bound['configuration']; total_samples = cfg['training']['updates']*cfg['training']['batch_size']
    global_totals = Totals(); cross = {(m, r, a, b): Totals() for m in TRAIN_MODES for r in REGIONS
        for a in BUCKETS for b in BUCKETS}
    seen, per_scene, samples, zero, valid_points = [], [], 0, 0, 0
    for raw in scenes:
        sid = str(raw['scene_id']); seen.append(sid)
        require(sid in bound['node_counts'] and len(raw['identities']) == bound['node_counts'][sid], 'T node population differs')
        scene = training_view(raw); totals = Totals(); scene_samples = 0; scene_zero = 0
        # Cache repeated membership calculations only within this scene.
        cache = {}
        for plan in bound['plans'].get(sid, []):
            mode, old = plan['mode'], plan['sample']; anchor = old['anchor_index']
            members = training_members(scene, anchor, mode)
            valid = scene['valid'][anchor, :, :2].all(-1); nvalid = int(valid.sum())
            support = scene['motion']['support']; original_count = support[members[1:]].sum(0)
            eligible = ~support[anchor] & (original_count > 0)
            expected = dict(scene_id=sid, anchor_index=anchor, node_id=str(scene['node_ids'][anchor]), members=members,
                valid_position_points=nvalid, eligible_query_points=int(eligible.sum()),
                eligible_valid_position_points=int((eligible & valid).sum()))
            require(same(expected, old), 'rebuilt original members/node/validity differ from source logs')
            audit = plan['augmentation']; retained = audit['retained_members']
            require(same(dropout_audit(scene, members, retained, audit['drop_mask'], valid), audit),
                    'rebuilt augmentation support/identity/validity differs from source logs')
            samples += 1; scene_samples += 1; zero += int(nvalid == 0); scene_zero += int(nvalid == 0)
            valid_points += nvalid
            for group in (members, retained):
                key = tuple(group)
                if key not in cache: cache[key] = sequence_inputs(scene['motion'], group)[0].numpy()
            full, dropped = cache[tuple(members)], cache[tuple(retained)]
            require(np.array_equal(full[~eligible, :2], dropped[~eligible, :2]),
                    'deletion changed fixed prior outside original opportunity region')
            truth = scene['target'][anchor]
            e0 = attribute_errors(full[valid], truth[valid])['position']
            e1 = attribute_errors(dropped[valid], truth[valid])['position']
            all_distance = attribute_errors(full, dropped)['position']; distance = all_distance[valid]
            global_totals.add_geometry(all_distance); totals.add_geometry(all_distance)
            coefficient = 1./(total_samples*nvalid) if nvalid else 0.
            global_totals.add(e0, e1, distance, coefficient); totals.add(e0, e1, distance, coefficient)
            region = np.where(support[anchor], 0, np.where(original_count > 0, 1, 2))
            before = np.minimum(original_count, 2)
            after = np.minimum(support[retained[1:]].sum(0), 2)
            require(bool(np.all(after <= before)), 'dropout cannot create supported helpers')
            for r, region_name in enumerate(REGIONS):
                for a, before_name in enumerate(BUCKETS):
                    for b, after_name in enumerate(BUCKETS):
                        all_selected = (region == r) & (before == a) & (after == b)
                        cell = cross[(mode, region_name, before_name, after_name)]
                        cell.add_geometry(all_distance[all_selected])
                        selected = all_selected[valid]
                        cell.add(
                            e0[selected], e1[selected], distance[selected], coefficient)
        result = dict(scene_id=sid, sample_count=scene_samples, zero_supervision_sample_count=scene_zero,
            exposed=scene_samples > 0, **totals.describe(total_samples))
        per_scene.append(result)
        if on_scene: on_scene(result)
    require(seen == list(bound['node_counts']), 'full ordered T scene population required')
    require(samples == total_samples and valid_points == bound['verification']['valid_position_points'],
            'reconstructed successful sample/valid-point population differs')
    require(sum(t.point_count for t in cross.values()) == global_totals.point_count, 'cross table population does not close')
    for field in vars(global_totals):
        require(np.isclose(sum(getattr(t, field) for t in cross.values()), getattr(global_totals, field), rtol=1e-10, atol=1e-6),
                'cross table additive field does not close: '+field)
    require(np.isclose(global_totals.coefficient_mass, (samples-zero)/samples, rtol=1e-12, atol=1e-12),
            'zero-supervision sample denominator changed')
    exposed = [row for row in per_scene if row['sample_count']]
    with_points = [row for row in exposed if row['point_count']]
    scene_equal = dict(exposed_scene_count=len(exposed), point_supervised_scene_count=len(with_points),
        full_point_mean_mae_m=float(np.mean([r['point_mae_full_m'] for r in with_points])) if with_points else None,
        dropped_point_mean_mae_m=float(np.mean([r['point_mae_dropped_m'] for r in with_points])) if with_points else None,
        full_sample_mean_loss=float(np.mean([r['sample_equal_linear_loss_full']*samples/r['sample_count'] for r in exposed])),
        dropped_sample_mean_loss=float(np.mean([r['sample_equal_linear_loss_dropped']*samples/r['sample_count'] for r in exposed])))
    return dict(status='COMPLETED', global_metrics=global_totals.describe(total_samples),
        successful_sample_count=samples, zero_supervision_sample_count=zero,
        training_scene_count=len(seen), unexposed_scene_ids=[r['scene_id'] for r in per_scene if not r['sample_count']],
        scene_equal=scene_equal, cross_table=[dict(mode=m, original_region=r, original_bucket=a, retained_bucket=b,
            **value.describe(total_samples)) for (m, r, a, b), value in cross.items()],
        per_scene=per_scene, limitations=LIMITS)
