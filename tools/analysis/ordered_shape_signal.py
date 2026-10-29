"""126: four matched 1000-update probes of ordered Q64 pair-distance shape.

Raw canonical measurements and identity labels only; no reference or test data.
The complete 46-column shared cache lives on disk; batches read selected rows.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from association.data.archive import MTADArchive, validate_manifest
from association.data.legacy_dataset import validate_archive
from association.data.motion_evidence import DATA, EARTH_RADIUS_M, scene_motion_evidence
from association.data.native import _record
from association.data.timed import prepare_timed_scene
from association.evaluation.association import evaluate_scenes, select_threshold, compare_predictions
from association.evaluation.tsactfer_level1 import scope_metrics
from association.models.evidence_association import FEATURE_NAMES, features_from_motion
from association.training.common import _update
from association.training.evidence_association import _auc
from association.training.sampling import BalancedScenePairSampler
from association.utils import runtime

IDEA = 'IDEA-ORDERED-SHAPE-SIGNAL-20261029'
ARMS = ('SUMMARY', 'ORDERED', 'DISTRIBUTION', 'SHUFFLED')
SEED, UPDATES, BATCH, CHUNK = 20, 1000, 256, 4096
RAM = 8*1024**3
FEATURE_COUNT, CACHE_COUNT = 30, 46
EXTRA_COLUMNS = dict(ORDERED=slice(22, 30), DISTRIBUTION=slice(30, 38), SHUFFLED=slice(38, 46))
DCT = np.sqrt(2./64.)*np.cos(np.pi*(np.arange(64)[:, None]+.5)*np.arange(1, 9)[None]/64.)


def keyed_permutations(split_index, scene_index, pair_indices):
    """Pair-specific deterministic rank keys; not an exact permutation test.

    SplitMix64 is a bijective uint64 mixer. Distinct query counters within a
    row therefore have distinct rank keys. No labels or values seed the keys.
    """
    pairs = np.asarray(pair_indices, np.uint64)
    if pairs.ndim != 1 or split_index not in (0, 1) or scene_index < 0:
        raise ValueError('explicit T/V index, scene ordinal and original pair indices required')
    with np.errstate(over='ignore'):
        keys = (np.uint64(SEED) + np.uint64(split_index+1)*np.uint64(0xD1B54A32D192ED03)
                + np.uint64(scene_index+1)*np.uint64(0x94D049BB133111EB)
                + pairs[:, None]*np.uint64(0x9E3779B97F4A7C15)
                + np.arange(64, dtype=np.uint64)[None]*np.uint64(0xBF58476D1CE4E5B9))
        keys = (keys ^ (keys >> np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        keys = (keys ^ (keys >> np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        keys ^= keys >> np.uint64(31)
    return np.argsort(keys, axis=1, kind='stable')


def extra_features(states, pairs, *, split_index, scene_index, pair_start):
    """Prediction-only shape API: accepts neither identities nor labels."""
    states, pairs = np.asarray(states, np.float64), np.asarray(pairs)
    if (states.ndim != 3 or states.shape[1:] != (64, 4) or not np.isfinite(states).all()
            or pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu'
            or (pairs < 0).any() or (pairs >= len(states)).any() or (pairs[:, 0] == pairs[:, 1]).any()
            or pair_start < 0):
        raise ValueError('complete finite Q64 states and original distinct pair rows required')
    left, right = np.sort(pairs, axis=1).T
    a, b = states[left], states[right]
    la, lb = np.deg2rad(a[..., 0]), np.deg2rad(b[..., 0])
    dl = np.deg2rad((a[..., 1]-b[..., 1]+180.) % 360.-180.)
    h = np.clip(np.sin((la-lb)/2.)**2+np.cos(la)*np.cos(lb)*np.sin(dl/2.)**2, 0., 1.)
    distance = 2.*EARTH_RADIUS_M*np.arctan2(np.sqrt(h), np.sqrt(1.-h))
    z = np.log1p(distance/1000.)
    if not len(z):
        return np.empty((0, 24), np.float64)
    ordered = (z-z.mean(1, keepdims=True)) @ DCT
    distribution = np.quantile(z, np.arange(1, 9)/10., axis=1).T
    perm = keyed_permutations(split_index, scene_index, np.arange(pair_start, pair_start+len(pairs)))
    shuffled = np.take_along_axis(z, perm, axis=1)
    shuffled = (shuffled-shuffled.mean(1, keepdims=True)) @ DCT
    value = np.column_stack((ordered, distribution, shuffled))
    if value.shape != (len(pairs), 24) or not np.isfinite(value).all():
        raise FloatingPointError('invalid ordered shape feature block')
    return value


def close_map(array):
    if getattr(array, '_mmap', None) is not None:
        array._mmap.close()


def _same(a, b, label):
    if not np.array_equal(a, b):
        raise ValueError(label+' differs')


def _bind_v(motion, prepared, source):
    with np.load(source/'predictions'/f'states-{prepared["scene_id"]}.npz', allow_pickle=False) as old:
        for key in ('node_ids', 'sources', 'identities'):
            _same(prepared[key], old[key], '056 canonical '+key)
        for key, saved in (('states', 'native_states'), ('query_seconds', 'query_seconds'),
                           ('support', 'support'), ('nearest_seconds', 'nearest_seconds'),
                           ('exact_observation', 'exact_observation')):
            _same(motion[key], old[saved], '056 canonical '+key)


def prepare_split(archive, manifest, split, output, source_system, check):
    """Same native loader/canonical_all operators, streaming one scene at a time."""
    if split not in ('T_core', 'V_select'):
        raise ValueError('T_test sealed')
    directory = output/'private'/'cache'/split
    directory.mkdir(parents=True, exist_ok=False)
    scenes = []
    for ordinal, sid in enumerate(manifest['splits'][split]):
        check(phase='preparing_'+split, preparation_completed_scenes=len(scenes), last_scene_id=sid)
        native = _record(archive.load_scene(sid), manifest['components'][sid], split,
                         DATA['max_observations'], split_unit=manifest.get('split_unit'))
        query = None
        if not native['tracks']:
            labels = archive.read_labels(sid)
            if not labels:
                raise ValueError('empty canonical scene has no declared query bounds')
            query = np.linspace(min(r['t_s'] for r in labels), max(r['t_e'] for r in labels), 64)
            if query[-1] <= query[0]:
                raise ValueError('empty scene query bounds invalid')
        prepared = prepare_timed_scene(native, **DATA, query_times=query)
        motion = scene_motion_evidence(prepared, motion_observation_policy='canonical_all')
        if split == 'V_select':
            _bind_v(motion, prepared, source_system)
        n, pairs = len(prepared['node_ids']), np.asarray(prepared['pairs'])
        if not np.array_equal(pairs, np.column_stack(np.triu_indices(n, 1))):
            raise ValueError('complete original unordered pair population required')
        identities = np.asarray(prepared['identities'])
        raw_labels = np.asarray(prepared['labels'])
        if not np.isin(raw_labels, [0, 1]).all():
            raise ValueError('original pair labels must be binary before boolean conversion')
        labels = raw_labels.astype(bool)
        _same(labels, identities[pairs[:, 0]] == identities[pairs[:, 1]], 'canonical identity labels')
        features_path, base_path = directory/f'{sid}-features.npy', directory/f'{sid}-base.npy'
        features = np.lib.format.open_memmap(features_path, mode='w+', dtype=np.float32, shape=(len(pairs), CACHE_COUNT))
        base = np.lib.format.open_memmap(base_path, mode='w+', dtype=np.float64, shape=(len(pairs),))
        joint = np.empty(len(pairs), np.uint8)  # exact Q64 counts, bounded0..64
        for start in range(0, len(pairs), CHUNK):
            check()
            block = pairs[start:start+CHUNK]
            normal, base[start:start+len(block)] = features_from_motion(motion, block)
            features[start:start+len(block), :22] = normal
            features[start:start+len(block), 22:] = extra_features(motion['states'], block,
                split_index=int(split == 'V_select'), scene_index=ordinal, pair_start=start)
            joint[start:start+len(block)] = (motion['support'][block[:, 0]] & motion['support'][block[:, 1]]).sum(1)
        features.flush(); base.flush(); close_map(features); close_map(base)
        pair_path, label_path = directory/f'{sid}-pairs.npy', directory/f'{sid}-labels.npy'
        np.save(pair_path, pairs); np.save(label_path, labels)
        intervals = np.asarray([[track['canonical_times'][0], track['canonical_times'][-1]]
                                for track in prepared['tracks']], np.float64).reshape(n, 2)
        meta = dict(scene_id=str(sid), component_id=str(manifest['components'][sid]), split=split,
            node_ids=np.asarray(prepared['node_ids']), sources=np.asarray(prepared['sources']),
            identities=identities, intervals=intervals, joint_support_queries=joint,
            pair_count=len(pairs), positive_count=int(labels.sum()),
            features_path=features_path, base_path=base_path, pairs_path=pair_path, labels_path=label_path)
        np.savez_compressed(directory/f'{sid}-metadata.npz', **{key: meta[key] for key in
            ('node_ids', 'sources', 'identities', 'intervals', 'joint_support_queries')})
        scenes.append(meta)
        del motion, prepared, native, pairs, labels
    return scenes


def fit_normalizers(scenes, check=lambda **_: None):
    if not scenes or any(s['split'] != 'T_core' for s in scenes):
        raise ValueError('normalizers require T_core only')
    count, mean, m2 = 0, np.zeros(CACHE_COUNT), np.zeros(CACHE_COUNT)
    for scene in scenes:
        data = np.load(scene['features_path'], mmap_mode='r', allow_pickle=False)
        try:
            for start in range(0, len(data), CHUNK):
                check()
                x = np.asarray(data[start:start+CHUNK], np.float64)
                size = len(x)
                delta, local_mean = x.mean(0)-mean, x.mean(0)
                total = count+size
                m2 += np.square(x-local_mean).sum(0)+delta*delta*(count*size/total)
                mean += delta*(size/total)
                count = total
        finally:
            close_map(data)
    if not count or not np.isfinite(mean).all() or not np.isfinite(m2).all():
        raise ValueError('finite nonempty original T feature population required')
    std = np.sqrt(np.maximum(m2/count, 0.)); std[std < 1e-6] = 1.
    normalizers = {}
    for arm in ARMS:
        extra_mean = np.zeros(8) if arm == 'SUMMARY' else mean[EXTRA_COLUMNS[arm]]
        extra_std = np.ones(8) if arm == 'SUMMARY' else std[EXTRA_COLUMNS[arm]]
        normalizers[arm] = dict(mean=np.r_[mean[:22], extra_mean].tolist(), std=np.r_[std[:22], extra_std].tolist(),
            fitted_split='T_core', pair_count=count, old_feature_names=list(FEATURE_NAMES),
            extra_feature_rule=arm, cache_dtype='float32', accumulation_dtype='float64',
            weighting='one weight per original T pair; 4096-row Welford')
    for arm in ARMS:
        _same(normalizers[arm]['mean'][:22], normalizers['SUMMARY']['mean'][:22], 'shared22 mean')
        _same(normalizers[arm]['std'][:22], normalizers['SUMMARY']['std'][:22], 'shared22 std')
    return normalizers


def sampler_input(scenes):
    return [dict(scene_id=s['scene_id'], pairs=np.load(s['pairs_path'], mmap_mode='r', allow_pickle=False),
                 labels=np.load(s['labels_path'], mmap_mode='r', allow_pickle=False)) for s in scenes]


def build_plan(scenes, steps=UPDATES):
    inputs = sampler_input(scenes)
    try:
        sampler = BalancedScenePairSampler(inputs, SEED)
    finally:
        for row in inputs:
            close_map(row['pairs']); close_map(row['labels'])
    plan = np.empty((steps, BATCH, 2), np.int64)
    for step in range(steps):
        groups = sampler.sample(BATCH)
        plan[step, :, 0] = np.concatenate([np.full(len(indices), sid, np.int64) for sid, indices in groups])
        plan[step, :, 1] = np.concatenate([indices for _, indices in groups])
        sampler.record_completed_batch(groups)
    return plan, sampler.coverage()


def groups_from_plan(batch):
    return [(int(index), batch[batch[:, 0] == index, 1]) for index in dict.fromkeys(batch[:, 0].tolist())]


def read_features(scene, indices, arm, normalizer):
    data = np.load(scene['features_path'], mmap_mode='r', allow_pickle=False)
    try:
        old = np.asarray(data[indices, :22])
        extra = np.zeros((len(old), 8), np.float32) if arm == 'SUMMARY' else np.asarray(data[indices, EXTRA_COLUMNS[arm]])
        value = np.column_stack((old, extra))
    finally:
        close_map(data)
    value = ((value-np.asarray(normalizer['mean']))/np.asarray(normalizer['std'])).astype(np.float32)
    if not np.isfinite(value).all():
        raise FloatingPointError('nonfinite normalized probe input')
    return value


class ShapeProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(30, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        with torch.no_grad():
            self.network[0].weight[:, 22:] = 0.
            self.network[-1].weight.zero_(); self.network[-1].bias.zero_()

    def forward(self, features, base):
        return base.to(torch.float64)+self.network(features.to(torch.float32)).squeeze(-1).to(torch.float64)


@torch.no_grad()
def score(model, scenes, arm, norm, device, check):
    model.eval(); records = []
    for scene in scenes:
        check()
        base = np.load(scene['base_path'], mmap_mode='r', allow_pickle=False)
        scores = np.empty(scene['pair_count'], np.float64)
        try:
            for start in range(0, len(base), CHUNK):
                check()
                indices = np.arange(start, min(start+CHUNK, len(base)))
                x = read_features(scene, indices, arm, norm)
                scores[indices] = model(torch.as_tensor(x, device=device),
                    torch.as_tensor(np.asarray(base[indices]).copy(), dtype=torch.float64, device=device)).cpu().numpy()
        finally:
            close_map(base)
        if not np.isfinite(scores).all():
            raise FloatingPointError('nonfinite final scores')
        row = {key: scene[key] for key in ('scene_id', 'component_id', 'node_ids', 'sources',
                                         'identities', 'intervals', 'joint_support_queries')}
        row.update(scores=scores, pairs=np.load(scene['pairs_path'], allow_pickle=False),
                   labels=np.load(scene['labels_path'], allow_pickle=False))
        records.append(row)
    return records


def describe(records, threshold):
    labels = np.concatenate([r['labels'] for r in records]); scores = np.concatenate([r['scores'] for r in records])
    supports = {}
    for label, has_support in (('joint_support', True), ('no_joint_support', False)):
        selected = []
        for row in records:
            mask = (row['joint_support_queries'] > 0) == has_support
            selected.append(dict(row, **{k: row[k][mask] for k in ('pairs', 'labels', 'scores', 'joint_support_queries')}))
        supports[label] = evaluate_scenes(selected, threshold)
    return dict(raw=evaluate_scenes(records, threshold), pooled_auc=_auc(labels, scores),
                paper_eligible=scope_metrics(records, threshold), support=supports)


def train_arm(scenes, plan, arm, norm, initial, device, directory, check):
    directory.mkdir(parents=True, exist_ok=False)
    runtime.seed_everything(SEED)
    model = ShapeProbe().to(device); model.load_state_dict(initial, strict=True)
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor.cpu(), initial[name]):
            raise ValueError('shared initial model differs')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = []
    def save(path):
        torch.save(dict(idea_id=IDEA, arm=arm, update=len(history), normalizer=norm,
            state_dict={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            optimizer_state=optimizer.state_dict(), parameter_count=sum(p.numel() for p in model.parameters())), path)
    try:
        for step, batch in enumerate(plan):
            check(phase='training', active_arm=arm, active_arm_updates=step)
            xs, bases, labels = [], [], []
            for sid, indices in groups_from_plan(batch):
                scene = scenes[sid]
                xs.append(read_features(scene, indices, arm, norm))
                for key, destination in (('base_path', bases), ('labels_path', labels)):
                    data = np.load(scene[key], mmap_mode='r', allow_pickle=False)
                    try:
                        destination.append(np.asarray(data[indices]).copy())
                    finally:
                        close_map(data)
            x, base, target = np.concatenate(xs), np.concatenate(bases), np.concatenate(labels)
            if len(x) != BATCH or int(target.sum()) != BATCH//2:
                raise ValueError('shared batch plan is not original balanced256')
            model.train()
            score_value = model(torch.as_tensor(x, device=device), torch.as_tensor(base, dtype=torch.float64, device=device))
            loss = F.binary_cross_entropy_with_logits(score_value, torch.as_tensor(target, dtype=torch.float64, device=device))
            loss_value, gradient = _update(loss, optimizer, model.parameters(), 1.)
            history.append(dict(update=step+1, loss=loss_value, gradient_norm=gradient))
            check(active_arm_updates=step+1)
            if (step+1) % 50 == 0 or step+1 == len(plan):
                runtime.write_json(directory/'history.json', history)
        save(directory/'final.pt')
    except BaseException:
        runtime.write_json(directory/'history.json', history)
        try:
            save(directory/'partial.pt')
        except Exception as error:
            runtime.write_json(directory/'partial-checkpoint-error.json', dict(error=f'{type(error).__name__}: {error}'))
        raise
    return model


def run(args):
    output, source = args.output.resolve(), args.source_system.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('new output must not overlap immutable056 source')
    if not 1 <= args.cpu_threads <= 4 or not str(args.device).startswith('cuda:'):
        raise ValueError('one explicit cuda device and1..4 CPU threads; no CPU fallback')
    manifest = runtime.read_json(args.manifest)
    validate_manifest(manifest)
    if (manifest.get('dataset_level') != 'Level1' or manifest.get('split_unit') != 'scene'
            or len(manifest['splits']['T_core']) != 2100 or len(manifest['splits']['V_select']) != 300):
        raise ValueError('formal Level1 scene T2100/V300 required; T_test sealed')
    output.mkdir(parents=True, exist_ok=False); (output/'private').mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.1', started_at=datetime.now(timezone.utc).isoformat(),
        archive=list(args.archive), manifest=str(args.manifest.resolve()), source_system=str(source),
        arms=list(ARMS), fixed_updates_per_arm=UPDATES, total_fixed_updates=4*UPDATES,
        completed_arms=[], completed_optimizer_updates=0,
        optimizer_updates_by_arm={arm: 0 for arm in ARMS}, cpu_threads=args.cpu_threads,
        reference_arrays_read=False, reference_csv_read=False, test_opened=False,
        motion_observation_policy='canonical_all', validation_forward_passes_per_arm=1,
        score_semantics='FP64 OLS base plus FP32 residual, not calibrated probability')
    def check(**extra):
        state.update(extra)
        if 'active_arm_updates' in extra:
            state['optimizer_updates_by_arm'][state['active_arm']] = extra['active_arm_updates']
        state['completed_optimizer_updates'] = sum(state['optimizer_updates_by_arm'].values())
        state.update(elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)),
            advisory_exceeded=time.monotonic()-started > 7200)
        if state['peak_rss_bytes'] > RAM:
            raise MemoryError('8GiB resident budget exceeded; preserve attempt')
        if extra:
            runtime.write_json(output/'run.json', state)
        if state['advisory_exceeded'] and not (output/'time_advisory.json').exists():
            runtime.write_json(output/'time_advisory.json', dict(elapsed_seconds=state['elapsed_seconds'],
                action='continue only original fixed preparation/4000 updates/final evaluation; no additional training'))
    try:
        check(phase='preparing')
        torch.set_num_threads(args.cpu_threads)
        with MTADArchive(args.archive) as archive:
            validate_archive(archive, manifest)
            train = prepare_split(archive, manifest, 'T_core', output, source, check)
            validation = prepare_split(archive, manifest, 'V_select', output, source, check)
        timings = dict(raw_and_feature_preparation_seconds=time.monotonic()-started, per_arm={})
        if {s['scene_id'] for s in train} & {s['scene_id'] for s in validation}:
            raise ValueError('T/V scene overlap')
        check(phase='normalizer')
        phase_started = time.monotonic()
        norms = fit_normalizers(train, check)
        timings['normalization_seconds'] = time.monotonic()-phase_started
        runtime.write_json(output/'normalizers.json', norms)
        phase_started = time.monotonic()
        plan, exposure = build_plan(train)
        timings['shared_sampling_plan_seconds'] = time.monotonic()-phase_started
        np.save(output/'private'/'shared-batch-plan.npy', plan)
        runtime.write_json(output/'sampling-plan-audit.json', dict(exposure, actual_per_arm_steps=UPDATES,
            plan_is_shared_across_arms=True, positive_per_batch=128, negative_per_batch=128,
            note='planned exposure, actual successful count separately in run and checkpoints'))
        runtime.seed_everything(SEED); initial_model = ShapeProbe()
        initial = {k: v.detach().cpu().clone() for k, v in initial_model.state_dict().items()}
        parameters = sum(v.numel() for v in initial_model.parameters())
        if parameters != 6209:
            raise ValueError('matched probe parameter budget differs')
        torch.save(initial, output/'private'/'shared-initial.pt'); del initial_model
        device = runtime.device_from_string(args.device)
        state['provenance'] = runtime.provenance(device)
        audit = dict(train_scene_count=len(train), validation_scene_count=len(validation),
            train_scene_ids=[s['scene_id'] for s in train], validation_scene_ids=[s['scene_id'] for s in validation],
            train_pair_count=sum(s['pair_count'] for s in train), validation_pair_count=sum(s['pair_count'] for s in validation),
            train_positive_count=sum(s['positive_count'] for s in train), validation_positive_count=sum(s['positive_count'] for s in validation),
            full_candidate_population=True, source056_V_motion_exact=True, parameter_count_per_arm=parameters,
            cache_file_bytes=sum(s[key].stat().st_size for s in train+validation
                                 for key in ('features_path', 'base_path', 'pairs_path', 'labels_path')),
            reference_read=False, test_opened=False, cache_columns=46, feature_cache_dtype='float32', base_dtype='float64',
            permission='offline full canonical measurements; T/V scene split shares identities',
            preparation_seconds=time.monotonic()-started,
            training_scene_components_overlap_validation=bool({s['component_id'] for s in train}&{s['component_id'] for s in validation}))
        runtime.write_json(output/'data-audit.json', audit)
        records_by_arm, methods, thresholds = {}, {}, {}
        for arm in ARMS:
            check(phase='training', active_arm=arm, active_arm_updates=0)
            phase_started = time.monotonic()
            model = train_arm(train, plan, arm, norms[arm], initial, device, output/'private'/arm, check)
            timings['per_arm'][arm] = dict(training_seconds=time.monotonic()-phase_started)
            check(phase='final_validation', active_arm=arm)
            phase_started = time.monotonic()
            records = score(model, validation, arm, norms[arm], device, check)
            selection = select_threshold(records); threshold = selection['threshold']
            thresholds[arm] = selection
            result = describe(records, threshold)
            components = sorted({r['component_id'] for r in records})
            result['components'] = {c: describe([r for r in records if r['component_id'] == c], threshold) for c in components}
            largest = max(components, key=lambda c: sum(r['component_id'] == c for r in records))
            result['without_largest_component'] = describe([r for r in records if r['component_id'] != largest], threshold)
            for row in records:
                np.savez_compressed(output/'private'/arm/f'scores-{row["scene_id"]}.npz',
                    **{k: row[k] for k in ('scores', 'pairs', 'labels', 'node_ids', 'sources', 'identities',
                                           'intervals', 'joint_support_queries')})
            runtime.write_json(output/f'{arm}-result.json', result)
            methods[arm], records_by_arm[arm] = result, records
            timings['per_arm'][arm]['final_validation_selection_metrics_and_writing_seconds'] = time.monotonic()-phase_started
            state['completed_arms'].append(arm)
            check(phase='arm_complete', active_arm=arm)
            del model
            torch.cuda.empty_cache()
        pairwise = {}
        for base, candidate in combinations(ARMS, 2):
            report = compare_predictions(records_by_arm[base], records_by_arm[candidate],
                thresholds[base]['threshold'], thresholds[candidate]['threshold'])
            left = methods[base]['raw']['macro_f1']; right = methods[candidate]['raw']['macro_f1']
            report.update(macro_f1_difference=right-left, baseline_method=base, candidate_method=candidate)
            pairwise[candidate+'_vs_'+base] = report
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.1', methods=methods, threshold_selection=thresholds,
            pairwise=pairwise, data_audit=audit, timings=timings, parameter_count_per_arm=parameters,
            optimizer_updates_per_arm=UPDATES, total_optimizer_updates=4*UPDATES,
            final_only=True, intermediate_validation=False, reference_read=False, test_opened=False,
            limitations=['single seed/fixed4000 total updates; repeatedly developed V; shared T/V identities',
                'eight truncated distance DCT coefficients do not preserve all native timing/direction',
                'rank-key shuffle is a pseudorandom negative control, not an exact permutation test',
                'matched parameter count does not equal matched active functional capacity',
                'raw association only; no physical fusion improvement or credible peer claim'])
        check(phase='finalizing')
        result.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'],
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device))
        runtime.write_json(output/'summary.json', result)
        check(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
                     exit_code=1, error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started,
                     finished_at=datetime.now(timezone.utc).isoformat())
        runtime.write_json(output/'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', nargs='+', required=True)
    for name in ('manifest', 'source-system', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'signal{signum}')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException:
        traceback.print_exc(); return 1
    print(f'COMPLETED: {args.output}/summary.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
