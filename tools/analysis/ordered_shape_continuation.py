"""132: reuse immutable126 caches/norms; bound sampling-input file handles.

Only the sampling input transport changes: ordinary ndarray loads release each
file immediately. The original sampler, model, batches and endpoint are reused.
No archive/reference reader, source-run writer or automatic retry is called.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
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
from tools.analysis import ordered_shape_signal as p

IDEA = 'IDEA-ORDERED-SHAPE-CONTINUATION-20261029'
META_KEYS = ('node_ids', 'sources', 'identities', 'intervals', 'joint_support_queries')


def verify_failed_source(path, source, manifest):
    source = Path(source).resolve()
    run = p.runtime.read_json(path/'run.json')
    if (run.get('status') != 'FAILED' or run.get('exit_code') != 1 or run.get('idea_id') != p.IDEA
            or run.get('plan_version') != '1.1' or run.get('phase') != 'normalizer'
            or run.get('completed_arms') != [] or run.get('completed_optimizer_updates') != 0
            or run.get('optimizer_updates_by_arm') != {arm: 0 for arm in p.ARMS}
            or run.get('arms') != list(p.ARMS) or run.get('fixed_updates_per_arm') != 1000
            or run.get('total_fixed_updates') != 4000 or run.get('motion_observation_policy') != 'canonical_all'
            or Path(run['source_system']).resolve() != source
            or p.runtime.read_json(Path(run['manifest'])) != manifest
            or run.get('reference_arrays_read') is not False or run.get('reference_csv_read') is not False
            or run.get('test_opened') is not False or 'Too many open files' not in run.get('error', '')):
        raise ValueError('132 only resumes the original zero-update126 descriptor failure')
    if any((path/'private'/name).exists() for name in ('shared-batch-plan.npy', 'shared-initial.pt', *p.ARMS)):
        raise ValueError('failed126 must have no prior plan, initial or arm training products')
    return run


def npy_header(path, shape, dtype):
    """Check the complete declared file size without scanning feature payload."""
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        if value.shape != shape or value.dtype != np.dtype(dtype) or not value.flags.c_contiguous:
            raise ValueError('cached NPY shape/dtype/order differs: '+path.name)
        if path.stat().st_size != value.offset+value.size*value.dtype.itemsize:
            raise ValueError('cached NPY payload length differs: '+path.name)
    finally:
        p.close_map(value)


def verify_candidates(scene, pairs, labels):
    n = len(scene['node_ids'])
    if pairs.dtype != np.int64 or not np.array_equal(pairs, np.column_stack(np.triu_indices(n, 1))):
        raise ValueError('cached complete original i<j pairs differ')
    if (labels.shape != (len(pairs),) or labels.dtype != bool
            or not np.array_equal(labels, scene['identities'][pairs[:, 0]] == scene['identities'][pairs[:, 1]])):
        raise ValueError('cached identity/label pair binding differs')


def verify_v_metadata(scene, source):
    """Small056 fields only; no targets, reference or native-state payload."""
    with np.load(source/'predictions'/f'states-{scene["scene_id"]}.npz', allow_pickle=False) as saved:
        for key in ('node_ids', 'sources', 'identities'):
            p._same(scene[key], saved[key], '056 cached '+key)
        query, support = saved['query_seconds'], saved['support']
    n = len(scene['node_ids'])
    if query.shape != (64,) or not np.isfinite(query).all() or (np.diff(query) <= 0).any():
        raise ValueError('056 declared Q64 query grid differs')
    if support.shape != (n, 64) or support.dtype != bool:
        raise ValueError('056 input support shape/dtype differs')
    if n:
        bounds = scene['intervals']
        p._same(np.linspace(bounds[:, 0].min(), bounds[:, 1].max(), 64), query, 'interval-derived original Q64')
        expected = (query[None] >= bounds[:, :1]) & (query[None] <= bounds[:, 1:])
        p._same(expected, support, 'interval-derived original support')
    pairs = np.load(scene['pairs_path'], allow_pickle=False); labels = np.load(scene['labels_path'], allow_pickle=False)
    verify_candidates(scene, pairs, labels)
    expected = np.empty(len(pairs), np.uint8)
    for start in range(0, len(pairs), p.CHUNK):
        block = pairs[start:start+p.CHUNK]
        expected[start:start+len(block)] = (support[block[:, 0]] & support[block[:, 1]]).sum(1)
    p._same(scene['joint_support_queries'], expected, '056 joint input support counts')
    scene['positive_count'] = int(labels.sum())


def load_cache(path, manifest, source, check=lambda **_: None):
    """One bounded header/metadata inventory; large46-column values not read."""
    result = {}; file_bytes = metadata_bytes = empty_v = 0
    for split in ('T_core', 'V_select'):
        directory = path/'private/cache'/split
        expected = {sid+'-'+suffix for sid in manifest['splits'][split]
                    for suffix in ('features.npy', 'base.npy', 'pairs.npy', 'labels.npy', 'metadata.npz')}
        if {entry.name for entry in directory.iterdir()} != expected:
            raise ValueError('complete exact five-file-per-scene cache inventory required')
        scenes = []
        for sid in manifest['splits'][split]:
            with np.load(directory/f'{sid}-metadata.npz', allow_pickle=False) as saved:
                if set(saved.files) != set(META_KEYS):
                    raise ValueError('original126 metadata schema differs')
                meta = {key: saved[key] for key in META_KEYS}
            n = len(meta['node_ids']); e = n*(n-1)//2
            if (any(meta[k].shape != (n,) or meta[k].dtype.kind not in 'US' for k in ('node_ids', 'sources', 'identities'))
                    or len(set(meta['node_ids'].tolist())) != n or meta['intervals'].shape != (n, 2)
                    or meta['intervals'].dtype != np.float64 or not np.isfinite(meta['intervals']).all()
                    or (meta['intervals'][:, 1] <= meta['intervals'][:, 0]).any()
                    or meta['joint_support_queries'].shape != (e,) or meta['joint_support_queries'].dtype != np.uint8
                    or (meta['joint_support_queries'] > 64).any()):
                raise ValueError('cached complete node/native interval/support metadata differs')
            scene = dict(meta, scene_id=sid, split=split, component_id=str(manifest['components'][sid]), pair_count=e)
            for key, shape, dtype in (('features', (e, 46), np.float32), ('base', (e,), np.float64),
                                      ('pairs', (e, 2), np.int64), ('labels', (e,), bool)):
                cache_path = directory/f'{sid}-{key}.npy'; npy_header(cache_path, shape, dtype)
                scene[key+'_path'] = cache_path; file_bytes += cache_path.stat().st_size
            metadata_bytes += (directory/f'{sid}-metadata.npz').stat().st_size
            if split == 'V_select':
                verify_v_metadata(scene, source); empty_v += int(n == 0)
            scenes.append(scene)
            check(phase='cache_inventory', cache_inventory_scenes=sum(len(v) for v in result.values())+len(scenes), last_scene_id=sid)
        result[split] = scenes
    return result, dict(cache_file_bytes=file_bytes, cache_metadata_bytes=metadata_bytes,
        cache_scene_count=sum(len(v) for v in result.values()), cache_file_count=5*sum(len(v) for v in result.values()),
        feature_payload_rescanned=False, source056_V_small_metadata_rebound=True,
        empty_V_grid_inherited_from056_count=empty_v,
        limitation='query/support not serialized directly in126 cache; nonempty values reconstructed from intervals and checked against056; feature payload trusted from completed126 preparation')


def restore_normalizers(path, train_pairs):
    norms = p.runtime.read_json(path/'normalizers.json')
    if set(norms) != set(p.ARMS): raise ValueError('complete four original126 normalizers required')
    for arm, norm in norms.items():
        mean, std = np.asarray(norm['mean']), np.asarray(norm['std'])
        if (mean.shape != (30,) or std.shape != (30,) or not np.isfinite(mean).all() or not np.isfinite(std).all()
                or (std <= 0).any() or norm.get('fitted_split') != 'T_core' or norm.get('pair_count') != train_pairs
                or norm.get('old_feature_names') != list(p.FEATURE_NAMES) or norm.get('extra_feature_rule') != arm
                or norm.get('cache_dtype') != 'float32' or norm.get('accumulation_dtype') != 'float64'
                or norm.get('weighting') != 'one weight per original T pair; 4096-row Welford'):
            raise ValueError('frozen126 T-only normalizer schema/finite values differ')
        for key in ('mean', 'std'): p._same(norm[key][:22], norms['SUMMARY'][key][:22], 'original shared22 '+key)
    p._same(norms['SUMMARY']['mean'][22:], np.zeros(8), 'original SUMMARY zero extra mean')
    p._same(norms['SUMMARY']['std'][22:], np.ones(8), 'original SUMMARY unit extra scale')
    return norms


def build_plan(scenes, steps=p.UPDATES, check=lambda **_: None):
    """Same sampler/seed/order; ordinary arrays never retain per-file FDs."""
    inputs = []
    for scene in scenes:
        pairs = np.load(scene['pairs_path'], allow_pickle=False)
        labels = np.load(scene['labels_path'], allow_pickle=False)
        verify_candidates(scene, pairs, labels)
        if isinstance(pairs, np.memmap) or isinstance(labels, np.memmap):
            raise ValueError('132 sampling inputs must own ordinary memory')
        scene['positive_count'] = int(labels.sum())
        inputs.append(dict(scene_id=scene['scene_id'], pairs=pairs, labels=labels))
        check(phase='sampling_inputs', sampling_input_scenes=len(inputs))
    # BalancedScenePairSampler copies labels and class pools, not the pair data.
    sampler = p.BalancedScenePairSampler(inputs, p.SEED)
    del inputs, pairs, labels
    plan = np.empty((steps, p.BATCH, 2), np.int64)
    for step in range(steps):
        groups = sampler.sample(p.BATCH)
        plan[step, :, 0] = np.concatenate([np.full(len(indices), sid, np.int64) for sid, indices in groups])
        plan[step, :, 1] = np.concatenate([indices for _, indices in groups])
        sampler.record_completed_batch(groups)
    return plan, sampler.coverage()


def run(args):
    output, failed, source = (Path(getattr(args, k)).resolve() for k in ('output', 'failed_run', 'source_system'))
    for protected in (failed, source.parent):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('new132 output must not overlap frozen126/056')
    if not 1 <= args.cpu_threads <= 4 or not str(args.device).startswith('cuda:'):
        raise ValueError('one explicit CUDA device and1..4 CPU threads required')
    manifest = p.runtime.read_json(args.manifest); p.validate_manifest(manifest)
    if manifest.get('dataset_level') != 'Level1' or manifest.get('split_unit') != 'scene' or len(manifest['splits']['T_core']) != 2100 or len(manifest['splits']['V_select']) != 300:
        raise ValueError('same formal Level1 T2100/V300 required')
    original = verify_failed_source(failed, source, manifest)
    output.mkdir(parents=True, exist_ok=False); (output/'private').mkdir()
    (output/'private/cache').symlink_to(failed/'private/cache', target_is_directory=True)
    start = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0', scientific_idea_id=p.IDEA, scientific_plan_version='1.1',
        started_at=datetime.now(timezone.utc).isoformat(), failed_run=str(failed), source_system=str(source), manifest=str(Path(args.manifest).resolve()),
        arms=list(p.ARMS), fixed_updates_per_arm=1000, total_fixed_updates=4000, completed_arms=[], completed_optimizer_updates=0,
        optimizer_updates_by_arm={arm: 0 for arm in p.ARMS}, cpu_threads=args.cpu_threads,
        reference_arrays_read=False, reference_csv_read=False, test_opened=False, archive_read=False,
        motion_observation_policy='canonical_all', validation_forward_passes_per_arm=1, original126_restarted=False,
        raw_cache_rebuilt=False, normalizers_refitted=False, cache_read_only_reference=True,
        sampling_input_transport='ordinary ndarray; original BalancedScenePairSampler',
        prior_failed_elapsed_seconds=original['elapsed_seconds'], prior_failed_optimizer_updates=0,
        score_semantics='FP64 OLS base plus FP32 residual, not calibrated probability')
    def check(**extra):
        state.update(extra)
        if 'active_arm_updates' in extra: state['optimizer_updates_by_arm'][state['active_arm']] = extra['active_arm_updates']
        state['completed_optimizer_updates'] = sum(state['optimizer_updates_by_arm'].values())
        state.update(elapsed_seconds=time.monotonic()-start, peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)),
            advisory_exceeded=time.monotonic()-start > 7200)
        p.runtime.write_json(output/'run.json', state)
        if state['peak_rss_bytes'] > p.RAM: raise MemoryError('original8GiB resident budget exceeded')
        if state['advisory_exceeded'] and not (output/'time_advisory.json').exists():
            p.runtime.write_json(output/'time_advisory.json', dict(elapsed_seconds=state['elapsed_seconds'], action='continue only original4000 updates/finalV; no additional fit/retry'))
    try:
        torch.set_num_threads(args.cpu_threads); check(phase='cache_inventory')
        cached, inventory = load_cache(failed, manifest, source, check)
        train, validation = cached['T_core'], cached['V_select']
        norms = restore_normalizers(failed, sum(s['pair_count'] for s in train))
        p.runtime.write_json(output/'normalizers.json', norms)
        timings = dict(cache_inventory_seconds=time.monotonic()-start, prior_failed_elapsed_seconds=original['elapsed_seconds'], per_arm={})
        check(phase='sampling_plan'); before = time.monotonic(); plan, exposure = build_plan(train, check=check)
        timings['shared_sampling_plan_seconds'] = time.monotonic()-before
        np.save(output/'private/shared-batch-plan.npy', plan)
        p.runtime.write_json(output/'sampling-plan-audit.json', dict(exposure, actual_per_arm_steps=1000,
            plan_is_shared_across_arms=True, positive_per_batch=128, negative_per_batch=128,
            note='planned exposure, actual successful count separately in run and checkpoints'))
        p.runtime.seed_everything(p.SEED); initial_model = p.ShapeProbe()
        initial = {k: v.detach().cpu().clone() for k,v in initial_model.state_dict().items()}
        parameters = sum(t.numel() for t in initial_model.parameters()); del initial_model
        if parameters != 6209: raise ValueError('original parameter budget differs')
        torch.save(initial, output/'private/shared-initial.pt')
        device = p.runtime.device_from_string(args.device); state['provenance'] = p.runtime.provenance(device)
        data = dict(train_scene_count=len(train), validation_scene_count=len(validation), train_scene_ids=[s['scene_id'] for s in train],
            validation_scene_ids=[s['scene_id'] for s in validation], train_pair_count=sum(s['pair_count'] for s in train),
            validation_pair_count=sum(s['pair_count'] for s in validation), train_positive_count=sum(s['positive_count'] for s in train),
            validation_positive_count=sum(s['positive_count'] for s in validation), full_candidate_population=True,
            source056_V_motion_exact=True, source056_V_motion_verification='inherited126 canonical preparation; small cache metadata independently rebound132',
            parameter_count_per_arm=6209, **inventory, reference_read=False, test_opened=False, cache_columns=46,
            feature_cache_dtype='float32', base_dtype='float64', permission='offline full canonical measurements; T/V scene split shares identities',
            training_scene_components_overlap_validation=bool({s['component_id'] for s in train}&{s['component_id'] for s in validation}))
        p.runtime.write_json(output/'data-audit.json', data)
        records_by_arm, methods, thresholds = {}, {}, {}
        for arm in p.ARMS:
            check(phase='training', active_arm=arm, active_arm_updates=0); before=time.monotonic()
            model=p.train_arm(train,plan,arm,norms[arm],initial,device,output/'private'/arm,check)
            timings['per_arm'][arm]=dict(training_seconds=time.monotonic()-before)
            check(phase='final_validation',active_arm=arm); before=time.monotonic()
            records=p.score(model,validation,arm,norms[arm],device,check)
            selection=p.select_threshold(records);theta=selection['threshold'];thresholds[arm]=selection
            result=p.describe(records,theta);components=sorted({r['component_id'] for r in records})
            result['components']={c:p.describe([r for r in records if r['component_id']==c],theta) for c in components}
            largest=max(components,key=lambda c:sum(r['component_id']==c for r in records))
            result['without_largest_component']=p.describe([r for r in records if r['component_id']!=largest],theta)
            for row in records:
                np.savez_compressed(output/'private'/arm/f'scores-{row["scene_id"]}.npz', **{k:row[k] for k in
                    ('scores','pairs','labels','node_ids','sources','identities','intervals','joint_support_queries')})
            p.runtime.write_json(output/f'{arm}-result.json',result);methods[arm]=result;records_by_arm[arm]=records
            timings['per_arm'][arm]['final_validation_selection_metrics_and_writing_seconds']=time.monotonic()-before
            state['completed_arms'].append(arm);check(phase='arm_complete',active_arm=arm);del model;torch.cuda.empty_cache()
        pairwise={}
        for base,candidate in combinations(p.ARMS,2):
            value=p.compare_predictions(records_by_arm[base],records_by_arm[candidate],thresholds[base]['threshold'],thresholds[candidate]['threshold'])
            value.update(macro_f1_difference=methods[candidate]['raw']['macro_f1']-methods[base]['raw']['macro_f1'],baseline_method=base,candidate_method=candidate)
            pairwise[candidate+'_vs_'+base]=value
        check(phase='finalizing')
        summary=dict(status='COMPLETED',idea_id=IDEA,plan_version='1.0',scientific_idea_id=p.IDEA,scientific_plan_version='1.1',
            failed_run=str(failed),methods=methods,threshold_selection=thresholds,pairwise=pairwise,data_audit=data,timings=timings,
            parameter_count_per_arm=6209,optimizer_updates_per_arm=1000,total_optimizer_updates=4000,final_only=True,
            intermediate_validation=False,reference_read=False,test_opened=False,original126_restarted=False,normalizers_refitted=False,
            elapsed_seconds=state['elapsed_seconds'],prior_failed_elapsed_seconds=original['elapsed_seconds'],
            total_prior_plus_current_elapsed_seconds=original['elapsed_seconds']+state['elapsed_seconds'],peak_rss_bytes=state['peak_rss_bytes'],
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
            limitations=['126 descriptor failure at0 updates is preserved;132 changes only sampling transport and reuses its cache/norms',
                'cache header/metadata checked; large feature payload not independently regenerated or rescanned',
                'single seed/repeated-development V/shared identities; no physical fusion/online/credible peer claim',
                'matched parameter counts do not guarantee matched active capacity; truncated DCT and one rank-key shuffle retain original limitations'])
        p.runtime.write_json(output/'summary.json',summary)
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat());return summary
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'FAILED',exit_code=1,
            error=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-start,finished_at=datetime.now(timezone.utc).isoformat())
        p.runtime.write_json(output/'run.json',state);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('failed-run','manifest','source-system','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--device',required=True);parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--time-policy',choices=('advisory',),default='advisory');args=parser.parse_args()
    def interrupted(signum,frame):raise InterruptedError(f'signal{signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:run(args)
    except BaseException:traceback.print_exc();return 1
    print(f'COMPLETED: {args.output}/summary.json',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
