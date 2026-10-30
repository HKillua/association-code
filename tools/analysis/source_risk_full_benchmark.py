"""147: frozen133 heads, original32 imported predictions and new268 inference.

Reads only declared V measurements, never raw reference CSV or T_test. All
prediction producers run before evaluation arrays are read. No fitting occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import resource
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.data.archive import MTADArchive
from association.data.legacy_dataset import validate_archive
from association.data.native import _record
from association.data.timed import prepare_timed_scene
from association.data.motion_evidence import DATA, scene_motion_evidence
from association.evaluation import source_risk as risk
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.query_contribution import query_contribution
from association.evaluation.tmtaff import aggregate_fusion_reports
from tools.analysis import source_risk_pilot as pilot

qc, old = pilot.qc, pilot.old
IDEA = 'IDEA-SOURCE-RISK-FULL-BENCHMARK-20261030'
CONTROLS = ('WS_MEAN', 'WS_ANCHOR', 'WS_NEAREST_SUPPORTED')
POLICIES = {'WS_SOURCE_EQUAL': 'equal', 'WS_SHARED_Q_SOFT': 'soft', 'WS_SHARED_Q_HARD': 'hard'}
METHODS = CONTROLS + tuple(POLICIES)
PRIMARY = ('WS_SHARED_Q_HARD_vs_WS_MEAN', 'WS_SHARED_Q_HARD_vs_WS_NEAREST_SUPPORTED')
VISIBLE = ('node_ids', 'sources', 'query_seconds', 'native_states', 'nearest_seconds', 'support')
RAM, ADDRESS_SPACE = 6*1024**3, 12*1024**3


def frozen_model(path):
    payload = pilot.load_npz(path)
    prefix = 'full__QUADRATIC__'
    result = {k[len(prefix):]: v for k, v in payload.items() if k.startswith(prefix)}
    result['head'] = str(result['head'].item())
    risk.predict_score(result, np.zeros((1, 10)))
    return result


def bind_lineage(paths, pilot_run, control_run):
    """133 records its056 input through127, rather than a source_system key."""
    native = Path(pilot_run['native_run']).resolve()
    native_run = old.frozen.read(native/'run.json')
    if native_run.get('status') != 'COMPLETED' or native_run.get('exit_code') != 0:
        raise ValueError('completed original127 required')
    for source in (native_run, control_run):
        for key in ('source_system', 'partition_run'):
            if Path(source[key]).resolve() != paths[key]:
                raise ValueError('source/partition lineage differs')
    if Path(pilot_run['partition_run']).resolve() != paths['partition_run']:
        raise ValueError('133 partition lineage differs')
    return native


def scene_motion(archive, manifest, sid, visible):
    native = _record(archive.load_scene(sid), manifest['components'][sid], 'V_select',
                     DATA['max_observations'], split_unit=manifest.get('split_unit'))
    prepared = prepare_timed_scene(native, **DATA, query_times=visible['query_seconds'])
    old._same(np.asarray(prepared['node_ids'], str), visible['node_ids'], 'raw canonical node order')
    result = scene_motion_evidence(prepared, motion_observation_policy='canonical_all')
    return dict(result, node_ids=np.asarray(prepared['node_ids'], str))


def predictions(motion, groups, model):
    first = np.asarray([g[0] for g in groups], np.int64)
    mean = ordinary_fusion(motion['states'], motion['nearest_seconds'], motion['support'], groups)[
        'predictions']['mean_supported'][first]
    result = dict(WS_MEAN=mean, WS_ANCHOR=motion['states'][first].copy(),
        WS_NEAREST_SUPPORTED=query_contribution(motion['states'], motion['nearest_seconds'],
            motion['support'], motion['sources'], groups, method='nearest_supported')['group_states'])
    pack = risk.build_source_features(**{k: motion[k] for k in pilot.MOTION_KEYS}, groups=groups)
    scores = risk.predict_score(model, pack['features'].reshape(-1, 10)).reshape(pack['available'].shape)
    return result, pack, scores


def evaluate_scene(motion, groups, pack, raw, fragment):
    """The sole truth boundary; input prediction arrays are already fixed."""
    target_ids = fragment['target_identity_ids']
    registry = {str(t): i for i, t in enumerate(target_ids)}
    truth = np.full((len(target_ids), 64, 4), np.nan)
    valid = np.zeros_like(truth, bool)
    # The original056 full registry includes targets without input nodes.
    # Read its original aligned reference instead of reconstructing from nodes.
    truth[:] = fragment['_truth']
    valid[:] = fragment['_valid']
    reports, artifacts = evaluate_fusion_methods(
        {n: emit_group_outputs(raw[n], groups, node_count=len(motion['states'])) for n in METHODS},
        prediction_speed_units={n: 'kn' for n in METHODS},
        node_identities=fragment['identities'], target_ids=target_ids, target_states=truth,
        target_valid=valid, support=motion['support'], nearest_seconds=motion['nearest_seconds'],
        target_speed_unit='mps')
    regions = old.fixed_regions(fragment, target_ids)
    counts = np.full((len(target_ids), 64), -1, np.int64)
    claimed = set()
    source_counts = np.asarray([pack['available'][a:b].sum(0)
        for a, b in zip(pack['group_offsets'][:-1], pack['group_offsets'][1:])])
    for gid, group in enumerate(groups):
        identity = str(fragment['identities'][group[0]])
        if identity in registry and identity not in claimed:
            counts[registry[identity]] = source_counts[gid]
            claimed.add(identity)
    regions.update(source_support_zero=counts == 0, source_support_one=counts == 1,
                   source_support_multi=counts >= 2)
    row = dict(methods={}, pairwise={})
    arrays = dict(methods={}, pairs={})
    slim = {}
    for name in METHODS:
        row['methods'][name], arrays['methods'][name] = qc.describe(reports[name], artifacts[name], fragment, regions)
        slim[name] = {k: reports[name][k] for k in ('full_target_metrics', 'paper_fusion_metrics')}
    for base, candidate in combinations(METHODS, 2):
        key = candidate+'_vs_'+base
        value, arr, _ = old.compare_pair(artifacts[base], artifacts[candidate], regions)
        value.update(baseline_method=base, candidate_method=candidate,
                     side_labels=dict(A0=base, candidate=candidate))
        delta = arr['all']['delta']
        value['actual_harm'] = dict(new_harm_over_1km=int((delta > 1000).sum()),
            max_error_increase_m=float(max(0., delta.max())) if len(delta) else None)
        row['pairwise'][key], arrays['pairs'][key] = value, arr
    return row, arrays, artifacts, slim


def aggregate(rows, arrays, physical, indices):
    indices = list(indices)
    if not indices:
        return None
    result = qc.aggregate([rows[i] for i in indices], [arrays[i] for i in indices], method_names=METHODS)
    result['physical'] = {n: aggregate_fusion_reports([physical[i][n] for i in indices]) for n in METHODS}
    for key in result['pairwise']:
        harms = [rows[i]['pairwise'][key]['actual_harm'] for i in indices]
        result['pairwise'][key]['actual_harm'] = dict(new_harm_over_1km=sum(h['new_harm_over_1km'] for h in harms),
            max_error_increase_m=max((h['max_error_increase_m'] or 0.) for h in harms))
    return result


def bootstrap(rows, arrays, indices):
    indices = list(indices)
    components = sorted({rows[i]['component_id'] for i in indices})
    result = {}
    for key in PRIMARY:
        sums, counts = [], []
        for component in components:
            delta = np.concatenate([arrays[i]['pairs'][key]['all']['delta'] for i in indices
                                    if rows[i]['component_id'] == component])
            sums.append(float(delta.sum())); counts.append(len(delta))
        sums, counts = np.asarray(sums), np.asarray(counts)
        draws = np.random.default_rng(20).integers(0, len(components), (2000, len(components)))
        denominator = counts[draws].sum(1)
        if (denominator <= 0).any():
            raise ValueError('bootstrap draw without evaluable points')
        delta = sums[draws].sum(1)/denominator
        result[key] = dict(point_estimate_m=float(sums.sum()/counts.sum()),
            percentile95_m=np.quantile(delta, [.025, .975]).tolist(), component_count=len(components),
            seed=20, resamples=2000, independent_test=False)
    return result


def run(args):
    paths = {key: Path(getattr(args, key)).resolve() for key in
             ('source_system', 'partition_run', 'pilot_run', 'control_run', 'manifest', 'output')}
    output = paths['output']
    if any(output == p or output in p.parents or p in output.parents for k, p in paths.items() if k != 'output'):
        raise ValueError('output overlaps immutable inputs')
    if not 1 <= args.cpu_threads <= 4 or args.time_policy != 'advisory':
        raise ValueError('CPU1..4 and advisory required')
    output.mkdir(parents=True, exist_ok=False)
    for directory in ('private', 'scenes', 'private/native-motion'):
        (output/directory).mkdir(exist_ok=True)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0', methods=list(METHODS),
        started_at=datetime.now(timezone.utc).isoformat(), completed_scenes=0,
        analytic_head_fits=0, optimizer_updates=0, neural_models_constructed=0,
        teacher_forward_calls=0, threshold_reselected=False, groups_recomputed=False,
        test_opened=False, raw_reference_read=False, raw_measurement_scenes=0,
        imported_source_prediction_arms=0, computed_source_prediction_arms=0,
        **{k: str(v) for k, v in paths.items() if k != 'output'})
    def check(**extra):
        state.update(extra, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes())
        old.frozen.write(output/'run.json', state)
        if state['peak_rss_bytes'] > RAM:
            raise MemoryError('6GiB resident budget exceeded')
        if state['elapsed_seconds'] > 3600 and not (output/'time_advisory.json').exists():
            old.frozen.write(output/'time_advisory.json', dict(action='continue original V300 only'))
    try:
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([ADDRESS_SPACE]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        check(phase='binding')
        manifest = old.frozen.read(paths['manifest'])
        meta = old.metadata(paths['source_system'], paths['partition_run'], manifest)
        pilot_run = old.frozen.read(paths['pilot_run']/'run.json')
        control_run = old.frozen.read(paths['control_run']/'run.json')
        if (pilot_run.get('status') != 'COMPLETED' or pilot_run.get('exit_code') != 0
                or pilot_run.get('idea_id') != pilot.IDEA or pilot_run.get('analytic_head_fits') != 39
                or control_run.get('status') != 'COMPLETED' or control_run.get('exit_code') != 0
                or control_run.get('completed_scenes') != 300
                or pilot_run['scene_ids'] != meta['ids'] or control_run['scene_ids'] != meta['all_ids']):
            raise ValueError('completed original133/117 and original V32/V300 required')
        native = bind_lineage(paths, pilot_run, control_run)
        original_ids = set(pilot_run['scene_ids'])
        model = frozen_model(paths['pilot_run']/'private/models.npz')
        check(scene_ids=meta['all_ids'], original32_ids=pilot_run['scene_ids'], phase='V_prediction_evaluation')
        rows, arrays, physical = [], [], []
        with MTADArchive(args.archive) as archive:
            validate_archive(archive, manifest)
            for sid in meta['all_ids']:
                first = sid in original_ids
                source_file = paths['source_system']/'predictions'/f'states-{sid}.npz'
                visible = pilot.load_npz(source_file, VISIBLE)
                membership = np.asarray(old.frozen.read(paths['partition_run']/'private'/f'scene-{sid}.json.gz')['memberships']['WS'], np.int64)
                if first:
                    motion = pilot.load_npz(native/'private/native-motion/V_select'/f'scene-{sid}.npz',
                                            pilot.MOTION_KEYS+('node_ids', 'query_seconds'))
                else:
                    motion = scene_motion(archive, manifest, sid, visible)
                    state['raw_measurement_scenes'] += 1
                for key, original_key in (('states', 'native_states'), ('support', 'support'),
                        ('nearest_seconds', 'nearest_seconds'), ('sources', 'sources'),
                        ('node_ids', 'node_ids'), ('query_seconds', 'query_seconds')):
                    old._same(motion[key], visible[original_key], '056 exact motion/'+key)
                groups = old.groups_from_membership(membership, len(motion['states']))
                raw, pack, scores = predictions(motion, groups, model)
                saved = {k: v for k, v in pack.items() if isinstance(v, np.ndarray)}
                saved.update(node_ids=motion['node_ids'], query_seconds=motion['query_seconds'],
                             membership=membership, scores=scores)
                if first:
                    previous = pilot.load_npz(paths['pilot_run']/'private'/f'policy-{sid}.npz')
                    for key in saved:
                        original_key = 'WS_SHARED_Q_HARD__scores' if key == 'scores' else key
                        old._same(saved[key], previous[original_key], '133 pack/score exact/'+key)
                    with np.load(paths['pilot_run']/'private'/f'states-{sid}.npz', allow_pickle=False) as z:
                        for name in POLICIES:
                            raw[name] = z[name+'_group_states'].copy()
                            saved[name+'__source_weights'] = previous[name+'__source_weights']
                        for name in ('WS_MEAN', 'WS_NEAREST_SUPPORTED'):
                            old._same(raw[name], z[name+'_group_states'], '133 exact ordinary control/'+name)
                    state['imported_source_prediction_arms'] += 3
                else:
                    for name, kind in POLICIES.items():
                        policy = risk.recover_source_policy(states=motion['states'], support=motion['support'],
                            groups=groups, pack=pack, scores=scores, kind=kind, mean_states=raw['WS_MEAN'])
                        raw[name] = policy['group_states']
                        saved[name+'__source_weights'] = policy['source_weights']
                    with np.load(paths['control_run']/'private'/f'states-{sid}.npz', allow_pickle=False) as z:
                        for name in CONTROLS:
                            old._same(raw[name], z[name+'_group_states'], '117 exact ordinary control/'+name)
                    state['computed_source_prediction_arms'] += 3
                np.savez_compressed(output/'private'/f'policy-{sid}.npz', **saved)
                np.savez_compressed(output/'private/native-motion'/f'scene-{sid}.npz',
                    **{k: motion[k] for k in pilot.MOTION_KEYS+('node_ids', 'query_seconds')})
                # All predictions are fixed before the original reference is opened.
                fragment = pilot.load_npz(source_file)
                with np.load(paths['source_system']/'system-predictions'/f'states-{sid}.npz', allow_pickle=False) as z:
                    t = len(fragment['target_identity_ids'])
                    old._same(z['learned_assoc_mean_supported_target_ids'], fragment['target_identity_ids'], 'full target registry')
                    fragment['_truth'] = z['learned_assoc_mean_supported_aligned_truth'][:t]
                    fragment['_valid'] = z['learned_assoc_mean_supported_aligned_truth_valid'][:t]
                row, arr, artifacts, physical_row = evaluate_scene(motion, groups, pack, raw, fragment)
                row.update(scene_id=sid, component_id=str(manifest['components'][sid]),
                    node_count=len(motion['states']), stratum='original32' if first else 'remaining268',
                    fixed_R_population_invariants_verified=True, feature_audit=pack['audit'])
                np.savez_compressed(output/'private'/f'states-{sid}.npz',
                    **{name+'_'+key: value for name, artifact in artifacts.items() for key, value in artifact.items()})
                old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                rows.append(row); arrays.append(arr); physical.append(physical_row)
                archive._label_cache.pop(sid, None)
                check(completed_scenes=len(rows), last_scene_id=sid)
        cs = sorted({row['component_id'] for row in rows})
        largest = max(cs, key=lambda c: sum(row['component_id'] == c for row in rows))
        original_components = {r['component_id'] for r in rows if r['stratum'] == 'original32'}
        def subset(indices):
            return aggregate(rows, arrays, physical, indices)
        original_indices = [i for i, row in enumerate(rows) if row['stratum'] == 'original32']
        remaining = [i for i, row in enumerate(rows) if row['stratum'] == 'remaining268']
        summary = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0',
            all_components=subset(range(len(rows))), original32=subset(original_indices), remaining268=subset(remaining),
            by_component={c: subset(i for i, r in enumerate(rows) if r['component_id'] == c) for c in cs},
            without_largest_component=subset(i for i, r in enumerate(rows) if r['component_id'] != largest),
            remaining268_unseen_components=subset(i for i in remaining if rows[i]['component_id'] not in original_components),
            primary_bootstrap=bootstrap(rows, arrays, range(len(rows))),
            remaining268_bootstrap=bootstrap(rows, arrays, remaining),
            component_count=len(cs), largest_component_id=largest,
            analytic_head_fits=0, neural_models_constructed=0, optimizer_updates=0,
            imported_source_prediction_arms=state['imported_source_prediction_arms'],
            computed_source_prediction_arms=state['computed_source_prediction_arms'],
            speed_output_origin='Q policies inherit Mean; kn converted to mps only at evaluation boundary',
            limitations=['V300 repeatedly developed and shares identities; no independent generalization',
                'Actual sources K2; no K>=3, online, end-to-end or credible peer superiority claim',
                'Fixed original T32 parameter fit; hard chosen after original V32 development',
                'Source scores predict supervised log error, not calibrated physical trust/covariance'])
        old.frozen.write(output/'scene-summary.json.gz', rows)
        check(phase='finalizing')
        summary.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'])
        old.frozen.write(output/'summary.json', summary)
        check(status='COMPLETED', exit_code=0, phase='completed', finished_at=datetime.now(timezone.utc).isoformat())
        return summary
    except BaseException as error:
        check(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}',
              finished_at=datetime.now(timezone.utc).isoformat())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-system', 'partition-run', 'pilot-run', 'control-run', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--archive', nargs='+', required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(InterruptedError(f'signal {signum}')))
    run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
