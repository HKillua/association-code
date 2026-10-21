"""Frozen 045 association / 046 fusion connection diagnostic, every fragment anchor.

Saved 046 membership is authoritative. Reference arrays are evaluation-only;
neither grouping nor forward functions accepts them. The original fragment pilot
and an additional one-group-one-output target view are reported separately.
No training, threshold search, or test access.
"""
from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import torch

from association.evaluation.fusion_controls import ordinary_fusion, metrics, attribute_errors
from association.evaluation.grouping import group_predicted_tracks
from association.evaluation.frozen_group_view import unique_target_view
from association.evaluation.physical import evaluate_aligned_fusion
from association.evaluation.tmtaff import aggregate_fusion_reports
from association.models.evidence_fusion import fusion_inputs
from association.utils.runtime import write_json

ATTRIBUTES = ('position', 'speed', 'heading')
GROUPS = ('ols046', 'assoc045')
ARMS = tuple(group+'_'+method for group in GROUPS for method in ('mean_supported', 'learned'))
COMPARISONS = dict(association_mean=(ARMS[2], ARMS[0]),
                   association_learned=(ARMS[3], ARMS[1]),
                   fusion_ols=(ARMS[1], ARMS[0]), fusion_045=(ARMS[3], ARMS[2]))
REPLAY_TOLERANCE = dict(anchor_weight_atol=1e-6, anchor_weight_rtol=1e-5,
    state_span_factor=1e-5, position_m=0.25, speed=0.001, heading_deg=0.001)


def read_json(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as handle:
        return json.load(handle)


def read_saved_predictions(path):
    """Read files or tar members without extracting filesystem paths."""
    result = {}
    def add(name, source):
        sid = Path(name).stem.removeprefix('states-')
        if sid in result:
            raise ValueError('duplicate saved scene '+sid)
        with np.load(source, allow_pickle=False) as z:
            result[sid] = {key: z[key].copy() for key in z.files}
    path = Path(path)
    if path.is_dir():
        for member in sorted(path.glob('states-*.npz')):
            add(member.name, member)
    else:
        with tarfile.open(path) as archive:
            for member in archive:
                if member.isfile() and Path(member.name).name.startswith('states-') and member.name.endswith('.npz'):
                    add(member.name, io.BytesIO(archive.extractfile(member).read()))
    return result


def read_scores(path):
    with np.load(path, allow_pickle=False) as z:
        ids = z['scene_ids'].tolist()
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate score scenes')
        # Deliberately do not load labels: they cannot affect predicted groups.
        return {sid: {key: z[sid+'_'+key].copy() for key in ('pairs', 'scores', 'node_ids', 'sources')}
                for sid in ids}


def saved_groups(membership):
    """Convert exact 046 membership to its original input-order partition."""
    membership = np.asarray(membership)
    if membership.ndim != 1 or membership.dtype.kind not in 'iu':
        raise ValueError('saved membership must be a one-dimensional integer vector')
    ids = list(dict.fromkeys(membership.tolist()))
    if ids != list(range(len(ids))):
        raise ValueError('saved membership is not in original group order')
    return [np.flatnonzero(membership == gid).tolist() for gid in ids]


def aligned_motion(scene, saved, scores):
    """Rebuild evidence from V measurements, preserving original physical states.

    046 did not save all native reliability features, so they must be rebuilt.
    States/support/clock use the actual saved arrays, not newly grouped scores.
    """
    if scene['split'] != 'V_select':
        raise ValueError('only V_select is permitted')
    for key in ('node_ids', 'sources'):
        if not np.array_equal(scene[key], saved[key]) or not np.array_equal(scene[key], scores[key]):
            raise ValueError('input order mismatch: '+key)
    if not np.array_equal(scene['identities'], saved['identities']):
        raise ValueError('evaluation identity order mismatch')
    motion = dict(scene['motion'])
    for key, old in (('query_seconds', 'query_seconds'), ('support', 'region_inside_support'),
                     ('exact_observation', 'region_exact_observation')):
        if not np.array_equal(motion[key], saved[old]):
            raise ValueError('visible motion metadata mismatch: '+key)
    for key, old in (('states', 'native_states'), ('nearest_seconds', 'nearest_seconds')):
        if not np.allclose(motion[key], saved[old], rtol=0., atol=1e-10):
            raise ValueError('reconstructed native evidence differs: '+key)
        motion[key] = saved[old].copy()
    n, q, width = motion['states'].shape
    if (q != 64 or width != 4 or saved['target'].shape != (n, q, 4)
            or saved['target_valid'].shape != (n, q, 4) or saved['target_valid'].dtype != bool
            or not np.array_equal(scores['pairs'], np.column_stack(np.triu_indices(n, 1)))):
        raise ValueError('complete original nodes/pairs/Q64/target mask required')
    return motion


@torch.inference_mode()
def predict_partition(model, motion, groups, device):
    """No target, identity, label or association score is accepted by forward."""
    controls = ordinary_fusion(motion['states'], motion['nearest_seconds'], motion['support'], groups)
    prediction = motion['states'].copy()
    n, q = prediction.shape[:2]
    own = np.ones((n, q, 3))
    anchors, helpers, utilities, weights = [], [], [], []
    for group in groups:
        for anchor in group:
            members = [anchor]+[node for node in group if node != anchor]
            output = model(*fusion_inputs(motion, members, device=device))
            prediction[anchor] = output['states'].cpu().numpy()
            own[anchor] = output['weights'][0].cpu().numpy()
            for i, helper in enumerate(members[1:]):
                anchors.append(anchor)
                helpers.append(helper)
                utilities.append(output['utility'][i].cpu().numpy())
                weights.append(output['weights'][i+1].cpu().numpy())
    routing = dict(helper_anchor=np.asarray(anchors, np.int64), helper_node=np.asarray(helpers, np.int64),
        utility=np.asarray(utilities, np.float32).reshape(-1, q, 3),
        helper_weights=np.asarray(weights, np.float64).reshape(-1, q, 3), anchor_weights=own,
        membership=controls['membership'])
    return dict(mean_supported=controls['predictions']['mean_supported'], learned=prediction), routing


def replay_check(saved, predictions, routing):
    """Fixed, bounded FP32 CPU/GPU tolerance; never widen it after observing V."""
    prefix = 'predicted_anchor_direct_'
    state, old = predictions['learned'], saved[prefix+'learned']
    delta = state-old
    for col in (1, 3):
        delta[..., col] = (delta[..., col]+180.) % 360.-180.
    physical = attribute_errors(state, old)
    maxima = {name: float(np.max(value, initial=0.)) for name, value in physical.items()}
    span = np.zeros_like(state)
    anchors, helpers = routing['helper_anchor'], routing['helper_node']
    if len(anchors):
        diff = saved['native_states'][helpers]-saved['native_states'][anchors]
        for col in (1, 3):
            diff[..., col] = (diff[..., col]+180.) % 360.-180.
        np.maximum.at(span, anchors, abs(diff))
    rounding = 32*np.finfo(np.float64).eps*np.maximum(abs(old), 1.)
    tolerance = REPLAY_TOLERANCE
    own_delta = abs(routing['anchor_weights']-saved[prefix+'anchor_weights'])
    checks = dict(
        membership_exact=np.array_equal(routing['membership'], saved[prefix+'membership']),
        ordinary_control_exact=np.array_equal(predictions['mean_supported'], saved[prefix+'mean_supported']),
        anchor_weights_close=np.allclose(routing['anchor_weights'], saved[prefix+'anchor_weights'],
            atol=tolerance['anchor_weight_atol'], rtol=tolerance['anchor_weight_rtol']),
        state_span_close=bool(np.all(abs(delta) <= tolerance['state_span_factor']*span+rounding)),
        physical_close=all(maxima[key] <= tolerance[limit]
                           for key, limit in (('position', 'position_m'), ('speed', 'speed'), ('heading', 'heading_deg'))))
    return dict(passed=bool(all(checks.values())), checks={k: bool(v) for k, v in checks.items()},
        max_physical_prediction_difference=maxima, max_anchor_weight_difference=float(own_delta.max(initial=0.)),
        tolerance=tolerance, rationale='Fixed FP32 cross-device rounding allowance plus absolute physical caps; '
        'not a theoretical error bound and never adjusted after replay.')


def full_metrics(prediction, target, valid, region):
    result = metrics(prediction, target, valid, region)
    errors = attribute_errors(prediction, target)
    for name, columns in (('position', [0, 1]), ('speed', [2]), ('heading', [3])):
        values = errors[name][valid[..., columns].all(-1) & region]
        result[name].update(p99=float(np.percentile(values, 99)) if len(values) else None,
                            max=float(values.max()) if len(values) else None)
    return result


def summarize(rows, region_name='all', source=None):
    target = np.concatenate([r['saved']['target'] for r in rows])
    valid = np.concatenate([r['saved']['target_valid'] for r in rows])
    region = np.concatenate([r['regions'][region_name] for r in rows]).copy()
    if source is not None:
        sources = np.concatenate([r['saved']['sources'] for r in rows])
        region &= sources[:, None] == source
    outputs = {arm: np.concatenate([r['predictions'][arm] for r in rows]) for arm in ARMS}
    methods = {arm: full_metrics(value, target, valid, region) for arm, value in outputs.items()}
    errors = {arm: attribute_errors(value, target) for arm, value in outputs.items()}
    comparisons = {}
    for name, (candidate, control) in COMPARISONS.items():
        comparisons[name] = {}
        for attribute, columns in (('position', [0, 1]), ('speed', [2]), ('heading', [3])):
            mask = valid[..., columns].all(-1) & region
            delta = (errors[candidate][attribute]-errors[control][attribute])[mask]
            a, b = methods[candidate][attribute], methods[control][attribute]
            comparisons[name][attribute] = dict(n=len(delta),
                **{'delta_'+metric: a[metric]-b[metric] if a[metric] is not None and b[metric] is not None else None
                   for metric in ('mae', 'rmse', 'p95', 'p99', 'max')},
                improved=int((delta < -1e-8).sum()), worsened=int((delta > 1e-8).sum()),
                unchanged=int((abs(delta) <= 1e-8).sum()))
    return dict(methods=methods, paired=comparisons)


def routing_diagnostics(saved, routing):
    """Truth enters only AFTER prediction, always the anchor's target/mask."""
    a, h = routing['helper_anchor'], routing['helper_node']
    result = {}
    for column, (attribute, target_columns, scale) in enumerate((
            ('position', [0, 1], 1000.), ('speed', [2], 10.), ('heading', [3], 30.))):
        anchor_error = attribute_errors(saved['native_states'][a], saved['target'][a])[attribute]
        helper_error = attribute_errors(saved['native_states'][h], saved['target'][a])[attribute]
        mask = saved['target_valid'][a][..., target_columns].all(-1)
        gain = np.log1p(anchor_error/scale)-np.log1p(helper_error/scale)
        u, w = routing['utility'][..., column], routing['helper_weights'][..., column]
        helpful, accepted = gain > 0, w > 0
        correct = np.broadcast_to((saved['identities'][a] == saved['identities'][h])[:, None], mask.shape)
        opportunities = np.zeros(saved['target'].shape[:2], bool)
        retained = np.zeros_like(opportunities)
        np.logical_or.at(opportunities, a, helpful & mask)
        np.logical_or.at(retained, a, helpful & mask & accepted)
        result[attribute] = dict(valid_helper_queries=int(mask.sum()),
            helpful_queries=int((mask & helpful).sum()), helpful_rejected=int((mask & helpful & ~accepted).sum()),
            nonhelpful_accepted=int((mask & ~helpful & accepted).sum()),
            correct_identity_helpful=int((mask & correct & helpful).sum()),
            wrong_identity_helpful=int((mask & ~correct & helpful).sum()),
            beneficial_anchor_queries=int(opportunities.sum()),
            all_beneficial_helpers_rejected=int((opportunities & ~retained).sum()),
            utility_squared_error_sum=float(((u-gain)[mask]**2).sum()),
            helpful_weight_sum=float(w[mask & helpful].sum()),
            nonhelpful_weight_sum=float(w[mask & ~helpful].sum()),
            opportunity_anchor_weight_sum=float(routing['anchor_weights'][..., column][opportunities].sum()))
    return result


def evaluate_combo(model, scenes, saved_by_scene, scores_by_scene, threshold, *, output, device='cpu', check=None):
    """One frozen forward per anchor and grouping; write all states and routing."""
    output = Path(output)
    destination = output/'predictions'
    destination.mkdir(parents=True, exist_ok=False)
    system_destination = output/'system-predictions'
    system_destination.mkdir(exist_ok=False)
    ids = [s['scene_id'] for s in scenes]
    if len(ids) != len(set(ids)) or set(ids) != set(saved_by_scene) or set(ids) != set(scores_by_scene):
        raise ValueError('exact complete scene population must agree')
    was_training = model.training
    model.eval()
    rows, replay, system_rows, common_rows = [], [], [], []
    try:
        for scene in scenes:
            if check:
                check()
            if scene.get('motion_observation_policy', 'selected_cap128') != getattr(
                    model, 'motion_observation_policy', 'selected_cap128'):
                raise ValueError('fusion model and scene motion_observation_policy differ')
            sid = scene['scene_id']
            saved, scores = saved_by_scene[sid], scores_by_scene[sid]
            motion = aligned_motion(scene, saved, scores)
            n, q = motion['states'].shape[:2]
            partitions = dict(ols046=saved_groups(saved['predicted_anchor_direct_membership']),
                assoc045=group_predicted_tracks(n, scores['pairs'], scores['scores'], threshold=threshold,
                    grouping_policy='anchor_direct', node_order='input'))
            predictions, routing = {}, {}
            for name, groups in partitions.items():
                values, routing[name] = predict_partition(model, motion, groups, device)
                predictions.update({name+'_'+arm: value for arm, value in values.items()})
                if name == 'ols046':
                    item = dict(scene_id=sid, **replay_check(saved, values, routing[name]))
                    replay.append(item)
                    write_json(output/'replay.json', dict(per_scene=replay, passed=all(r['passed'] for r in replay)))
                    if not item['passed']:
                        raise ValueError('046 frozen replay mismatch; fixed tolerance not relaxed: '+sid)
            views = unique_target_view(scene, saved, partitions,
                {group: {arm: predictions[group+'_'+arm] for arm in ('mean_supported', 'learned')}
                 for group in GROUPS}, return_artifact=True)
            system_payload, reports, artifacts = {}, {}, {}
            for group in GROUPS:
                for arm, report in views[group].items():
                    name = group+'_'+arm
                    artifacts[name] = report.pop('artifact')
                    system_payload.update({name+'_'+key: value for key, value in artifacts[name].items()})
                    reports[name] = report
            # Full metrics preserve missed targets/duplicates. This additional
            # intersection view compares errors on the SAME unique target points.
            count = len(scene['target_identity_ids'])
            common_valid = np.logical_and.reduce([
                artifacts[arm]['aligned_prediction_valid'][:count] for arm in ARMS])
            common = {}
            for arm in ARMS:
                artifact = artifacts[arm]
                truth, truth_valid = artifact['aligned_truth'][:count], artifact['aligned_truth_valid'][:count]
                observed = artifact['aligned_observed'][:count]
                common[arm] = dict(full_target_metrics={name: evaluate_aligned_fusion(
                    artifact['aligned_prediction'][:count], truth, truth_valid,
                    prediction_valid=common_valid, region_mask=region)
                    for name, region in (('all', np.ones_like(observed)), ('observed', observed), ('missing', ~observed))})
            system_payload['common_prediction_valid_on_target_rows'] = common_valid
            np.savez_compressed(system_destination/('states-'+sid+'.npz'), **system_payload)
            system_rows.append(dict(scene_id=sid, component_id=scene['component_id'], methods=reports))
            common_rows.append(common)
            # Labels are read for diagnostics only after BOTH predicted outputs exist.
            mixed = {}
            for name, groups in partitions.items():
                flags = np.zeros(n, bool)
                for group in groups:
                    flags[group] = len(set(saved['identities'][group])) > 1
                mixed[name] = np.broadcast_to(flags[:, None], (n, q))
            regions = {key.removeprefix('region_'): value for key, value in saved.items() if key.startswith('region_')}
            for old, oldmask in (('pure', ~mixed['ols046']), ('mixed', mixed['ols046'])):
                for new, newmask in (('pure', ~mixed['assoc045']), ('mixed', mixed['assoc045'])):
                    regions['old_'+old+'_new_'+new] = oldmask & newmask
            payload = {key: saved[key] for key in ('node_ids', 'identities', 'sources', 'query_seconds',
                'target', 'target_valid', 'native_states', 'nearest_seconds')}
            payload.update(predictions)
            payload.update({'region_'+key: value for key, value in regions.items()})
            for name, values in routing.items():
                payload.update({name+'_'+key: value for key, value in values.items()})
            np.savez_compressed(destination/('states-'+sid+'.npz'), **payload)
            rows.append(dict(scene_id=sid, component_id=scene['component_id'], saved=saved,
                regions=regions, predictions=predictions,
                routing={key: routing_diagnostics(saved, value) for key, value in routing.items()}))
        if not rows:
            raise ValueError('nonempty validation population required')
        sources = sorted(set(np.concatenate([r['saved']['sources'] for r in rows]).tolist()))
        components = list(dict.fromkeys(r['component_id'] for r in rows))
        per_scene = [dict(scene_id=r['scene_id'], component_id=r['component_id'], **summarize([r]),
                          routing=r['routing']) for r in rows]
        per_component = [dict(component_id=c, weighting='point pooled within component',
                              **summarize([r for r in rows if r['component_id'] == c]))
                         for c in components]
        paired_units = {}
        for name in COMPARISONS:
            paired_units[name] = {}
            for attribute in ATTRIBUTES:
                paired_units[name][attribute] = {}
                for unit, records in (('scene', per_scene), ('component_point_pooled', per_component)):
                    delta = [r['paired'][name][attribute]['delta_mae'] for r in records]
                    valid_delta = np.array([d for d in delta if d is not None])
                    paired_units[name][attribute][unit] = dict(defined=len(valid_delta),
                        improved=int((valid_delta < -1e-8).sum()), worsened=int((valid_delta > 1e-8).sum()),
                        unchanged=int((abs(valid_delta) <= 1e-8).sum()),
                        mean_delta_mae=float(valid_delta.mean()) if len(valid_delta) else None)
                component_means = []
                for component in components:
                    values = [r['paired'][name][attribute]['delta_mae'] for r in per_scene
                              if r['component_id'] == component]
                    values = [v for v in values if v is not None]
                    component_means.append(dict(component_id=component, scene_count=len(values),
                        delta_mae=float(np.mean(values)) if values else None))
                differences = np.array([r['delta_mae'] for r in component_means if r['delta_mae'] is not None])
                paired_units[name][attribute]['component_scene_macro'] = dict(
                    weighting='mean of scene delta MAE within each component',
                    defined=len(differences), improved=int((differences < -1e-8).sum()),
                    worsened=int((differences > 1e-8).sum()), unchanged=int((abs(differences) <= 1e-8).sum()),
                    per_component=component_means)
        routing_total = {group: {attr: {key: sum(r['routing'][group][attr][key] for r in rows)
            for key in rows[0]['routing'][group][attr]} for attr in ATTRIBUTES} for group in GROUPS}
        valid = np.concatenate([r['saved']['target_valid'] for r in rows])
        system = dict(scope='additional-one-group-one-output-view',
            target_identity_count=sum(row['methods'][ARMS[0]]['target_identity_count'] for row in system_rows),
            overall={arm: aggregate_fusion_reports([row['methods'][arm] for row in system_rows]) for arm in ARMS},
            common_valid_target_points={arm: aggregate_fusion_reports([row[arm] for row in common_rows]) for arm in ARMS},
            output_counts={arm: {key: sum(row['methods'][arm][key] for row in system_rows)
                for key in ('target_identity_count', 'predicted_group_count', 'duplicate_prediction_group_count',
                            'anchor_matched_target_count')} for arm in ARMS},
            per_scene=system_rows, correspondence_policy='anchor_identity_first',
            representative_policy='first node in existing input-order group',
            limitations=['Additional view, not replacement of original fragment pilot',
                'Singleton is OLS-complete; baseline singleton is observed-only; read coverage and common intersection',
                'Both are predicted grouping; baseline oracle diagnostics are not a comparable main result'],
            test_opened=False, optimizer_updates=0)
        write_json(output/'system-evaluation.json', system)
        return dict(scope='frozen-evidence-connection-048', scene_ids=ids, scene_count=len(rows),
            component_count=len(components), fragment_count=len(valid), query_count=int(np.prod(valid.shape[:2])),
            target_valid_by_attribute=valid.sum((0, 1)).tolist(), threshold045=threshold,
            grouping_policy='anchor_direct/input; 046 original membership reused exactly',
            coverage='every original fragment anchor; not unique identity output',
            overall={name: summarize(rows, name) for name in rows[0]['regions']},
            by_source={source: {region: summarize(rows, region, source) for region in rows[0]['regions']}
                       for source in sources}, per_scene=per_scene, per_component=per_component,
            paired_units=paired_units, routing=routing_total, replay_passed=all(r['passed'] for r in replay),
            optimizer_updates=0, threshold_selected=False, test_opened=False,
            limitations=['Repeatedly exposed V; no independent confirmation',
                'Fragment-weighted pilot repeats identities; not full-system comparison',
                'Truth is post-prediction diagnostics only; helper gain uses anchor truth',
                '9001/reference reuse limits denoising conclusions'])
    finally:
        model.train(was_training)
