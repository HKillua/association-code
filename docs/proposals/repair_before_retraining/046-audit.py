"""CPU-only independent audit of returned 046 predictions, no model inference.

Run from repo root: .venv/bin/python docs/proposals/repair_before_retraining/046-audit.py
Uses only NumPy, returned predictions/JSON, and optional saved R4 predictions.
"""
from pathlib import Path
import gzip
import io
import json
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / 'HandOver/results/EVIDENCE-FUSION-20261021-046'
GROUPS = ('predicted_anchor_direct', 'oracle_identity')
ARMS = ('anchor', 'nearest_observation', 'mean_supported', 'learned')
ATTRS = ('position', 'speed', 'heading')


def load_json(name):
    with gzip.open(RESULT / name, 'rt') as f:
        return json.load(f)


def archive(name):
    result = {}
    with tarfile.open(RESULT / name) as t:
        for member in t:
            if member.name.endswith('.npz'):
                with np.load(io.BytesIO(t.extractfile(member).read())) as z:
                    result[Path(member.name).stem.removeprefix('states-')] = {
                        k: z[k] for k in z.files}
    return result


def errors(prediction, target):
    lat1, lat2 = np.deg2rad(prediction[..., 0]), np.deg2rad(target[..., 0])
    dlon = np.deg2rad((prediction[..., 1] - target[..., 1] + 180) % 360 - 180)
    h = np.clip(np.sin((lat1-lat2)/2)**2
                + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2, 0, 1)
    return np.stack((2*6371000*np.arctan2(np.sqrt(h), np.sqrt(1-h)),
                     abs(prediction[..., 2]-target[..., 2]),
                     abs((prediction[..., 3]-target[..., 3]+180) % 360-180)), -1)


def metrics(e, mask):
    result = {}
    for i, attribute in enumerate(ATTRS):
        v = e[..., i][mask[..., i]]
        result[attribute] = dict(n=len(v), mae=float(v.mean()),
                                rmse=float(np.sqrt(np.mean(v*v))),
                                p95=float(np.percentile(v, 95)),
                                p99=float(np.percentile(v, 99)), max=float(v.max())) if len(v) else dict(n=0)
    return result


def compare(e, control, mask):
    result = {}
    for i, a in enumerate(ATTRS):
        delta = (e[..., i]-control[..., i])[mask[..., i]]
        result[a] = dict(delta_mae=float(delta.mean()), improved=int((delta < -1e-8).sum()),
                         worsened=int((delta > 1e-8).sum()), unchanged=int((abs(delta) <= 1e-8).sum()))
    return result


def main():
    report = load_json('evaluation-step-1000.json.gz')
    zero, final = archive('predictions-step-0.tar.gz'), archive('predictions-final.tar.gz')
    assert set(zero) == set(final) == set(report['scene_ids'])
    component = {x['scene_id']: x['component_id'] for x in report['per_scene']}
    rows, same_control, zero_exact, r4_differences = [], True, True, []
    r4dir = ROOT / 'runs/FIXED-GROUP-FUSION-20261020-local/predictions'
    r4errors = {arm: [] for arm in ARMS[:-1]}
    for sid in sorted(final):
        z, p = zero[sid], final[sid]
        for k in ('node_ids', 'identities', 'sources', 'query_seconds', 'target_valid'):
            assert np.array_equal(z[k], p[k])
        assert np.array_equal(z['target'], p['target'], equal_nan=True)
        v = p['target_valid']
        valid = np.stack((v[..., :2].all(-1), v[..., 2], v[..., 3]), -1)
        entry = dict(scene_id=sid, component_id=component[sid], payload=p, valid=valid, error={})
        for g in GROUPS:
            zero_exact &= np.array_equal(z[g+'_learned'], z[g+'_anchor'])
            same_control &= np.array_equal(z[g+'_membership'], p[g+'_membership'])
            for arm in ARMS:
                if arm != 'learned':
                    same_control &= np.array_equal(z[g+'_'+arm], p[g+'_'+arm])
                entry['error'][g+'_'+arm] = errors(p[g+'_'+arm], p['target'])
        membership = p[GROUPS[0]+'_membership']
        mixed = np.array([len(set(p['identities'][membership == m])) > 1 for m in membership])
        entry['mixed'] = np.broadcast_to(mixed[:, None], v.shape[:2])
        oldpath = r4dir / ('states-'+sid+'.npz')
        if oldpath.exists():
            with np.load(oldpath) as old:
                for k in ('node_ids', 'identities', 'sources', 'target_valid'):
                    assert np.array_equal(old[k], p[k])
                assert np.array_equal(old['target'], p['target'], equal_nan=True)
                changed = not np.array_equal(old[GROUPS[0]+'_membership'], membership)
                if changed:
                    pos = p['native_states'][..., :2]
                    oldpos = old['native_states'][..., :2]
                    pairs = np.column_stack(np.triu_indices(len(pos), 1))
                    dist = np.sqrt(np.mean(errors(np.concatenate((pos[pairs[:, 0]],
                        np.zeros((*pos[pairs[:, 0]].shape[:2], 2))), -1),
                        np.concatenate((pos[pairs[:, 1]], np.zeros((*pos[pairs[:, 1]].shape[:2], 2))), -1))[..., 0]**2, axis=1))
                    closest = int(np.argmin(abs(dist-report['fixed_distance_threshold_m'])))
                    i, j = pairs[closest]
                    oldd = float(np.sqrt(np.mean(errors(old['native_states'][i], old['native_states'][j])[:, 0]**2)))
                    r4_differences.append(dict(scene_id=sid,
                        changed_membership_nodes=int(np.count_nonzero(old[GROUPS[0]+'_membership'] != membership)),
                        threshold_pair=p['node_ids'][[i, j]].tolist(),
                        identities=p['identities'][[i, j]].tolist(),
                        old_distance_m=oldd, new_distance_m=float(dist[closest]),
                        threshold_m=report['fixed_distance_threshold_m'],
                        native_position_max_abs_delta=float(np.max(abs(pos-oldpos)))))
                for arm in ARMS[:-1]:
                    r4errors[arm].append(errors(old[GROUPS[0]+'_'+arm], p['target']))
        rows.append(entry)
    cat = lambda f: np.concatenate([f(r) for r in rows])
    valid = cat(lambda r: r['valid'])
    regions = {k.removeprefix('region_'): cat(lambda r: r['payload'][k])
               for k in rows[0]['payload'] if k.startswith('region_')}
    regions.update(mixed_group=cat(lambda r: r['mixed']), pure_group=~cat(lambda r: r['mixed']))
    source = cat(lambda r: r['payload']['sources'])
    allerr = {g+'_'+arm: cat(lambda r: r['error'][g+'_'+arm]) for g in GROUPS for arm in ARMS}
    output = dict(experiment='046', source='saved predictions and reports; no inference',
        scene_count=len(rows), component_count=len(set(component.values())),
        anchors=len(valid), queries=int(np.prod(valid.shape[:2])), valid_by_attribute=valid.sum((0, 1)).tolist(),
        step0_learned_bitwise_equals_anchor=bool(zero_exact), controls_unchanged_step0_final=bool(same_control),
        overall={}, by_source={}, routing={}, comparison={}, per_scene=[], per_component=[],
        r4_control_comparison=dict(available_scene_count=len(r4errors['anchor']), grouping_differences=r4_differences))
    max_difference = 0.
    for g in GROUPS:
        output['overall'][g] = {}
        for arm in ARMS:
            e = allerr[g+'_'+arm]
            result = {name: metrics(e, valid & region[..., None]) for name, region in regions.items()
                      if g == GROUPS[0] or name not in ('mixed_group', 'pure_group')}
            output['overall'][g][arm] = result
            # Oracle report has no mixed anchors; predicted mixed mask is only our diagnostic there.
            for name in regions if g == GROUPS[0] else [n for n in regions if n not in ('mixed_group','pure_group')]:
                for a in ATTRS:
                    for metric in ('mae', 'rmse', 'p95'):
                        ours = result[name][a].get(metric)
                        theirs = report['overall'][g][arm]['regions'][name][a][metric]
                        if ours is not None:
                            max_difference = max(max_difference, abs(ours-theirs))
        p = cat(lambda r: r['payload'][g+'_anchor_weights'])
        rejected = cat(lambda r: r['payload'][g+'_zero_helper_counts'])
        counts = cat(lambda r: r['payload'][g+'_helper_counts'])
        eligible = counts > 0
        assert np.all((p > 0) & (p <= 1))
        assert np.array_equal(rejected == counts[:, None, None], p == 1)
        output['routing'][g] = dict(helper_query_count=int(counts.sum()*valid.shape[1]),
            rejection_fraction=(rejected.sum((0, 1))/(counts.sum()*valid.shape[1])).tolist(),
            all_helpers_rejected_fraction=(p[eligible] == 1).mean((0, 1)).tolist(),
            eligible_anchor_weight_mean=p[eligible].mean((0, 1)).tolist(),
            eligible_anchor_weight_quantiles=np.quantile(p[eligible].reshape(-1, 3), [.05,.5,.95], axis=0).tolist(),
            singleton_anchors=int((~eligible).sum()),
            by_source={str(s): dict(
                anchor_weight_mean=p[(source == s) & eligible].mean((0, 1)).tolist(),
                all_rejected_fraction=(p[(source == s) & eligible] == 1).mean((0, 1)).tolist())
                for s in sorted(set(source))})
        potential, retained, masks = [], [], []
        for r in rows:
            data = r['payload']; members = data[g+'_membership']; n, q = data['target'].shape[:2]
            best = np.full((n, q, 3), np.inf)
            for i in range(n):
                helpers = np.flatnonzero((members == members[i]) & (np.arange(n) != i))
                if len(helpers):
                    best[i] = errors(data['native_states'][helpers], data['target'][i]).min(0)
            potential.append(best < r['error'][g+'_anchor']-1e-8)
            retained.append(data[g+'_anchor_weights'])
            masks.append(r['valid'])
        can_help, weights, v = map(np.concatenate, (potential, retained, masks))
        output['routing'][g]['useful_single_helper_available'] = {a: dict(
            valid_queries=int(v[..., i].sum()),
            queries_with_some_better_helper=int((v[..., i] & can_help[..., i]).sum()),
            all_rejected_despite_better_helper=int((v[..., i] & can_help[..., i] & (weights[..., i] == 1)).sum()),
            mean_anchor_weight_when_better_helper=float(weights[..., i][v[..., i] & can_help[..., i]].mean()))
            for i, a in enumerate(ATTRS)}
    output['max_metric_absolute_difference_vs_report'] = max_difference
    assert max_difference < 1e-7
    g = GROUPS[0]
    for s in sorted(set(source)):
        mask = np.broadcast_to((source == s)[:, None, None], valid.shape) & valid
        output['by_source'][s] = {arm: {name: metrics(allerr[g+'_'+arm], mask & region[..., None])
            for name, region in regions.items()} for arm in ARMS}
    for arm in ARMS[:-1]:
        output['comparison'][arm] = {name: compare(allerr[g+'_learned'], allerr[g+'_'+arm],
            valid & region[..., None]) for name, region in regions.items()}
    for r in rows:
        e, control = r['error'][g+'_learned'], r['error'][g+'_mean_supported']
        output['per_scene'].append(dict(scene_id=r['scene_id'], component_id=r['component_id'],
            learned=metrics(e, r['valid']), mean_supported=metrics(control, r['valid']),
            comparison=compare(e, control, r['valid'])))
    for cid in sorted(set(component.values())):
        rs = [r for r in rows if r['component_id'] == cid]
        v = np.concatenate([r['valid'] for r in rs])
        e = np.concatenate([r['error'][g+'_learned'] for r in rs])
        control = np.concatenate([r['error'][g+'_mean_supported'] for r in rs])
        output['per_component'].append(dict(component_id=cid, scene_count=len(rs),
            learned=metrics(e, v), mean_supported=metrics(control, v), comparison=compare(e, control, v)))
    output['aggregate_units'] = {}
    for unit, entries in [('scene', output['per_scene']), ('component', output['per_component'])]:
        output['aggregate_units'][unit] = {a: dict(
            better=sum(x['comparison'][a]['delta_mae'] < -1e-8 for x in entries),
            worse=sum(x['comparison'][a]['delta_mae'] > 1e-8 for x in entries),
            tied=sum(abs(x['comparison'][a]['delta_mae']) <= 1e-8 for x in entries),
            equal_unit_mean_delta=float(np.mean([x['comparison'][a]['delta_mae'] for x in entries]))) for a in ATTRS}
    if len(r4errors['anchor']) == len(rows):
        output['r4_control_comparison']['old_metrics'] = {
            arm: metrics(np.concatenate(es), valid) for arm, es in r4errors.items()}
    # Saved aggregate weights cannot identify which helper was rejected in multi-helper groups.
    # Single-helper groups do permit an exact helpfulness-versus-acceptance diagnostic.
    single_rows = []
    for r in rows:
        p = r['payload']; membership = p[g+'_membership']; counts = p[g+'_helper_counts']
        for i in np.flatnonzero(counts == 1):
            j = int(np.flatnonzero((membership == membership[i]) & (np.arange(len(counts)) != i))[0])
            helper_error = errors(p['native_states'][j], p['target'][i])
            anchor_error = r['error'][g+'_anchor'][i]
            single_rows.append((helper_error-anchor_error, p[g+'_anchor_weights'][i] < 1,
                                r['valid'][i], p['identities'][i] == p['identities'][j]))
    output['single_helper_diagnostic'] = {}
    for a, attr in enumerate(ATTRS):
        delta = np.concatenate([x[0][:, a] for x in single_rows])
        accepted = np.concatenate([x[1][:, a] for x in single_rows])
        v = np.concatenate([x[2][:, a] for x in single_rows])
        same = np.concatenate([np.full(len(x[2]), x[3]) for x in single_rows])
        helpful, harmful = delta < -1e-8, delta > 1e-8
        output['single_helper_diagnostic'][attr] = dict(valid_queries=int(v.sum()),
            accepted=int((v & accepted).sum()), helpful_accepted=int((v & accepted & helpful).sum()),
            harmful_accepted=int((v & accepted & harmful).sum()),
            helpful_rejected=int((v & ~accepted & helpful).sum()), harmful_rejected=int((v & ~accepted & harmful).sum()),
            wrong_identity_queries=int((v & ~same).sum()), wrong_identity_accepted=int((v & ~same & accepted).sum()))
    output['validation_curve'] = []
    for step in range(0, 1001, 100):
        d = load_json(f'evaluation-step-{step}.json.gz')
        learned = d['overall'][g]['learned']['regions']['all']
        output['validation_curve'].append(dict(step=step, **{a: learned[a]['mae'] for a in ATTRS},
            rejection_fraction=d['routing'][g]['rejection_fraction_by_attribute']))
    training = [json.loads(line) for line in gzip.open(RESULT/'training.jsonl.gz', 'rt')]
    output['training'] = dict(updates=len(training), all_steps_sequential=[x['step'] for x in training] == list(range(1, 1001)),
        bins=[dict(start=start+1, stop=start+100, **{k: float(np.mean([x[k] for x in training[start:start+100]]))
            for k in ('total', 'direct', 'utility', 'gradient_norm')}) for start in range(0, 1000, 100)],
        by_mode={mode: {k: float(np.mean([x[k] for x in training if x['mode'] == mode]))
            for k in ('total', 'direct', 'utility')} for mode in ('clean', 'clean_plus_wrong', 'wrong_only')})
    output['limits'] = ['No inference or retraining was run.',
        'Saved individual helper weights/utilities are unavailable; multi-helper rejection correctness cannot be identified.',
        'One seed, repeatedly exposed V; these are descriptive pilot results, not independent confirmation.',
        'Oracle grouping is diagnostic and does not establish an upper bound for every metric.',
        'Position/heading support tradeoffs remain; final step 1000 is the precommitted endpoint.']
    path = Path(__file__).with_suffix('.json')
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: output[k] for k in ('scene_count','anchors','queries','valid_by_attribute',
        'step0_learned_bitwise_equals_anchor','controls_unchanged_step0_final','max_metric_absolute_difference_vs_report',
        'aggregate_units','single_helper_diagnostic','validation_curve')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
