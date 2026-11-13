"""C105. Compare fixed and fitted physical position readouts on whole held scenes."""

import argparse
from collections import Counter
from pathlib import Path
import resource
import time

import numpy as np
import torch

from association.data.archive import MTADArchive
from association.data.motion_evidence import DATA, scene_motion_evidence
from association.data.physical_units import SPEED_UNIT_POLICY
from association.data.timed import prepare_timed_scene
from association.evaluation.fusion_controls import attribute_errors, ordinary_fusion
from association.models.physical_fusion import geographic_exp, geographic_log
from association.training.common import normalize_scene
from tools.analysis import held_fusion_population as held
from tools.analysis import tmtaff_true_pair_qualification as fixed


C102 = 'LEVEL1-TMTAFF-POSITION-READOUT-20261112-attempt-1'
C93 = 'LEVEL1-TMTAFF-FEEDBACK-GRADIENT-20261112-attempt-1'
SNAPSHOT = 'ORIGINAL-u032'
RIDGE = .1
SMALL_M = 1000.
BASES = ('MEAN', 'NEAREST', 'TFM_U32')
HEADS = {
    'DIRECT': (False, False, True), 'DIRECT_Q': (False, True, True),
    'RESIDUAL': (True, False, True), 'RESIDUAL_Q': (True, True, True),
    'PHYSICAL_RESIDUAL_Q': (True, True, False),
}


def registry(path):
    """Load pair-eligibility identity metadata without reading physical target arrays."""
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in ('node_ids', 'identities', 'query_seconds')}


def choose_pairs(identities, sources):
    left, right = np.triu_indices(len(identities), 1)
    usable = (identities[left] == identities[right]) & (sources[left] != sources[right])
    pairs = np.column_stack((left[usable], right[usable]))
    if len(pairs) < 2:
        return pairs
    other = np.flatnonzero(identities[pairs[:, 0]] != identities[pairs[0, 0]])
    return pairs[[0, int(other[0]) if len(other) else 1]]


def held_scene(archive, row, sid, evaluation, motion):
    tracks = held._canonical_tracks(archive._rows(sid, 'measurements'), sid, row['node_ids'], motion['sources'])
    np.testing.assert_array_equal(row['node_ids'], motion['node_ids'])
    np.testing.assert_array_equal(row['query_seconds'], motion['query_seconds'])
    np.testing.assert_array_equal([len(track['canonical_times']) for track in tracks], motion['native_count'])
    for track, identity in zip(tracks, row['identities'], strict=True):
        track['identity_id'] = str(identity)
    intervals = [[track['canonical_times'][0], track['canonical_times'][-1]] for track in tracks]
    scene = prepare_timed_scene(dict(scene_id=sid, split='V_select', tracks=tracks, sources=motion['sources'],
                                    node_ids=row['node_ids'], identities=row['identities'], intervals=intervals),
                                **DATA, query_times=row['query_seconds'], speed_unit_policy=SPEED_UNIT_POLICY)
    return dict(scene=scene, pairs=choose_pairs(row['identities'], motion['sources']),
                evaluation=evaluation, identities=row['identities'], sid=sid)


def physical(scene, pairs):
    motion = scene_motion_evidence(scene, motion_observation_policy='selected_cap128')
    mean, nearest, geometry = [], [], []
    query = motion['query_seconds']
    for pair in pairs:
        states, ages, masks = (motion[name][pair] for name in ('states', 'nearest_seconds', 'support'))
        controls = ordinary_fusion(states, ages, masks, [[0, 1]])['predictions']
        mean.append(controls['mean_supported'][0, :, :2])
        nearest.append(controls['nearest_observation'][0, :, :2])
        chart = mean[-1][0]
        loc = geographic_log(mean[-1], chart)
        donor = geographic_log(states[..., :2], chart) - loc[None]
        geometry.append(np.column_stack((loc, donor.transpose(1, 0, 2).reshape(len(query), 4),
                                         np.log1p(ages.T / 20.), masks.T)))
    return dict(mean=np.asarray(mean), nearest=np.asarray(nearest), physical=np.asarray(geometry),
                support=motion['support'][pairs].sum(1), query=query)


def labels(row, expected_query):
    pairs = row['pairs']
    with np.load(row['evaluation'], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['node_ids'], row['scene']['node_ids'])
        np.testing.assert_array_equal(saved['identities'], row['identities'])
        np.testing.assert_array_equal(saved['query_seconds'], expected_query)
        by_id = {identity: i for i, identity in enumerate(saved['target_ids'])}
        idx = [by_id[identity] for identity in row['identities'][pairs[:, 0]]]
        truth = saved['target_states'][idx, :, :2]
        valid = saved['target_valid'][idx, :, :2].all(-1)
    fixed.require(valid.dtype == bool and np.isfinite(truth[valid]).all(), 'bad position target or mask')
    return truth, valid


def capture(fusion, left, right):
    output = []

    def collect(_module, _args, state):
        output.append(state.detach())

    hooks = [encoder.register_forward_hook(collect) for encoder in (fusion.left_encoder, fusion.right_encoder)]
    try:
        states = fusion(left.detach().requires_grad_(True), right.detach().requires_grad_(True))['states']
    finally:
        for hook in hooks:
            hook.remove()
    fixed.require(len(output) == 2, 'encoder forward count differs')
    return torch.cat(output, dim=-1).cpu().numpy(), states.detach().cpu().numpy()


def query_design(z, q, conditioned):
    if not conditioned:
        return z
    return np.concatenate((z, q[..., None], q[..., None] ** 2, z * q[..., None]), axis=-1)


def stack(rows, name):
    return np.concatenate([row[name] for row in rows], axis=0)


def head_inputs(row, with_encoder):
    return (np.concatenate((row['physical'], row['encoder']), axis=-1) if with_encoder
            else row['physical'])


def fit_head(rows, residual, conditioned, with_encoder):
    xs, ys, qs, weights = [], [], [], []
    for row in rows:
        use = row['valid']
        xs.append(head_inputs(row, with_encoder)[use])
        truth = [geographic_log(row['truth'][i, valid], row['mean'][i, 0])
                 for i, valid in enumerate(use)]
        y = np.concatenate(truth)
        ys.append(y - row['physical'][use, :2] if residual else y)
        qs.append(row['fraction'][use])
        weights.append(np.full(int(use.sum()), 1. / (len(rows) * int(use.sum()))))
    w, x, y, q = (np.concatenate(value) for value in (weights, xs, ys, qs))
    mean = w @ x
    std = np.sqrt(w @ ((x - mean)**2))
    scale = np.where(std > 1e-8, std, 1.)
    design = query_design((x - mean) / scale, q, conditioned)
    offset = w @ design
    design -= offset
    y0 = w @ y
    normal = design.T @ (w[:, None] * design) + np.eye(design.shape[1]) * RIDGE
    rhs = design.T @ (w[:, None] * (y - y0))
    coefficients = np.linalg.solve(normal, rhs)
    closure = float(np.max(np.abs(normal @ coefficients - rhs)))
    fixed.require(closure < 1e-7, 'closed-form ridge equation did not close')
    return dict(mean=mean, scale=scale, offset=offset, intercept=y0, beta=coefficients, closure=closure)


def predict(row, head, residual, conditioned, with_encoder):
    x = (head_inputs(row, with_encoder) - head['mean']) / head['scale']
    z = query_design(x, row['fraction'], conditioned) - head['offset']
    local = z @ head['beta'] + head['intercept']
    if residual:
        local += row['physical'][..., :2]
    return np.stack([geographic_exp(position, center) for position, center in zip(local, row['mean'][:, 0], strict=True)])


def distance(first, second):
    return attribute_errors(np.pad(first, [(0, 0)] * (first.ndim - 1) + [(0, 2)]),
                            np.pad(second, [(0, 0)] * (second.ndim - 1) + [(0, 2)]))['position']


def delta_m(first, second, origin):
    return geographic_log(first, origin) - geographic_log(second, origin)


def evaluate_rows(rows, components, strict_ids):
    scene_index, pair_scene, support, valid, gap, arms = [], [], [], [], [], (*BASES, *HEADS)
    stats = {key: dict(errors=[], vectors=[], assignment=[]) for key in arms}
    for index, row in enumerate(rows):
        y, ok = labels(row, row['query'])
        scene_index.extend([index] * len(y))
        support.append(row['support'])
        valid.append(ok)
        common = ok.all(0) if len(y) == 2 and row['distinct'] else np.zeros(y.shape[1], bool)
        n = int(common.sum())
        pair_scene.extend([index] * n)
        if n:
            truth_gap = distance(y[0, common], y[1, common])
            gap.extend(truth_gap.tolist())
            center = row['mean'][0, 0]
            truth_delta = delta_m(y[0, common], y[1, common], center)
        for name in arms:
            p = row['predictions'][name]
            errors = distance(p, y)
            stats[name]['errors'].append(errors)
            if n:
                vector = delta_m(p[0, common], p[1, common], center)
                swap = distance(p[1, common], y[0, common]) + distance(p[0, common], y[1, common])
                stats[name]['vectors'].extend(np.linalg.norm(vector - truth_delta, axis=-1).tolist())
                stats[name]['assignment'].extend((swap - errors[:, common].sum(0)).tolist())
    scene_index, pair_scene, gap = map(np.asarray, (scene_index, pair_scene, gap))
    support, valid = np.concatenate(support), np.concatenate(valid)
    scene_ids = [r['sid'] for r in rows]
    groups = np.asarray([components[sid] for sid in scene_ids])
    largest = Counter(groups[scene_index]).most_common(1)[0][0]
    regions = dict(all=valid, dual=valid & (support == 2), single=valid & (support == 1),
                   unsupported=valid & (support == 0), no_largest=valid & (groups[scene_index] != largest)[:, None],
                   full_T_component_disjoint=valid & np.isin(np.asarray(scene_ids)[scene_index], list(strict_ids))[:, None])
    base = np.concatenate(stats['MEAN']['errors'])
    public = dict(scene_index=scene_index, pair_scene=pair_scene, support=support, valid=valid, gap_m=gap)
    result = {}
    for arm, samples in stats.items():
        err = np.concatenate(samples['errors'])
        contrast, assigned = map(np.asarray, (samples['vectors'], samples['assignment']))
        fixed.require(np.isfinite(err[valid]).all(), 'nonfinite common-population error')
        change = err[valid] - base[valid]
        strata = {name: fixed.statistics(err[mask]) for name, mask in regions.items()}
        for name, mask in (('contrast_all', np.ones(len(gap), bool)), ('contrast_le_1km', gap <= SMALL_M)):
            strata[name] = dict(vector=fixed.statistics(contrast[mask]),
                                assign_correct=int((assigned[mask] > 1e-6).sum()),
                                assign_tied=int((np.abs(assigned[mask]) <= 1e-6).sum()))
        by_scene = {sid: fixed.statistics(err[valid & (scene_index == i)[:, None]])
                    for i, sid in enumerate(scene_ids)}
        by_component = {str(group): fixed.statistics(err[valid & (groups[scene_index] == group)[:, None]])
                        for group in sorted(set(groups))}
        strata.update(scenes=by_scene, components=by_component, paired=dict(
            improved=int((change < -1e-6).sum()), harmed=int((change > 1e-6).sum()),
            harmed_gt_1km=int((change > 1000).sum()), worst_harm_m=float(change.max())))
        result[arm] = strata
        public.update({f'{arm}/error_m': err, f'{arm}/vector_m': contrast, f'{arm}/assignment_gain_m': assigned})
    return result, public, str(largest)


def gate(results):
    r, b, q, plain = (results[name] for name in ('RESIDUAL_Q', 'MEAN', 'DIRECT_Q', 'RESIDUAL'))
    small, base_small = (x['contrast_le_1km'] for x in (r, b))
    scene_checks = [r['scenes'][sid]['mae'] < value['mae'] for sid, value in b['scenes'].items() if value['count']]
    return dict(mean_and_p99=all(r['all'][key] < b['all'][key] for key in ('mae', 'p99')),
                better_matched_heads=all(r['all'][key] < m['all'][key] for m in (q, plain) for key in ('mae', 'p99')),
                dual_tail=r['dual']['p99'] <= b['dual']['p99'],
                small_targets=small['vector']['count'] > 0 and small['vector']['mae'] <= base_small['vector']['mae']
                and small['assign_correct'] >= base_small['assign_correct'],
                half_scenes=sum(scene_checks) * 2 >= len(scene_checks),
                without_largest=all(r['no_largest'][key] < b['no_largest'][key] for key in ('mae', 'p99')))


def run(args, start):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    fixed.require(device.type == 'cuda' and torch.cuda.is_available(), 'assigned CUDA device required')
    private = args.output / 'private'
    private.mkdir(mode=0o700)
    roots = {name: args.runs / value for name, value in fixed.RUNS.items()}
    tcm, normalizer, config, _, _, _ = fixed.source_checkpoints(roots['checkpoint'], device)
    tcm.eval()
    checkpoint = torch.load(roots['checkpoint'] / 'tcm/checkpoint.pt', map_location='cpu', weights_only=True)
    train_rows, old_v_rows = checkpoint['data_contract']
    model_t = {str(row['scene_id']) for row in train_rows}
    old_model_v = {str(row['scene_id']) for row in old_v_rows}
    fixed.require(checkpoint['selected'] is None, '107 TCM was V-selected')
    del checkpoint, train_rows, old_v_rows
    population, components, archives = held.load_population(args.runs)
    full = args.runs / held.PARENTS['full_data']
    manifest = held.read(Path(held.read(full / 'run.json')['manifest']))
    c = manifest['components']
    fixed.require(not model_t.intersection(population) and not set(fixed.SCENES).intersection(population)
                  and not {c[s] for s in fixed.SCENES} & set(components.values()), 'unseen roster leakage')
    strict_ids = [sid for sid in population if c[sid] not in {c[t] for t in model_t}]
    exposure = dict(TCM_optimization_contract_scenes=len(model_t),
                    TCM_legacy_V_contract_scenes=len(old_model_v),
                    held_in_TCM_legacy_V_contract=sum(sid in old_model_v for sid in population),
                    held_sharing_TCM_T_component=len(population)-len(strict_ids),
                    held_component_disjoint_from_TCM_T=strict_ids, TCM_checkpoint_selected=False)
    cached = fixed.read_npz(args.runs / C102 / 'private/original_u32-representations.npz')
    original_states = fixed.read_npz(args.runs / C93 / 'private' / f'{SNAPSHOT}.npz')['states']
    np.testing.assert_array_equal(cached['states'], original_states)
    contract = fixed.read_npz(args.runs / fixed.REFERENCE_C90 / 'private/contract.npz')
    training, train_ids = [], set()
    for i, sid in enumerate(fixed.SCENES):
        filename = f'scene-{sid}.npz'
        events = fixed.read_npz(roots['native'] / 'private/native-events/T_core' / filename)
        evaluation = roots['native'] / 'private/native-evaluation/T_core' / filename
        binding = registry(evaluation)
        ids = binding['identities']
        pairs = fixed.make_pairs(ids, events['sources'])
        scene = fixed.timed_scene(events, ids, sid)
        row = dict(sid=sid, scene=scene, pairs=pairs, evaluation=evaluation, identities=ids)
        row.update(physical(scene, pairs))
        sl = slice(2*i, 2*i+2)
        row['truth'], row['valid'] = labels(row, row['query'])
        row['encoder'] = cached['encoder_local'][sl]
        row['fraction'] = np.broadcast_to(np.linspace(-1., 1., len(row['query'])), row['valid'].shape)
        for key, check in (('CAP128_MEAN', row['mean']), ('selected_support', row['support']),
                           ('target_states', row['truth'])):
            wanted = contract[key][sl]
            np.testing.assert_array_equal(check, wanted[..., :2] if wanted.ndim == 3 else wanted)
        np.testing.assert_array_equal(row['valid'], contract['target_valid'][sl, :, :2].all(-1))
        train_ids.update(str(ids[i]).split('-', 1)[0] for i in pairs[:, 0])
        training.append(row)
    fixed.require(int(stack(training, 'valid').sum()) == 732, 'C102 T8 fit count differs')
    heads = {name: fit_head(training, *spec) for name, spec in HEADS.items()}
    np.savez_compressed(private / 'heads.npz', **{f'{name}/{k}': v for name, head in heads.items() for k, v in head.items()})

    eligible, roster = [], []
    with MTADArchive(archives) as archive:
        held.validate_measurement_archive(args.runs, archive)
        for sid in population:
            filename = f'scene-{sid}.npz'
            evaluation = full / 'private/evaluation/V_select' / filename
            row = held_scene(archive, registry(evaluation), sid, evaluation,
                             fixed.read_npz(full / 'private/motion/V_select' / filename))
            pairs = row['pairs']
            ids = row['identities'][pairs[:, 0]]
            fixed.require(not train_ids & {str(identity).split('-', 1)[0] for identity in ids},
                          'T8 fit identity crossed held pair')
            row['distinct'] = len(np.unique(ids)) == 2
            roster.append(dict(scene=sid, pairs=len(pairs), distinct_targets=row['distinct']))
            if len(pairs):
                scene = row['scene']
                row.update(physical(scene, pairs))
                row['fraction'] = np.broadcast_to(np.linspace(-1., 1., len(row['query'])), row['support'].shape)
                packed, observed = normalize_scene(scene, normalizer)
                nodes, lookup = np.unique(pairs, return_inverse=True)
                lookup = lookup.reshape(pairs.shape)
                with torch.no_grad():
                    completed = tcm.complete(torch.as_tensor(packed[nodes], dtype=next(tcm.parameters()).dtype,
                                                             device=device),
                                             torch.as_tensor(observed[nodes], device=device))
                row['left'], row['right'] = (completed[lookup[:, i].tolist()].detach() for i in (0, 1))
                row['scene'] = dict(node_ids=scene['node_ids'])
                eligible.append(row)
        fixed.require([(entry['scene_id'], entry['kind']) for entry in archive.members_read]
                      == [(sid, 'measurements') for sid in population], 'unexpected archive members read')
    del tcm
    fusion = fixed.PairFusion(config, normalizer).to(device).eval().requires_grad_(False)
    fusion.load_state_dict(torch.load(args.runs / C93 / 'private' / f'{SNAPSHOT}.pt', map_location='cpu', weights_only=True)['fusion'])
    left, right = (torch.cat([row.pop(side) for row in eligible]) for side in ('left', 'right'))
    chunks = [capture(fusion, left[i:i+fixed.MICROBATCH], right[i:i+fixed.MICROBATCH])
              for i in range(0, len(left), fixed.MICROBATCH)]
    encoder, states = (np.concatenate(values) for values in zip(*chunks, strict=True))
    n = 0
    for row in eligible:
        k = len(row['pairs'])
        row['encoder'] = encoder[n:n+k]
        reconstruction = np.stack([geographic_exp(values[:, :2], mean[0])
                                   for values, mean in zip(row['physical'], row['mean'], strict=True)])
        fixed.require(float(distance(reconstruction, row['mean']).max()) < 1e-6, 'local chart did not round-trip Mean')
        predictions = dict(MEAN=row['mean'], NEAREST=row['nearest'], TFM_U32=states[n:n+k, :, :2])
        predictions.update({arm: predict(row, heads[arm], *spec) for arm, spec in HEADS.items()})
        fixed.require(all(np.isfinite(value).all() and (np.abs(value[..., 0]) <= 90).all()
                          for value in predictions.values()), 'invalid position prediction')
        np.savez_compressed(private / f"pred-{row['sid']}.npz", pairs=row['pairs'], query=row['query'], **predictions)
        n += k
    fixed.write_json(args.output / 'freeze.json', dict(roster=roster, exposure=exposure,
                     predictions_before_V_target_read=True, archive_measurements=len(population), raw_label_or_reference_members=0))
    for row in eligible:
        saved = fixed.read_npz(private / f"pred-{row['sid']}.npz")
        np.testing.assert_array_equal(saved.pop('pairs'), row['pairs'])
        np.testing.assert_array_equal(saved.pop('query'), row['query'])
        row['predictions'] = saved
    results, public, largest = evaluate_rows(eligible, components, strict_ids)
    gates = gate(results)
    fixed.write_json(args.output / 'summary.json', dict(roster=roster, exposure=exposure, metrics=results,
                     eligible_scene_ids=[row['sid'] for row in eligible], components=components,
                     fit=dict(scene_count=len(training), pairs=16, valid_positions=732, ridge=RIDGE,
                              heads={name: dict(coefficients=int(head['beta'].size), closure=head['closure'])
                                     for name, head in heads.items()}),
                     largest_eligible_component=largest, gate=gates, passed=all(gates.values())))
    np.savez_compressed(args.output / 'public-errors.npz', **public)
    fixed.write_json(args.output / 'run.json', dict(status='COMPLETED', elapsed_seconds=time.monotonic()-start,
                     optimizer_updates=0, test_opened=False, held_archive_measurements=len(population),
                     held_scenes=len(eligible), held_pairs=len(left),
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    start = time.monotonic()
    try:
        run(args, start)
    except Exception as error:
        fixed.write_json(args.output / 'run.json', dict(status='FAILED', error=repr(error),
                         elapsed_seconds=time.monotonic()-start, test_opened=False))
        raise


if __name__ == '__main__':
    main()
