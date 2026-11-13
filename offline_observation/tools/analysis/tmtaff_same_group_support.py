"""C129: saved same-group comparisons and a same-input TFM replay."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
import torch

from association.evaluation.artifacts import load_scene_prediction
from association.evaluation.mixture_opportunity import _distance
from association.evaluation.paper_fusion_metrics import aggregate_paper_fusion
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.tmtaff import predicted_observed_controls, scene_input_support
from association.training.common import normalize_scene
from association.training.tmtaff import load_complete_snapshot
from tools.analysis.held_fusion_population import HELD_IDS, PARENTS, SPLIT, load_full_population, read_evaluation
from tools.analysis.motion_readout_evaluation import _stats, _wrap
from tools.analysis.same_source_swap_pilot import group_members, read, write
from tools.analysis.tmtaff_c94_paired_bridge import C94, errors, metrics, summary_state
from tools.analysis.v300_far_paper_bridge import PEER, RUN113, verify_peer_inputs


C126 = 'LEVEL1-TMTAFF-C94-PAIRED-20261113-attempt-1'
REPLAY = HELD_IDS[:4]
ARMS = ('FAR', 'T128_FAR', PEER)
ATTRIBUTES = ('longitude', 'latitude', 'speed', 'heading', 'position')
REGIONS = ('paired', 'different_group', 'same_group', 'same_support', 'full_count',
           'zero_active', 'one_active', 'multiple_active')


def first_claim(groups, node_ids, targets):
    result = {}
    index = {identity: i for i, identity in enumerate(targets.tolist())}
    for gi, group in enumerate(groups):
        result.setdefault(index[node_ids[group[0]]], gi)
    return result


def masks_for(saved, ev, old, scene, motion):
    groups = group_members(saved['membership'])
    peer_groups = np.split(old['group_node_indices'], old['group_offsets'][1:-1])
    if len(old['group_states']) == 0:
        peer_groups = []
    targets = ev['target_ids']
    ws_first = first_claim(groups, ev['identities'], targets)
    tm_first = first_claim(peer_groups, ev['identities'], targets)
    peer_rows = {target: i for i, target in enumerate(old['target_ids'].tolist())}
    order = np.asarray([peer_rows[target] for target in targets.tolist()], dtype=int)
    for key, expected in (('aligned_truth_valid', ev['target_valid']), ('aligned_truth', ev['target_states'])):
        np.testing.assert_array_equal(old[key][order], expected)
    np.testing.assert_array_equal(old['aligned_group_indices'][order], [tm_first.get(i, -1) for i in range(len(order))])
    peer_support = scene_input_support(scene)
    for key, actual in (('cap128_support', peer_support['support']),
                        ('cap128_nearest_seconds', peer_support['nearest_seconds']),
                        ('support', saved['support']), ('nearest_seconds', saved['nearest'])):
        np.testing.assert_array_equal(motion[key], actual)
    original_count = motion['native_count']
    cap_count = scene['native_observed'].sum(1)
    assert (original_count >= cap_count).all()

    own_valid = np.zeros_like(ev['target_valid'])
    predictions = {name: np.zeros_like(ev['target_states']) for name in ARMS[:2]}
    predictions[PEER] = old['aligned_prediction'][order]
    ordered_same = np.zeros(len(targets), bool)
    same_support = np.zeros(own_valid.shape[:2], bool)
    same_count = np.zeros(len(targets), bool)
    active = np.zeros(own_valid.shape[:2], int)
    for ti, gi in ws_first.items():
        own_valid[ti] = True
        for name in ARMS[:2]:
            predictions[name][ti] = saved[name][gi]
        nodes = groups[gi]
        active[ti] = saved['support'][nodes].sum(axis=0)
        gj = tm_first.get(ti)
        if gj is None or not np.array_equal(nodes, peer_groups[gj]):
            continue
        ordered_same[ti] = True
        same_support[ti] = ((saved['support'][nodes] == peer_support['support'][nodes])
                            & (saved['nearest'][nodes] == peer_support['nearest_seconds'][nodes])).all(0)
        same_count[ti] = (original_count[nodes] == cap_count[nodes]).all()
    np.testing.assert_array_equal(predictions['FAR'][..., 2:], predictions['T128_FAR'][..., 2:])
    common = own_valid & old['aligned_prediction_valid'][order] & ev['target_valid']
    filters = dict(paired=np.ones(common.shape[:2], bool),
                   different_group=np.broadcast_to(~ordered_same[:, None], common.shape[:2]),
                   same_group=np.broadcast_to(ordered_same[:, None], common.shape[:2]),
                   same_support=same_support,
                   full_count=same_support & same_count[:, None],
                   zero_active=same_support & (active == 0), one_active=same_support & (active == 1),
                   multiple_active=same_support & (active >= 2))
    scopes = {region: common & mask[..., None] for region, mask in filters.items()}
    return predictions, scopes, dict(declared=len(targets), both_claimed=len(ws_first.keys() & tm_first.keys()),
               same_group=int(ordered_same.sum()), full_count=int(same_count.sum())), groups, peer_groups


def collect(predictions, scopes, ev):
    vectors = {scope: {name: errors(value, ev['target_states'], mask) for name, value in predictions.items()}
               for scope, mask in scopes.items()}
    paper = {scope: metrics(predictions, dict.fromkeys(predictions, scopes[scope]), ev)
             for scope in ('same_group', 'same_support', 'full_count')}
    return vectors, paper


def support_mean(scene, groups):
    return predicted_observed_controls(scene['values'], scene['observed'], [g.tolist() for g in groups])['predicted_mean']


@torch.inference_mode()
def complete(snapshot, scene):
    normalizer = snapshot['normalizer']
    config = normalizer['adapter_config']
    prepared = dict(scene, adapter_config=config, time_origin_seconds=float(scene['grid'][0]),
                    **{key: config[key] for key in ('length', 'max_observations', 'time_unit_seconds',
                                                    'observation_interpolation')})
    x, mask = normalize_scene(prepared, normalizer)
    model = snapshot['model']
    model.requires_grad_(False)
    batch = 16
    return torch.cat([model.complete(torch.as_tensor(x[i:i+batch]), torch.as_tensor(mask[i:i+batch]))
                      for i in range(0, len(x), batch)])


@torch.inference_mode()
def recur(fusion, latent, scene, groups):
    predictions = []
    forwards = 0
    for members in groups:
        group = members.tolist()
        if len(group) == 1:
            idx = group[0]
            predictions.append(dict(node_indices=group, states=scene['values'][idx],
                                    prediction_valid=np.repeat(scene['observed'][idx, :, None], 4, axis=1)))
            continue
        memory = latent[group[0]:group[0]+1]
        for idx in group[1:]:
            fused = fusion(memory, latent[idx:idx+1])
            memory = fused['fused_features']
            forwards += 1
        predictions.append(dict(node_indices=group, states=fused['states'][0].numpy()))
    return predictions, forwards


def evaluate(outputs, scene, ev):
    return evaluate_full_scene(outputs, node_identities=scene['identities'], target_identities=ev['target_ids'],
        target_states=ev['target_states'], target_valid=ev['target_valid'], node_observed=scene['observed'],
        split=SPLIT, grouping_kind='predicted', correspondence_policy='anchor_identity_first', return_artifact=True)['artifact']


def four_scene_replay(snapshot, scene, ev, saved, old, groups, peer_groups):
    latent = complete(snapshot, scene)
    parity, parity_calls = recur(snapshot['fusion'], latent, scene, peer_groups)
    tcm_tfm, new_calls = recur(snapshot['fusion'], latent, scene, groups)
    cap_mean = support_mean(scene, groups)
    supplied = {name: [dict(node_indices=g.tolist(), states=saved[name][gi])
                       for gi, g in enumerate(groups)] for name in ARMS[:2]}
    parity_eval = evaluate(parity, scene, ev)
    np.testing.assert_array_equal(parity_eval['group_valid'], old['group_valid'])
    pm = old['group_valid']
    lhs, rhs = parity_eval['group_states'], old['group_states']
    delta = [float(_distance(lhs[..., :2][pm[..., :2].all(-1)], rhs[..., :2][pm[..., :2].all(-1)]).max(initial=0)),
             float(abs(lhs[..., 2][pm[..., 2]]-rhs[..., 2][pm[..., 2]]).max(initial=0)),
             float(abs(_wrap(lhs[..., 3][pm[..., 3]]-rhs[..., 3][pm[..., 3]])).max(initial=0))]
    evaluated = {name: evaluate(out, scene, ev) for name, out in
                 dict(WS_TCM_TFM=tcm_tfm, WS_CAP128_MEAN=cap_mean, **supplied).items()}
    n = len(ev['target_ids'])
    align = evaluated['WS_TCM_TFM']['aligned_group_indices']
    for item in evaluated.values():
        np.testing.assert_array_equal(item['aligned_group_indices'], align)
    pred = {name: a['aligned_prediction'][:n] for name, a in evaluated.items()}
    valid = {name: a['aligned_prediction_valid'][:n] for name, a in evaluated.items()}
    common = np.logical_and.reduce(list(valid.values())) & ev['target_valid']
    region = {name: value & ev['target_valid'] for name, value in valid.items()}
    return dict(parity_max=dict(zip(('position_m', 'speed_mps', 'heading_deg'), delta, strict=True)),
                parity_within_tolerance=all(a <= b for a, b in zip(delta, (25, .01, .05), strict=True)),
                tcm_forward_batches=(len(latent)+15)//16, tfm_forward_calls=parity_calls+new_calls,
                groups=len(groups), declared=n, matched=int((align[:n] >= 0).sum()),
                reference=ev['target_valid'].sum((0, 1)).tolist(),
                per_arm_valid={name: mask.sum((0, 1)).tolist() for name, mask in region.items()}), \
        {name: errors(p, ev['target_states'], common) for name, p in pred.items()}, \
        metrics(pred, dict.fromkeys(pred, common), ev)


def summaries(vectors):
    return {arm: {a: _stats(np.concatenate([item[arm][a] for item in vectors])) for a in ATTRIBUTES}
            for arm in vectors[0]}


def run(runs, output, start):
    ids, components, _ = load_full_population(runs)
    peer_paths = verify_peer_inputs(runs, ids)
    c94 = read(runs/C94/'run.json')
    assert c94['status'] == 'COMPLETED' and c94['scene_ids'] == ids and not c94['test_opened']
    historical = read(runs/RUN113/'run.json')
    checkpoint = Path(historical['tfm_run'])/'checkpoint.pt'
    source = read(checkpoint.parent/'run.json')
    assert source['status'] == 'COMPLETED' and not source['test_opened'] and source['phase'] == 'tfm'
    snapshot = load_complete_snapshot(checkpoint, device='cpu', selected=False)
    assert snapshot['input_adapter'] == 'native_time_v1' and tuple(REPLAY) == ('38', '152', '203', '472')
    expected = read(runs/C126/'summary.json')
    threshold = expected['own']['FAR']['point']['position']['p99']
    rows, vectors, papers = [], defaultdict(list), defaultdict(list)
    replay_rows, replay_vectors, replay_paper = [], [], []
    for i, sid in enumerate(ids, 1):
        with np.load(runs/C94/'private'/f'{sid}.npz', allow_pickle=False) as a:
            saved = dict(a)
        with np.load(runs/PARENTS['full_data']/'private/motion'/SPLIT/f'scene-{sid}.npz', allow_pickle=False) as a:
            motion = {key: a[key] for key in ('node_ids', 'query_seconds', 'native_count', 'support',
                                             'nearest_seconds', 'cap128_support', 'cap128_nearest_seconds')}
        peer = load_scene_prediction(peer_paths[sid])
        scene, old = peer['scene'], peer['methods']['predicted']
        assert scene['split'] == SPLIT and scene['scene_id'] == sid and scene['component_id'] == components[sid]
        assert scene['physical_units']['state_speed_unit'] == scene['physical_units']['reference_speed_unit'] == 'mps'
        for a, b, c in (('nodes', 'node_ids', 'node_ids'), ('query', 'grid', 'query_seconds')):
            np.testing.assert_array_equal(saved[a], scene[b]); np.testing.assert_array_equal(saved[a], motion[c])
        ev = read_evaluation(runs, sid, saved['nodes'], saved['query'], scene_ids=ids)
        np.testing.assert_array_equal(ev['identities'], scene['identities'])
        prediction, scopes, sizes, groups, old_groups = masks_for(saved, ev, old, scene, motion)
        local, papers_local = collect(prediction, scopes, ev)
        for key, value in local.items():
            vectors[key].append(value)
        for key, value in papers_local.items():
            papers[key].append(value)
        rows.append(dict(scene=sid, component=components[sid], **sizes,
                         position_tail={region: int((x['FAR']['position'] >= threshold).sum())
                                        for region, x in local.items()},
                         scope_counts={region: {attr: len(value['FAR'][attr]) for attr in ATTRIBUTES}
                                       for region, value in local.items()}))
        if sid in REPLAY:
            audit, e, p = four_scene_replay(snapshot, scene, ev, saved, old, groups, old_groups)
            replay_rows.append(dict(scene=sid, **audit))
            replay_vectors.append(e)
            replay_paper.append(p)
        if i % 50 == 0:
            summary_state(output, start, status='RUNNING', processed_scenes=i)
    summary = {region: summaries(items) for region, items in vectors.items()}
    for arm in ARMS:
        for key in ATTRIBUTES:
            for field, val in expected['paired'][arm]['point'][key].items():
                np.testing.assert_allclose(summary['paired'][arm][key][field], val, rtol=1e-13, atol=1e-12)
    write(output/'summary.json', dict(scenes=len(rows), components=len(set(components.values())),
        targets={key: sum(r[key] for r in rows) for key in ('declared', 'both_claimed', 'same_group', 'full_count')},
        metrics=summary, paper={scope: {name: aggregate_paper_fusion([r[name] for r in items]) for name in ARMS}
                               for scope, items in papers.items()},
        tail=dict(threshold_m=threshold, counts={region: sum(r['position_tail'][region] for r in rows) for region in REGIONS}),
        replay=dict(rows=replay_rows, metrics=summaries(replay_vectors),
                    paper={name: aggregate_paper_fusion([p[name] for p in replay_paper]) for name in replay_paper[0]})))
    write(output/'scenes.json.gz', rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--time-policy', choices=['advisory'], required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(1)
    try:
        run(args.runs, args.output, start)
    except Exception as exc:
        summary_state(args.output, start, status='FAILED', error=repr(exc))
        raise
    summary_state(args.output, start, status='COMPLETED', optimizer_updates=0, tam_forwards=0,
                  raw_archive_reads=0, test_opened=False, processed_scenes=300, replay_scenes=len(REPLAY))


if __name__ == '__main__':
    main()
