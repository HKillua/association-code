"""C107: untouched scene replay and same-readout supplier-contamination attribution."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from association.data.alignment import (
    MAX_GAP_SECONDS, TEMPORAL_CLOCK_POLICY, _ClockArchive, derive_scene_alignment,
)
from association.data.legacy_dataset import validate_archive
from association.data.motion_evidence import DATA, scene_motion_evidence
from association.data.native import _record
from association.data.physical_units import KNOT_TO_MPS
from association.data.reference_states import iter_reference_states
from association.data.targets import build_scene_fusion_targets
from association.data.timed import prepare_timed_scene
from association.evaluation.association import _aggregate
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.group_prediction import predict_group_outputs
from association.evaluation.mixture_opportunity import _distance
from association.evaluation.paper_fusion_metrics import aggregate_paper_fusion
from association.models.evidence_association import EvidenceAssociation
from tools.analysis import association_registered_supplier_exclusion as c104
from tools.analysis.association_frozen_pr_audit import RUNS
from tools.analysis.offline_association_tail import coverage, infer, load_checkpoint, read_identity, read_motion, summary_state, write
from tools.analysis.same_source_swap_pilot import read


C104 = 'LEVEL1-OFFLINE-REGISTERED-SUPPLIER-EXCLUSION-20261112-attempt-1'
ASSOCIATION = ('WS', 'STRICT', 'REG')
ORACLE = 'DROP_ADDED_WRONG'
FEATURES = ('latitude', 'longitude', 'speed', 'heading', 'position')
REGIONS = ('all', 'changed_members', 'added_wrong', 'new_pure_wrong', 'same_ship_recovered')


def holdout(manifest):
    components = manifest['components']
    sealed = {components[sid] for sid in manifest['splits']['T_test']}
    developed = {components[sid] for split in ('T_core', 'V_select') for sid in manifest['splits'][split]}
    roster = [sid for sid in manifest['all_scenes']
              if manifest['scene_splits'][sid] == 'unused' and components[sid] not in sealed]
    isolated = [sid for sid in roster if components[sid] not in developed]
    c104.c98.require((len(roster), len({components[s] for s in roster}), len(isolated),
                      len({components[s] for s in isolated})) == (72, 44, 25, 24), 'frozen holdout population differs')
    supplement = deepcopy(manifest)
    supplement['splits']['V_select'].extend(roster)
    supplement['caps']['V_select'] += len(roster)
    supplement['scene_splits'].update(dict.fromkeys(roster, 'V_select'))
    supplement.pop('split_counts', None)
    supplement.pop('dependence', None)
    supplement['selection_note'] = 'C107 development appendix only; adds exactly the declared original-unused scenes.'
    return roster, isolated, supplement


def models(runs):
    teacher, checkpoint = load_checkpoint(runs/RUNS['056']/'association/final.pt', 'cpu')
    c104.c98.require(checkpoint['update'] == 1000 and teacher.motion_observation_policy == 'canonical_all',
                     '056 frozen model differs')
    weights = torch.load(runs/c104.c98.C95/'private/heads.pt', map_location='cpu', weights_only=True)
    head = EvidenceAssociation()
    head.load_state_dict(weights['M_BCE'], strict=True)
    return teacher, checkpoint['normalizer'], {'M_BCE': head.eval()}


def mean_groups(motion, partition, support=None):
    flags = motion['support'] if support is None else support
    groups = [g.tolist() for g in c104.c98.group_nodes(partition)]
    return predict_group_outputs(motion['states'], motion['nearest_seconds'], flags, motion['sources'],
                                 groups, method='mean_supported')['group_outputs']


def partition_and_predict(sid, scene, raw, motion, args):
    model, calibration, thresholds = args
    n = len(scene.tracks)
    pairs = np.column_stack(np.triu_indices(n, 1))
    scores = infer(motion, pairs, model[0], model[1], 'cpu', model[2])
    if n:
        old, blocks, margin = c104.catalog_from_scores(motion, pairs, scores, thresholds)
        evidence = c104.build_evidence(sid, motion, old, blocks, margin, None, measurement_rows=raw)
        states, choices = c104.choose_scene(old, blocks, motion['support'], evidence, calibration)
        strict = c104.c98.materialize(old, blocks, [c104.c98.choose(b, 'SUPPLIER') for b in blocks])
        groups = dict(WS=old, STRICT=strict, REG=states['REG_MIX'])
    else:
        blocks, evidence, choices = [], [], {}
        groups = {name: np.empty(0, dtype=np.int64) for name in ASSOCIATION}
    outputs = {key: mean_groups(motion, part) for key, part in groups.items()}
    return groups, outputs, evidence, choices, blocks, scores


def support_changes(old, new, identities, support):
    was, now = old[:, None] == old, new[:, None] == new
    identity = identities[:, None] == identities
    view = lambda mask: (mask.astype(np.int32) @ support.astype(np.int32)) > 0
    old_any, new_any = view(was), view(now)
    old_true, new_true = view(was & identity), view(now & identity)
    return dict(added_wrong=view(now & ~was & ~identity), new_pure_wrong=~old_any & new_any & ~new_true,
                same_ship_recovered=~old_true & new_true, same_ship_lost=old_true & ~new_true,
                true_to_only_wrong=old_true & new_any & ~new_true,
                changed_members=np.broadcast_to((was != now).any(axis=1)[:, None], support.shape))


def stripped_support(old, new, identities, support):
    """Evaluation-only removal of donors added to an incorrect anchor's new group."""
    result = support.copy()
    for group in c104.c98.group_nodes(new):
        anchor = group[0]
        wrong_new = group[(old[group] != old[anchor]) & (identities[group] != identities[anchor])]
        result[wrong_new] = False
    return result


def masks_for_targets(partition, tags, alignment):
    groups = c104.c98.group_nodes(partition)
    assigned = np.flatnonzero(alignment >= 0)
    anchors = [groups[alignment[i]][0] for i in assigned]
    result = {}
    for key, mask in tags.items():
        values = np.zeros((len(alignment), mask.shape[1]), bool)
        values[assigned] = mask[anchors]
        result[key] = values
    return result


def feature_errors(pred, target, valid):
    diff = pred - target
    diff[..., [1, 3]] = (diff[..., [1, 3]] + 180.) % 360. - 180.
    position_valid = valid[..., :2].all(axis=-1)
    position = np.full(position_valid.shape, np.nan)
    position[position_valid] = _distance(pred[..., :2][position_valid], target[..., :2][position_valid])
    return dict(zip(FEATURES, (*np.moveaxis(np.where(valid, np.abs(diff), np.nan), -1, 0), position), strict=True))


def fusion(motion, groups, outputs, ev):
    old, new = groups['WS'], groups['REG']
    ids, support = ev['identities'], motion['support']
    tags = support_changes(old, new, ids, support)
    oracle = mean_groups(motion, new, stripped_support(old, new, ids, support))
    actual = dict(outputs, **{ORACLE: oracle})
    reports, arrays = evaluate_fusion_methods(actual, prediction_speed_units=dict.fromkeys(actual, 'kn'),
                    node_identities=ids, target_ids=ev['target_ids'], target_states=ev['target_states'],
                    target_valid=ev['target_valid'], support=support, nearest_seconds=motion['nearest_seconds'],
                    target_speed_unit='mps', split='V_select')
    nt = len(ev['target_ids'])
    coverage_rows = {arm: {key: len(reports[arm][key]) if isinstance(reports[arm][key], list) else reports[arm][key]
                           for key in ('target_count', 'predicted_group_count', 'duplicate_group_count', 'unmatched_target_ids')
                           if key in reports[arm]} for arm in actual}
    errors = {}
    for arm in actual:
        item = arrays[arm]
        valid = item['aligned_prediction_valid'][:nt] & ev['target_valid']
        errors[arm] = feature_errors(item['aligned_prediction'][:nt], ev['target_states'], valid)
    target_tags = masks_for_targets(new, tags, arrays['REG']['aligned_group_indices'][:nt])
    # All affected oracle readings must originate from a newly co-grouped wrong donor.
    reg = arrays['REG']['aligned_prediction'][:nt]
    drop = arrays[ORACLE]['aligned_prediction'][:nt]
    changed = (reg != drop) & np.isfinite(reg) & np.isfinite(drop)
    np.testing.assert_array_equal(changed & ~target_tags['added_wrong'][..., None], False)
    regions = dict(all=np.ones((nt, support.shape[1]), bool), **{key: target_tags[key] for key in REGIONS[1:]})
    matrices = {region: {feat: np.column_stack([errors[arm][feat][mask] for arm in actual])
                        for feat in FEATURES} for region, mask in regions.items()}
    point_rows = []
    for ti, qi in zip(*np.nonzero(target_tags['added_wrong']), strict=True):
        point_rows.append(dict(target_ordinal=int(ti), q=int(qi), pure_new=bool(target_tags['new_pure_wrong'][ti, qi]),
                               errors={name: {feature: float(value[ti, qi]) if np.isfinite(value[ti, qi]) else None
                                              for feature, value in err.items()} for name, err in errors.items()}))
    return dict(nodes={key: int(flag.sum()) for key, flag in tags.items()},
                output_targets={key: int(flag.sum()) for key, flag in target_tags.items()},
                coverage=coverage_rows, paper={name: reports[name]['paper_fusion_metrics'] for name in actual},
                wrong_points=point_rows), matrices


def attribute_stats(values):
    return dict(n=len(values), mae=float(values.mean()) if len(values) else None,
                p99=float(np.quantile(values, .99)) if len(values) else None,
                max=float(values.max()) if len(values) else None)


def pool_fusion(rows, matrices, arm_names):
    result = dict(nodes=Counter(), output_targets=Counter(), coverage={arm: Counter() for arm in arm_names})
    for row in rows:
        for key in ('nodes', 'output_targets'):
            result[key].update(row[key])
        for arm in arm_names:
            result['coverage'][arm].update(row['coverage'][arm])
    result['paper'] = {arm: aggregate_paper_fusion([row['paper'][arm] for row in rows]) for arm in arm_names}
    result['errors'] = {}
    for region in REGIONS:
        result['errors'][region] = {}
        for feat in FEATURES:
            x = np.concatenate([scene[region][feat] for scene in matrices], axis=0) if matrices else np.empty((0, len(arm_names)))
            shared = np.isfinite(x).all(axis=1)
            row = dict(common=int(shared.sum()),
                       own={name: attribute_stats(x[np.isfinite(x[:, j]), j]) for j, name in enumerate(arm_names)},
                       common_metrics={name: attribute_stats(x[shared, j]) for j, name in enumerate(arm_names)})
            reg, drop = (arm_names.index(arm) for arm in ('REG', ORACLE))
            delta = x[shared, reg]-x[shared, drop]
            row['REG_minus_drop'] = dict(harm=int((delta > 1e-9).sum()), rescue=int((delta < -1e-9).sum()),
                                         mean=float(delta.mean()) if len(delta) else None,
                                         max=float(delta.max()) if len(delta) else None)
            result['errors'][region][feat] = row
    return result


def association_metrics(groups, support, sources, ids):
    left, right = np.triu_indices(len(ids), 1)
    truth, same = ids[left] == ids[right], sources[left] == sources[right]
    old = groups['WS']
    was = old[left] == old[right]
    result = {}
    for arm, membership in groups.items():
        now = membership[left] == membership[right]
        cuts = was & ~now
        unsafe = (support[left] != support[right]).any(axis=1)
        result[arm] = dict(decoded=c104.c98.edge_rows(truth, now, same),
                           removed=c104.c98.edge_rows(truth, cuts, same),
                           added=c104.c98.edge_rows(truth, now & ~was, same),
                           coverage=coverage(old, membership, ids, support),
                           true_supply_cut=int((cuts & unsafe & truth).sum()),
                           false_supply_cut=int((cuts & unsafe & ~truth).sum()))
    return result


def pool_association(rows, arms=ASSOCIATION):
    return {arm: dict(**{stage: {scope: _aggregate([r[arm][stage][scope] for r in rows]) for scope in c104.c98.SCOPES}
                         for stage in ('decoded', 'added', 'removed')},
                      coverage={key: sum(r[arm]['coverage'][key] for r in rows) for key in rows[0][arm]['coverage']},
                      **{key: sum(r[arm][key] for r in rows) for key in ('true_supply_cut', 'false_supply_cut')})
            for arm in arms}


def gate(pooled, baseline='WS'):
    old, new = (pooled[arm]['decoded'] for arm in (baseline, 'REG'))
    return (all(new['mstsa'][key] < old['mstsa'][key] for key in ('fn', 'fp'))
            and all(new['itsa'][key] <= old['itsa'][key] for key in ('fn', 'fp')))


def summarize(rows, matrices, *, arms=ASSOCIATION, baseline='WS'):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['component']].append(row['association'])
    biggest = max(grouped, key=lambda key: (len(grouped[key]), key))
    full = pool_association([row['association'] for row in rows], arms)
    without = pool_association([row['association'] for row in rows if row['component'] != biggest], arms)
    components = {key: pool_association(group, arms) for key, group in grouped.items()}
    reg = full['REG']
    no_harm = not (reg['true_supply_cut'] or any(reg['coverage'][key] for key in
                   ('identities_lost', 'same_ship_lost', 'duplicate_increase_scenes')))
    joint_components = sum(gate(group, baseline) for group in components.values())
    association_pass = no_harm and gate(full, baseline) and gate(without, baseline) and joint_components >= 2
    fusion_totals = pool_fusion([row['fusion'] for row in rows], matrices, (*arms, ORACLE))
    metrics = fusion_totals['errors']['all']
    fusion_pass = association_pass and all(value['common'] > 0
        and all(value['common_metrics']['REG'][key] <= value['common_metrics'][baseline][key] for key in ('mae', 'p99'))
        and value['own']['REG']['n'] >= value['own'][baseline]['n'] for value in metrics.values())
    eligibility = Counter()
    for row in rows:
        eligibility.update(row['exclusion'])
    return dict(scenes=len(rows), components=len(components), largest=biggest, association=full,
                without_largest=without, components_association=components,
                joint_components=joint_components, association_pass=association_pass,
                fusion_pass=fusion_pass, clock_unsupported=[row['scene'] for row in rows if not row['clock_supported']],
                exclusion=eligibility, fusion=fusion_totals)


def evaluation(scene, labels, saved, reference):
    clock = derive_scene_alignment(scene, labels, (), reference, clock_policy=TEMPORAL_CLOCK_POLICY)
    valid_clock = clock['status'] == 'CLOCK_HYPOTHESIS_SUPPORTED'
    target_ids = list(dict.fromkeys(row['mmsi'].rsplit('-', 1)[0] for row in labels))
    query = saved['query_seconds']
    ids = saved['identities']
    if valid_clock:
        stub = dict(scene_id=scene.scene_id, split='V_select', identities=ids, grid=query)
        targets = build_scene_fusion_targets(stub, reference, target_identity_ids=target_ids, longitude_policy='shortest_arc',
                      time_offset_seconds=clock['time_offset_seconds'], max_gap_seconds=MAX_GAP_SECONDS)
        states = np.asarray([targets[t]['states'] for t in target_ids], float).reshape(-1, len(query), 4)
        valid = np.asarray([targets[t]['valid'] for t in target_ids], bool).reshape(-1, len(query), 4)
    else:
        states = np.full((len(target_ids), len(query), 4), np.nan)
        valid = np.zeros(states.shape, bool)
    return dict(target_ids=np.asarray(target_ids, str), identities=ids, target_states=states, target_valid=valid), valid_clock


def history(runs, manifest, index, output):
    root = runs/C104/'private/V_select'
    baseline = runs/c104.C98/'private/V_select'
    rows, matrices = [], []
    for sid in manifest['splits']['V_select']:
        entry = index[sid]
        motion = read_motion(entry)
        ids = read_identity(entry, motion)
        with np.load(baseline/f'{sid}.npz', allow_pickle=False) as saved:
            old = saved['WS']
        with np.load(root/f'{sid}.npz', allow_pickle=False) as saved:
            new = saved['REG_MIX']
        if np.array_equal(old, new) or not support_changes(old, new, ids, motion['support'])['new_pure_wrong'].any():
            continue
        with np.load(entry['evaluation'], allow_pickle=False) as saved:
            ev = {key: saved[key] for key in ('target_ids', 'identities', 'target_states', 'target_valid')}
        arms = dict(WS=old, REG=new)
        row, metrics = fusion(motion, arms, {key: mean_groups(motion, group) for key, group in arms.items()}, ev)
        rows.append(dict(scene=sid, **row))
        matrices.append(metrics)
    total = pool_fusion(rows, matrices, ('WS', 'REG', ORACLE))
    c104.c98.require(total['nodes']['new_pure_wrong'] == 13, 'original C104 pure wrong support count changed')
    write(output/'historical-13.json', dict(scenes=rows, pooled=total))


def run(args, start):
    manifest = read(args.manifest)
    roster, isolated, supplement = holdout(manifest)
    c104.c98.require(read(args.runs/C104/'run.json')['status'] == 'COMPLETED', 'C104 must be frozen')
    calibration = read(args.runs/C104/'calibration.json')
    thresholds = read(args.runs/c104.c98.C95/'summary.json')['fixed_thresholds']
    c104.c98.require({k: v['upper'] for k, v in calibration.items()} ==
                      dict(same=None, motion=4.0, raw=4205.475, registered=420.168), 'C104 gate differs')
    full = args.runs/c104.c98.DATA
    archives = read(full/'run.json')['archive']
    private = args.output/'private'
    private.mkdir()
    write(private/'supplement-manifest.json', supplement)
    write(args.output/'roster.json', dict(new=roster, isolated=isolated, sealed_opened=False))
    model = models(args.runs)
    seen = {}
    with _ClockArchive(archives) as archive:
        validate_archive(archive, supplement)
        for sid in roster:
            scene = archive.load_scene(sid)
            labels = archive.read_labels(sid)
            native = _record(scene, manifest['components'][sid], 'V_select', 128, split_unit='scene')
            bounds = None if scene.tracks else np.linspace(min(row['t_s'] for row in labels),
                                                           max(row['t_e'] for row in labels), 64)
            prepared = prepare_timed_scene(native, **DATA, query_times=bounds)
            motion = scene_motion_evidence(prepared, motion_observation_policy='canonical_all')
            motion['node_ids'] = np.asarray(prepared['node_ids'], str)
            groups, outputs, evidence, choices, blocks, scores = partition_and_predict(
                sid, scene, archive.raw_measurements, motion, (model, calibration, thresholds))
            np.savez_compressed(private/f'{sid}.npz', **motion, identities=np.asarray(prepared['identities'], str),
                                **{f'group_{key}': val for key, val in groups.items()},
                                **{f'prediction_{key}': np.asarray([o['states'] for o in val]).reshape(-1, 64, 4)
                                   for key, val in outputs.items()}, **{f'score_{key}': val for key, val in scores.items()})
            write(private/f'{sid}.json', dict(evidence=evidence, choices=choices, blocks=blocks))
            seen[sid] = scene, labels
            archive._label_cache.pop(sid, None)
            summary_state(args.output, start, status='RUNNING', phase='freeze', frozen_scenes=len(seen))
        audit = Counter(row['kind'] for row in archive.members_read)
        c104.c98.require(audit == {'measurements': 72, 'labels': 72}
                         and {row['scene_id'] for row in archive.members_read} == set(roster), 'raw roster differs')
    write(args.output/'freeze.json', dict(association_scenes=len(seen), measurement_members=audit['measurements'],
                         label_members=audit['labels'], reference_before_prediction=0, calibrated_again=False))
    summary_state(args.output, start, status='RUNNING', phase='evaluation')
    rows, matrices = [], []
    for reference, sid in zip(iter_reference_states(archives, supplement, split='V_select', scene_ids=roster), roster, strict=True):
        scene, labels = seen.pop(sid)
        with np.load(private/f'{sid}.npz', allow_pickle=False) as saved:
            motion = {key: saved[key] for key in ('states', 'support', 'sources', 'nearest_seconds')}
            groups = {key: saved[f'group_{key}'] for key in ASSOCIATION}
            outputs = {key: [dict(node_indices=group.tolist(), states=values)
                             for group, values in zip(c104.c98.group_nodes(groups[key]), saved[f'prediction_{key}'], strict=True)]
                       for key in ASSOCIATION}
            ev, clock_supported = evaluation(scene, labels, saved, reference)
            np.savez_compressed(private/f'{sid}-evaluation.npz', **ev, query=saved['query_seconds'], nodes=saved['node_ids'])
        selected = read(private/f'{sid}.json')
        support = motion['support']
        ids = ev['identities']
        counts, _ = c104.summarize_evidence(selected['evidence'], ids, calibration)
        associations = association_metrics(groups, support, motion['sources'], ids)
        fusion_row, metric = fusion(motion, groups, outputs, ev)
        rows.append(dict(scene=sid, component=manifest['components'][sid], isolated=sid in isolated,
                         clock_supported=clock_supported, exclusion=dict(counts), association=associations,
                         fusion=fusion_row))
        matrices.append(metric)
    independent = [i for i, row in enumerate(rows) if row['isolated']]
    shared = [i for i, row in enumerate(rows) if not row['isolated']]
    write(args.output/'summary.json', dict(all=summarize(rows, matrices),
          isolated=summarize([rows[i] for i in independent], [matrices[i] for i in independent]),
          historically_shared=summarize([rows[i] for i in shared], [matrices[i] for i in shared])))
    with gzip.open(args.output/'scenes.json.gz', 'wt') as stream:
        json.dump(rows, stream, separators=(',', ':'), allow_nan=False)
    history(args.runs, manifest, read(full/'private/index.json')['V_select'], args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir()
    start = time.monotonic()
    try:
        run(args, start)
    except Exception as error:
        summary_state(args.output, start, status='FAILED', error=repr(error))
        raise
    summary_state(args.output, start, status='COMPLETED', optimizer_updates=0, test_opened=False)


if __name__ == '__main__':
    main()
