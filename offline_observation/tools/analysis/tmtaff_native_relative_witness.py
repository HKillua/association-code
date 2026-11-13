"""C125: audit raw observation support for the two frozen C122 exchanges."""

import argparse
from pathlib import Path
import resource
import time

import numpy as np

from association.data.archive import MTADArchive
from association.data.native import uniform_observation_indices
from association.evaluation.native_motion import _short_delta
from tools.analysis import held_fusion_population as held
from tools.analysis.tmtaff_equal_activation_position import EPS_M
from tools.analysis.tmtaff_neighbor_exchange_eligibility import C119, margin
from tools.analysis.tmtaff_true_pair_qualification import read_npz, write_json
from tools.analysis.tmtaff_unseen_position_readout import delta_m, distance


SCENE = '2887'
C107 = 'LEVEL1-OFFLINE-REGISTERED-HOLDOUT-FUSION-20261112-attempt-1'
C122 = 'LEVEL1-TMTAFF-NEIGHBOR-EXCHANGE-ELIGIBILITY-20261113-attempt-1'
MAX_BRACKET_SECONDS = 120.
METHODS = ('ALL', 'OMITTED', 'OTHER_FRAGMENT')
PREDICTIONS = ('MEAN', 'QUALITY_CLIP', 'PAIR_CLIP')


def bracket(times, values, take, query):
    position = np.full((len(query), 2), np.nan)
    span = np.full(len(query), np.inf)
    endpoints = np.full((len(query), 2), -1, dtype=int)
    if len(take):
        visible = times[take]
        for k in np.flatnonzero((query >= visible[0]) & (query <= visible[-1])):
            right = int(np.searchsorted(visible, query[k]))
            if visible[right] == query[k]:
                left = right
                position[k] = values[take[right]]
            else:
                left = right - 1
                weight = (query[k] - visible[left]) / (visible[right] - visible[left])
                a, b = values[take[[left, right]]]
                position[k] = a + weight * np.array([b[0] - a[0], _short_delta(b[1] - a[1])])
                position[k, 1] = (position[k, 1] + 180.) % 360. - 180.
            span[k] = visible[right] - visible[left]
            endpoints[k] = take[[left, right]]
    return position, span, endpoints


def native_inputs(tracks, pairs, query):
    times = [track['canonical_times'] for track in tracks]
    position = [track['canonical_values'][:, :2] for track in tracks]
    cap = [uniform_observation_indices(len(row), 128) for row in times]
    cap_keys = {(tracks[n]['source'], times[n][j], *position[n][j])
                for n in pairs.ravel() for j in cap[n]}
    fields = {name: [] for name in ('ALL', 'OMITTED', 'CAP')}
    for n, (t, xy, selected) in enumerate(zip(times, position, cap, strict=True)):
        all_rows = np.arange(len(t))
        omitted = np.setdiff1d(all_rows, selected, assume_unique=True)
        shared = np.asarray([(tracks[n]['source'], tm, *loc) in cap_keys
                             for tm, loc in zip(t, xy, strict=True)])
        for key, take in zip(fields, (all_rows, omitted, selected), strict=True):
            estimate, width, endpoints = bracket(t, xy, take, query)
            safe = (endpoints >= 0).all(axis=-1)
            overlap = np.zeros(len(query), dtype=int)
            ages = np.full((len(query), 2), np.nan)
            for q in np.flatnonzero(safe):
                row = endpoints[q]
                overlap[q] = shared[np.unique(row)].sum()
                ages[q] = t[row]-query[q]
            fields[key].append((estimate, width, overlap, ages))
    arrays = {f'{key}_{field}': np.asarray(items) for key, rows in fields.items()
              for field, items in zip(('position', 'width', 'shared', 'ages'), zip(*rows, strict=True), strict=True)}
    return dict(arrays, native_count=np.array([len(row) for row in times]),
                cap_count=np.array([len(row) for row in cap]),
                start_age_s=np.array([row[0]-query for row in times]),
                end_age_s=np.array([row[-1]-query for row in times]))


def choose_nodes(mode, source, pairs, identities, sources):
    if mode == 'OTHER_FRAGMENT':
        return [np.flatnonzero((identities == identities[pair[0]]) & (sources == source)
                              & ~np.isin(np.arange(len(sources)), pairs)) for pair in pairs]
    return [pair[sources[pair] == source] for pair in pairs]


def anchor_rows(frozen, mode, options, q):
    field = 'ALL' if mode == 'OTHER_FRAGMENT' else mode
    rows = []
    for candidates in options:
        if not len(candidates):
            return None
        width = frozen[f'{field}_width'][candidates, q]
        shared = frozen[f'{field}_shared'][candidates, q]
        qualifies = width <= MAX_BRACKET_SECONDS
        if mode != 'ALL':
            qualifies &= shared == 0
        take = candidates[qualifies]
        if not len(take):
            return None
        index = take[np.argmin(frozen[f'{field}_width'][take, q])]
        rows.append(index)
    return (frozen[f'{field}_position'][rows, q], frozen[f'{field}_ages'][rows, q],
            frozen[f'{field}_shared'][rows, q], frozen[f'{field}_width'][rows, q])


def angle(first, second):
    scale = np.linalg.norm(first)*np.linalg.norm(second)
    return float(np.rad2deg(np.arccos(np.clip(first @ second/scale, -1., 1.)))) if scale > 0 else None


def compare(frozen, forecast, options, q, mode, truth, center):
    row = anchor_rows(frozen, mode, options, q)
    if row is None:
        return None
    xy, ages, shared, width = row
    relative = delta_m(xy[0], xy[1], center)
    true_relative = delta_m(truth[0], truth[1], center)
    return dict(bracket_offsets_seconds=ages.tolist(), bracket_seconds=width.tolist(),
                shared_cap_points=shared.tolist(), relative_gap_m=float(distance(xy[0], xy[1])),
                reference_vector_angle_deg=angle(relative, true_relative), relative_vector_m=relative.tolist(),
                margin_m={arm: float(margin(forecast[arm][:, q], xy)) for arm in PREDICTIONS})


def evaluate(runs, frozen, forecast, registry, scene_index, output):
    query, pairs = forecast['query'], forecast['pairs']
    origin = runs / C107
    with np.load(origin/'private'/f'{SCENE}-evaluation.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['query'], query)
        np.testing.assert_array_equal(saved['nodes'], registry['node_ids'])
        identities = saved['identities']
        targets = {identity: i for i, identity in enumerate(saved['target_ids'])}
        ordered = [targets[identities[pair[0]]] for pair in pairs]
        truth, valid = saved['target_states'][ordered, :, :2], saved['target_valid'][ordered, :, :2].all(-1)
    assert (identities[pairs[:, 0]] == identities[pairs[:, 1]]).all()
    assert identities[pairs[0, 0]] != identities[pairs[1, 0]]
    c119 = read_npz(runs / C119 / 'public-errors.npz')
    np.testing.assert_array_equal(c119['valid'][c119['scene_index'] == scene_index], valid)
    public_mask = c119['pair_scene'] == scene_index
    common = valid.all(0)
    center = forecast['MEAN'][0, 0]
    expected = [row['common_index_within_scene'] for row in held.read(runs/C122/'summary.json')['reversals']
                if row['scene'] == SCENE and row['transition'] == 'lost']
    sources = registry['sources']
    source_names = np.unique(sources[pairs])
    arms = {(mode, source): choose_nodes(mode, source, pairs, identities, sources)
            for mode in METHODS for source in source_names}
    rows = []
    for i, q in enumerate(np.flatnonzero(common)):
        deltas = {arm: float(margin(forecast[arm][:, q], truth[:, q])) for arm in PREDICTIONS}
        for arm in PREDICTIONS:
            np.testing.assert_allclose(deltas[arm], c119[f'{arm}/margin'][public_mask][i], atol=1e-6, rtol=0.)
        gap = float(distance(truth[0, q], truth[1, q]))
        np.testing.assert_allclose(gap, c119['gap_m'][public_mask][i], atol=1e-6, rtol=0.)
        changed = bool((distance(forecast['QUALITY_CLIP'][:, q], forecast['PAIR_CLIP'][:, q]) > EPS_M).any())
        candidate = {f'{mode}_{source}': compare(frozen, forecast, options, q, mode, truth[:, q], center)
                     for (mode, source), options in arms.items()}
        for mode in METHODS:
            a, b = (candidate[f'{mode}_{source}'] for source in source_names)
            if a is not None and b is not None:
                a['other_source_angle_deg'] = b['other_source_angle_deg'] = angle(
                    np.array(a['relative_vector_m']), np.array(b['relative_vector_m']))
        focus = deltas['QUALITY_CLIP'] > EPS_M and deltas['PAIR_CLIP'] <= EPS_M
        assert focus == (i in expected)
        rows.append(dict(q=int(q), common_index=i, reference_gap_m=gap, reference_margin_m=deltas,
                         focus=focus, changed=changed, candidates=candidate))
    assert len(expected) == 2
    focus = [row for row in rows if row['focus']]
    controls = [row for row in rows if row['reference_gap_m'] <= 1000. and row['changed']
                and row['reference_margin_m']['QUALITY_CLIP'] > EPS_M
                and row['reference_margin_m']['PAIR_CLIP'] > EPS_M]
    diagnostics = {}
    for mode, source in arms:
        key = f'{mode}_{source}'
        active = [row['candidates'][key] for row in rows if row['candidates'][key] is not None]
        def flag(row):
            item = row['candidates'][key]
            return (item is not None and item['margin_m']['QUALITY_CLIP'] > EPS_M
                    and item['margin_m']['PAIR_CLIP'] <= EPS_M)
        diagnostics[key] = dict(per_target_node_count=[len(row) for row in arms[mode, source]],
            all_common_available=len(active), all_common_zero_cap_shared=sum(not any(r['shared_cap_points']) for r in active),
            focus_available=sum(row['candidates'][key] is not None for row in focus),
            focus_signed_transition_matches=sum(flag(row) for row in focus),
            focus_angle_lt_90_and_transition_matches=sum(flag(row)
                and row['candidates'][key]['reference_vector_angle_deg'] is not None
                and row['candidates'][key]['reference_vector_angle_deg'] < 90. for row in focus),
            near_changed_still_correct_available=sum(row['candidates'][key] is not None for row in controls),
            near_changed_still_correct_flags=sum(flag(row) for row in controls))
    def audit_track(node, q):
        meta = dict(native_count=int(frozen['native_count'][node]), cap_count=int(frozen['cap_count'][node]),
                    start_relative_seconds=float(frozen['start_age_s'][node, q]),
                    stop_relative_seconds=float(frozen['end_age_s'][node, q]))
        for mode in ('ALL', 'CAP', 'OMITTED'):
            item = frozen[f'{mode}_ages'][node, q]
            meta[mode] = (dict(offsets_seconds=item.tolist(),
                               shared_cap_points=int(frozen[f'{mode}_shared'][node, q]))
                          if np.isfinite(item).all() else None)
        return meta
    focus_tracks = [dict(q=row['q'], target_sources=[{str(sources[node]): audit_track(node, row['q'])
                   for node in pair} for pair in pairs]) for row in focus]
    result = dict(scene=SCENE, both_reference_valid=int(common.sum()), focus=len(focus),
        near_changed_still_correct=len(controls), arms=diagnostics, focus_rows=focus, focus_tracks=focus_tracks,
        empirical_new_holdout_warranted=any(value['focus_angle_lt_90_and_transition_matches'] == len(focus)
            and 4*value['near_changed_still_correct_flags'] <= len(controls)
            for name, value in diagnostics.items() if name.startswith('OMITTED_')),
        independent_instrument_data=False)
    write_json(output/'summary.json', result)
    # Geographical coordinates, offsets against absolute UTC and identities stay private.
    write_json(output/'public-q.json', rows)


def run(args):
    runs = args.runs
    configuration = held.read(runs/C119/'freeze.json')
    assert SCENE in held.read(runs/C107/'roster.json')['isolated']
    scene_index = [row['scene'] for row in configuration['rows']].index(SCENE)
    forecast = read_npz(runs/C119/'private'/f'{SCENE}.npz')
    query, pairs = forecast['query'], forecast['pairs']
    with np.load(runs/C107/'private'/f'{SCENE}.npz', allow_pickle=False) as saved:
        registry = {key: saved[key] for key in ('node_ids', 'sources', 'query_seconds', 'native_count')}
    np.testing.assert_array_equal(query, registry['query_seconds'])
    full = held.read(runs/held.PARENTS['full_data']/'run.json')
    with MTADArchive(full['archive']) as archive:
        held.validate_measurement_archive(runs, archive)
        tracks = held._canonical_tracks(archive._rows(SCENE, 'measurements'), SCENE,
                                       registry['node_ids'], registry['sources'])
        assert [(row['scene_id'], row['kind']) for row in archive.members_read] == [(SCENE, 'measurements')]
    observations = native_inputs(tracks, pairs, query)
    np.testing.assert_array_equal(observations['native_count'], registry['native_count'])
    for key in ('CAP', 'ALL'):
        np.testing.assert_array_equal(np.isfinite(observations[f'{key}_width'][pairs]).sum(axis=1), forecast['support'])
    write_json(args.output/'freeze.json', dict(scene=SCENE, max_bracket_seconds=MAX_BRACKET_SECONDS,
        query_count=len(query), native_members=[{'scene': SCENE, 'kind': 'measurements'}],
        targets_before_freeze=0, source_identity_lookup_before_freeze=0, prediction_or_action_writes=0))
    private = args.output/'private'
    private.mkdir(mode=0o700)
    np.savez_compressed(private/'anchors.npz', **observations)
    evaluate(runs, read_npz(private/'anchors.npz'), forecast, registry, scene_index, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    start = time.monotonic()
    try:
        run(args)
    except Exception as error:
        write_json(args.output/'run.json', dict(status='FAILED', error=repr(error), test_opened=False))
        raise
    write_json(args.output/'run.json', dict(status='COMPLETED', elapsed_seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        optimizer_updates=0, model_forwards=0, test_opened=False))


if __name__ == '__main__':
    main()
