"""C108: replay fixed physical residual only where both CAP128 sources support Q."""

import argparse
from collections import Counter
from pathlib import Path
import resource
import time

import numpy as np

from association.data.archive import MTADArchive
from tools.analysis import held_fusion_population as held
from tools.analysis import tmtaff_true_pair_qualification as fixed
from tools.analysis import tmtaff_unseen_position_readout as c105


C105 = 'LEVEL1-TMTAFF-UNSEEN-POSITION-READOUT-20261112-attempt-1'
RESIDUAL = 'PHYSICAL_RESIDUAL_Q'
ARMS = ('MEAN', RESIDUAL, 'DUAL_FALLBACK')
HEAD_KEYS = ('mean', 'scale', 'offset', 'intercept', 'beta', 'closure')


def load_head(origin):
    with np.load(origin / 'private/heads.npz', allow_pickle=False) as stored:
        head = {key: stored[f'{RESIDUAL}/{key}'] for key in HEAD_KEYS}
    fixed.require(head['beta'].shape == (22, 2) and all(np.isfinite(v).all() for v in head.values()),
                  'physical head shape or numeric state changed')
    fixed.require((head['scale'] > 0).all(), 'invalid C105 normalization')
    return head


def fallback(mean, residual, support):
    fixed.require(mean.shape == residual.shape == (*support.shape, 2), 'position/support shape changed')
    fixed.require(np.isin(support, (0, 1, 2)).all(), 'invalid two-source support count')
    dual = support == 2
    prediction = mean.copy()
    prediction[dual] = residual[dual]
    fixed.require(prediction[~dual].tobytes() == mean[~dual].tobytes()
                  and prediction[dual].tobytes() == residual[dual].tobytes(), 'position fallback changed a coordinate')
    return prediction


def save_scene(folder, row, mean, residual, support, query):
    gated = fallback(mean, residual, support)
    fixed.require(all(np.isfinite(p).all() and (np.abs(p[..., 0]) <= 90).all()
                      for p in (mean, residual, gated)), 'invalid frozen position')
    np.savez_compressed(folder / f"{row['sid']}.npz", pairs=row['pairs'], query=query, support=support,
                        MEAN=mean, PHYSICAL_RESIDUAL_Q=residual, DUAL_FALLBACK=gated)


def freeze_replay(origin, folder, scenes, index):
    metadata = held.read(origin / 'summary.json')
    assert metadata['eligible_scene_ids'] == list(scenes)
    prior = fixed.read_npz(origin / 'public-errors.npz')
    rows = []
    for i, sid in enumerate(scenes):
        evaluation = Path(index[sid]['evaluation'])
        registry = c105.registry(evaluation)
        saved = fixed.read_npz(origin / 'private' / f'pred-{sid}.npz')
        row = dict(sid=sid, scene={'node_ids': registry['node_ids']}, identities=registry['identities'],
                   evaluation=evaluation, pairs=saved['pairs'])
        np.testing.assert_array_equal(saved['query'], registry['query_seconds'])
        support = prior['support'][prior['scene_index'] == i]
        save_scene(folder, row, saved['MEAN'], saved[RESIDUAL], support, saved['query'])
        rows.append(row)
    return rows, prior


def freeze_new(runs, folder, scenes, index, head, archives, components):
    rows, roster = [], []
    with MTADArchive(archives) as archive:
        held.validate_measurement_archive(runs, archive)
        for sid in scenes:
            evaluation = Path(index[sid]['evaluation'])
            motion = fixed.read_npz(Path(index[sid]['motion']))
            row = c105.held_scene(archive, c105.registry(evaluation), sid, evaluation, motion)
            pairs = row['pairs']
            distinct = len(np.unique(row['identities'][pairs[:, 0]]))
            roster.append(dict(scene=sid, component=components[sid], pairs=len(pairs), targets=distinct))
            if not len(pairs):
                continue
            row.update(c105.physical(row['scene'], pairs))
            row['fraction'] = np.broadcast_to(np.linspace(-1., 1., len(row['query'])), row['support'].shape)
            residual = c105.predict(row, head, *c105.HEADS[RESIDUAL])
            save_scene(folder, row, row['mean'], residual, row['support'], row['query'])
            rows.append({key: row[key] for key in ('sid', 'pairs', 'evaluation', 'identities')}
                        | {'scene': {'node_ids': row['scene']['node_ids']}})
        fixed.require([(entry['scene_id'], entry['kind']) for entry in archive.members_read]
                      == [(sid, 'measurements') for sid in scenes], 'unexpected raw input')
    return rows, roster


def load_scene(folder, row):
    saved = fixed.read_npz(folder / f"{row['sid']}.npz")
    np.testing.assert_array_equal(saved['pairs'], row['pairs'])
    truth, valid = c105.labels(row, saved['query'])
    support = saved['support']
    distinct = len(np.unique(row['identities'][row['pairs'][:, 0]])) == 2
    common = valid.all(axis=0) if distinct else np.zeros(valid.shape[1], bool)
    center = saved['MEAN'][0, 0]
    y = truth[:, common]
    gap = c105.distance(y[0], y[1]) if distinct else np.empty(0)
    target_vector = c105.delta_m(y[0], y[1], center) if distinct else np.empty((0, 2))
    states = {}
    for name in ARMS:
        p = saved[name]
        err = c105.distance(p, truth)
        fixed.require(np.isfinite(err[valid]).all(), 'nonfinite joint-population error')
        if distinct:
            a, b = p[:, common]
            distance = c105.distance(a, b)
            vector = np.linalg.norm(c105.delta_m(a, b, center) - target_vector, axis=-1)
            margin = c105.distance(a, y[1]) + c105.distance(b, y[0]) - err[:, common].sum(axis=0)
        else:
            distance = vector = margin = np.empty(0)
        states[name] = dict(error=err, vector=vector, margin=margin, distance=distance)
    if distinct:
        actions = np.count_nonzero(support[:, common] == 2, axis=0)
    else:
        actions = np.empty(0, dtype=int)
    return support, valid, gap, actions, states


def changes(samples):
    return dict(better=int((samples < -1e-6).sum()), worse=int((samples > 1e-6).sum()),
                harm_gt_1km=int((samples > 1000.).sum()), worst_harm_m=float(samples.max(initial=0.)))


def contrast(stats, mask):
    margin = stats['margin'][mask]
    return dict(vector=fixed.statistics(stats['vector'][mask]), gap=fixed.statistics(stats['distance'][mask]),
                correct=int((margin > 1e-6).sum()), ties=int((np.abs(margin) <= 1e-6).sum()),
                incorrect=int((margin < -1e-6).sum()))


def evaluate(folder, rows, components, training_components, prior=None):
    loaded = [load_scene(folder, row) for row in rows]
    support, valid, gap, actions = (np.concatenate([record[i] for record in loaded]) for i in range(4))
    scene_index = np.repeat(np.arange(len(rows)), [len(record[0]) for record in loaded])
    pair_scene = np.repeat(np.arange(len(rows)), [len(record[2]) for record in loaded])
    names = np.asarray([row['sid'] for row in rows])
    groups = np.asarray([components[sid] for sid in names])
    largest = Counter(groups).most_common(1)[0][0]
    public = dict(scene_index=scene_index, pair_scene=pair_scene, support=support, valid=valid, gap_m=gap,
                  enabled_targets=actions)
    masks = dict(all=valid, dual=valid & (support == 2), single=valid & (support == 1),
                 unsupported=valid & (support == 0), no_largest=valid & (groups[scene_index] != largest)[:, None],
                 full_T_component_disjoint=valid & ~np.isin(groups[scene_index], list(training_components))[:, None])
    metrics = {}
    errors = np.concatenate([record[4]['MEAN']['error'] for record in loaded])
    small = gap <= c105.SMALL_M
    for arm in ARMS:
        st = {key: np.concatenate([rec[4][arm][key] for rec in loaded])
              for key in ('error', 'vector', 'margin', 'distance')}
        err = st['error']
        delta = err - errors
        metrics[arm] = {key: fixed.statistics(err[mask]) for key, mask in masks.items()}
        metrics[arm].update(paired={key: changes(delta[mask]) for key, mask in masks.items()},
                            scenes={str(name): fixed.statistics(err[valid & (scene_index == i)[:, None]])
                                    for i, name in enumerate(names)},
                            components={str(group): fixed.statistics(err[valid & (groups[scene_index] == group)[:, None]])
                                        for group in sorted(set(groups))},
                            contrast_all=contrast(st, np.ones(len(gap), bool)),
                            contrast_le_1km=contrast(st, small),
                            small_by_enabled={k: contrast(st, small & (actions == int(k))) for k in ('0', '1', '2')})
        public.update({f'{arm}/{key}': values for key, values in st.items()})
    if prior is not None:
        for key in ('scene_index', 'pair_scene', 'support', 'valid', 'gap_m'):
            np.testing.assert_array_equal(public[key], prior[key])
        for arm in ('MEAN', RESIDUAL):
            for actual, old in (('error', 'error_m'), ('vector', 'vector_m'), ('margin', 'assignment_gain_m')):
                np.testing.assert_allclose(public[f'{arm}/{actual}'], prior[f'{arm}/{old}'],
                                           atol=1e-8, rtol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(public['DUAL_FALLBACK/error'][support != 2], errors[support != 2])
    deltas = np.where(valid, public['DUAL_FALLBACK/error'] - errors, -np.inf)
    worst = sorted(np.argwhere(valid).tolist(), key=lambda ij: (-deltas[tuple(ij)], ij))[:10]
    tails = [dict(scene=str(names[scene_index[i]]), q=int(q), support=int(support[i, q]),
                  base_m=float(errors[i, q]), harm_m=float(deltas[i, q])) for i, q in worst]
    return dict(metrics=metrics, largest_component=str(largest), valid_points=int(valid.sum()),
                action_valid_points=int((valid & (support == 2)).sum()), all_queries=valid.size,
                all_action_queries=int((support == 2).sum()), worst_gate_harms=tails), public


def gates(result):
    gate, base = [result['metrics'][name] for name in ('DUAL_FALLBACK', 'MEAN')]
    small, old = gate['contrast_le_1km'], base['contrast_le_1km']
    return dict(full_mean_and_tail=gate['all']['mae'] < base['all']['mae'] and gate['all']['p99'] <= base['all']['p99'],
                dual_mean_and_tail=all(gate['dual'][k] <= base['dual'][k] for k in ('mae', 'p99')),
                no_largest=gate['no_largest']['mae'] < base['no_largest']['mae']
                and gate['no_largest']['p99'] <= base['no_largest']['p99'],
                small=bool(small['vector']['count'] and small['vector']['mae'] <= old['vector']['mae']
                           and small['correct'] >= old['correct']), zero_km_harm=gate['paired']['all']['harm_gt_1km'] == 0)


def run(args, start):
    origin = args.runs / C105
    source = held.read(origin / 'run.json')
    fixed.require(source['status'] == 'COMPLETED' and not source['test_opened'], 'completed C105 required')
    full = args.runs / held.PARENTS['full_data']
    manifest = held.read(Path(held.read(full / 'run.json')['manifest']))
    components = manifest['components']
    index = held.read(full / 'private/index.json')['V_select']
    train = set(manifest['splits']['T_core'])
    train_components = {components[sid] for sid in train}
    replay, _, archives = held.load_population(args.runs)
    excluded = {components[sid] for sid in (*fixed.SCENES, *replay)}
    new = [sid for sid in manifest['splits']['V_select'] if components[sid] not in excluded]
    fixed.require(len(new) == 57 and len({components[sid] for sid in new}) == 10, 'frozen new roster changed')
    fixed.require(not set(new + replay) & (train | set(manifest['splits']['T_test'])), 'development role changed')
    head = load_head(origin)
    folders = {name: args.output / 'private' / name for name in ('replay', 'new')}
    for folder in folders.values():
        folder.mkdir(mode=0o700, parents=True)
    old_rows, prior = freeze_replay(origin, folders['replay'], replay, index)
    new_rows, roster = freeze_new(args.runs, folders['new'], new, index, head, archives, components)
    fixed.write_json(args.output / 'freeze.json', dict(C105_roster=replay, new_roster=roster,
                     frozen_before_target_read=True, raw_measurements=len(new), coefficients=44, new_fit=False,
                     applied_if='CAP128 selected support == 2', historical_v_contract=len(new),
                     new_scenes_with_T_component=sum(components[s] in train_components for s in new)))
    results = {}
    for name, rows, previous in (('replay', old_rows, prior), ('new', new_rows, None)):
        result, public = evaluate(folders[name], rows, components, train_components, previous)
        result['gate'] = gates(result)
        result['eligible_scenes'] = [row['sid'] for row in rows]
        result['pairs'] = len(public['support'])
        results[name] = result
        np.savez_compressed(args.output / f'{name}-public-errors.npz', **public)
    fixed.write_json(args.output / 'summary.json', results)
    fixed.write_json(args.output / 'run.json', dict(status='COMPLETED', elapsed_seconds=time.monotonic()-start,
                     optimizer_updates=0, fit_operations=0, model_forward_calls=0, test_opened=False,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
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
