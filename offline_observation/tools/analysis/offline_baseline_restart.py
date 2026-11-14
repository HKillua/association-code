"""C145 common-protocol association and four-state readout on frozen MTAD V300."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from association.data.archive import MTADArchive
from association.data.physical_units import KNOT_TO_MPS, SPEED_UNIT_POLICY
from association.evaluation.association import _aggregate, _counts
from association.evaluation.fusion_controls import attribute_errors, ordinary_fusion
from association.evaluation.paper_fusion_metrics import aggregate_paper_fusion
from association.evaluation.scene import evaluate_full_scene
from tools.analysis import held_fusion_population as population
from tools.analysis.offline_baseline_core import predict_arms

ARMS = ('WS', 'END_DTW', 'CV_DTW', 'END_SYNC')
ATTRIBUTES = ('latitude', 'longitude', 'speed', 'heading', 'position')
UNITS = ('degrees', 'degrees', 'mps', 'degrees', 'metres')
SCOPES = ('ITSA', 'MSTSA')


def write(path, payload):
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
    temporary = path.with_name(path.name + '.tmp')
    if path.suffix == '.gz':
        temporary.write_bytes(gzip.compress(serialized.encode(), mtime=0))
    else:
        temporary.write_text(serialized, encoding='utf-8')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def output_groups(membership, n):
    if membership.shape != (n,) or membership.dtype.kind not in 'iu':
        raise ValueError('expected an integer membership for each frozen measurement node')
    groups = {}
    for index, group in enumerate(membership):
        groups.setdefault(int(group), []).append(index)
    return list(groups.values())


def freeze_scene(scene, path):
    arms = predict_arms(scene)
    if set(arms) != set(ARMS[1:]):
        raise ValueError('candidate arms differ from the frozen contract')
    saved = {key: scene[key] for key in ('nodes', 'query', 'support', 'sources')}
    diagnostics = {}
    for name in ARMS:
        membership = scene['membership'] if name == 'WS' else np.asarray(arms[name]['membership'])
        groups = output_groups(membership, len(scene['nodes']))
        anchors = [group[0] for group in groups]
        fitted = ordinary_fusion(scene['native_motion'], scene['nearest'], scene['support'], groups)
        states = fitted['predictions']['mean_supported'][anchors]
        states[..., 2] *= KNOT_TO_MPS
        if name == 'WS':
            np.testing.assert_array_equal(states, scene['baseline'])
        else:
            result = arms[name]
            cross_edges = [(left, right) for a, b in result['mstsa_links'] for left in a for right in b]
            for scope, edges in zip(SCOPES, (result['itsa_links'], cross_edges), strict=True):
                saved[f'{name}_{scope}'] = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
            diagnostics[name] = result['audit']
        saved[f'{name}_membership'] = membership
        saved[f'{name}_states'] = states
    np.savez_compressed(path, **saved)
    return diagnostics


def edge_rows(saved, truth, name, sid, component):
    left, right = np.triu_indices(len(truth), 1)
    actual = truth[left] == truth[right]
    same = saved['sources'][left] == saved['sources'][right]
    membership = saved[f'{name}_membership']
    decided = membership[left] == membership[right]
    rows = {}
    for scope, mask in zip(SCOPES, (same, ~same), strict=True):
        counts = dict(scene_id=sid, component_id=component, pair_count=int(mask.sum()),
                      positive_count=int(actual[mask].sum()))
        rows[scope] = {**counts, **_counts(actual[mask], decided[mask])}
        if name != 'WS':
            edges = saved[f'{name}_{scope}']
            if (edges.size and ((edges < 0).any() or (edges >= len(truth)).any()
                    or (edges[:, 0] == edges[:, 1]).any())):
                raise ValueError('native decision contains invalid node keys')
            direct = {(min(a, b), max(a, b)) for a, b in edges.tolist()}
            used = np.fromiter(((a, b) in direct for a, b in zip(left, right, strict=True)),
                               dtype=bool, count=len(left))
            if len(direct) != len(edges) or (used & ~(mask & decided)).any():
                raise ValueError('native decisions must be unique, source-valid and inside final groups')
            rows[f'{scope}_direct'] = {**counts, **_counts(actual[mask], used[mask])}
    return rows


def residual_arrays(aligned):
    actual, reference = aligned['aligned_prediction'], aligned['aligned_truth']
    valid_actual = aligned['aligned_prediction_valid']
    valid_reference = aligned['aligned_truth_valid']
    delta = actual - reference
    delta[..., 1] = (delta[..., 1] + 180.) % 360. - 180.
    delta[..., 3] = (delta[..., 3] + 180.) % 360. - 180.
    distance = attribute_errors(actual, reference)['position']
    errors = np.concatenate((delta, distance[..., None]), axis=-1)
    errors = np.where(np.concatenate((valid_actual, valid_actual[..., :2].all(axis=-1, keepdims=True)), -1),
                      errors, np.nan)
    mask = np.concatenate((valid_reference, valid_reference[..., :2].all(axis=-1, keepdims=True)), -1)
    return errors, mask


def score_scene(path, runs, sid, component, ids):
    with np.load(path, allow_pickle=False) as saved:
        reference = population.read_evaluation(runs, sid, saved['nodes'], saved['query'], scene_ids=ids)
        return score_outputs(saved, reference, sid, component, ARMS)


def score_outputs(saved, reference, sid, component, arms):
    targets = len(reference['target_ids'])
    declarations = dict(node_identities=reference['identities'], target_identities=reference['target_ids'],
                        target_states=reference['target_states'], target_valid=reference['target_valid'],
                        node_observed=saved['support'], split=population.SPLIT, grouping_kind='predicted',
                        correspondence_policy='anchor_identity_first', return_artifact=True,
                        speed_unit_policy=SPEED_UNIT_POLICY)
    rows = {}
    for name in arms:
        membership = saved[f'{name}_membership']
        groups = output_groups(membership, len(reference['identities']))
        group_outputs = [dict(node_indices=group, states=states) for group, states in
                         zip(groups, saved[f'{name}_states'], strict=True)]
        result = evaluate_full_scene(group_outputs, **declarations)
        errors, available = residual_arrays(result['artifact'])
        rows[name] = dict(
            errors=errors[:targets], available=available[:targets],
            association=edge_rows(saved, reference['identities'], name, sid, component),
            paper=result['paper_fusion_metrics'], groups=len(groups),
            duplicates=result['duplicate_prediction_group_count'],
            unmatched_targets=len(result['unmatched_target_ids']),
        )
    raw_reference = rows[arms[0]]['available']
    for row in rows.values():
        np.testing.assert_array_equal(raw_reference, row['available'])
    common = raw_reference & np.logical_and.reduce([np.isfinite(row['errors']) for row in rows.values()])
    for row in rows.values():
        row['common'] = np.where(common, row['errors'], np.nan)
        row['own'] = np.where(raw_reference, row.pop('errors'), np.nan)
        row['available'] = raw_reference.sum(axis=(0, 1))
    return rows


def physical(arrays, available):
    combined = np.concatenate([value.reshape(-1, len(ATTRIBUTES)) for value in arrays])
    summary = {}
    for i, (name, unit) in enumerate(zip(ATTRIBUTES, UNITS, strict=True)):
        err = combined[np.isfinite(combined[:, i]), i]
        count = len(err)
        absolute = np.abs(err)
        denominator = int(available[i])
        summary[name] = dict(count=count, available=denominator,
            coverage=count / denominator if denominator else None, unit=unit,
            mae=float(absolute.mean()) if count else None,
            rmse=float(np.sqrt(np.mean(err**2))) if count else None,
            p99=float(np.percentile(absolute, 99.)) if count else None)
    return summary


def aggregate(records, *, paper=True, arms=ARMS):
    summary = {}
    for name in arms:
        rows = [row[name] for row in records]
        available = np.sum([row['available'] for row in rows], axis=0)
        summary[name] = dict(
            association={scope: _aggregate([row['association'][scope] for row in rows])
                         for scope in rows[0]['association']},
            fusion={scope: physical([row[scope] for row in rows], available) for scope in ('own', 'common')},
            **{key: sum(row[key] for row in rows) for key in ('groups', 'duplicates', 'unmatched_targets')},
        )
        if paper:
            summary[name]['paper_fusion'] = aggregate_paper_fusion(row['paper'] for row in rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-policy', choices=['advisory'], required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    wall = time.monotonic()
    output = args.output.resolve()
    runs = args.runs.resolve()
    private = output / 'private'
    output.mkdir()
    private.mkdir(mode=0o700)
    ids, components, archive_paths = population.load_full_population(runs)
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    write(output/'run.json', dict(status='PREDICTING', started=started.isoformat(), code=code,
          time_policy=args.time_policy, scenes=len(ids), split=population.SPLIT))
    diagnostics, hashes = {}, {}
    with MTADArchive(archive_paths) as archive:
        population.validate_measurement_archive(runs, archive)
        for index, sid in enumerate(ids):
            scene = population.load_scene(runs, sid, archive, scene_ids=ids)
            path = private/f'scene-{sid}.npz'
            diagnostics[sid] = freeze_scene(scene, path)
            hashes[sid] = digest(path)
            if (index + 1) % 25 == 0:
                print(f'prediction {index + 1}/{len(ids)}', flush=True)
        access = [(row['scene_id'], row['kind']) for row in archive.members_read]
        if access != [(sid, 'measurements') for sid in ids]:
            raise ValueError('unexpected archive scope during prediction')
    write(output/'freeze.json', dict(code=code, split=population.SPLIT,
          scene_ids=ids, components=components, parents=population.PARENTS,
          archive=archive_paths, members_read=len(access), sha256=hashes,
          purpose='exposed development, no fitting, measurement-only predictions', test_opened=False))
    records, per_scene = [], []
    for index, sid in enumerate(ids):
        path = private/f'scene-{sid}.npz'
        if digest(path) != hashes[sid]:
            raise ValueError('a frozen prediction changed before scoring')
        rows = score_scene(path, runs, sid, components[sid], ids)
        records.append(rows)
        per_scene.append(dict(scene_id=sid, component_id=components[sid],
                              arms=aggregate([rows], paper=False), candidate_diagnostics=diagnostics[sid]))
        if (index + 1) % 25 == 0:
            print(f'evaluation {index + 1}/{len(ids)}', flush=True)
    components_seen = Counter(components.values())
    largest = min(components_seen, key=lambda key: (-components_seen[key], key))
    by_component = {key: [row for row, sid in zip(records, ids, strict=True) if components[sid] == key]
                    for key in sorted(components_seen)}
    write(output/'scenes.json.gz', per_scene)
    write(output/'summary.json', dict(
        arms=aggregate(records), by_component={key: aggregate(rows, paper=False) for key, rows in by_component.items()},
        component_sizes=dict(components_seen), excluded_largest_component=largest,
        without_largest=aggregate([row for row, sid in zip(records, ids, strict=True)
                                   if components[sid] != largest]),
        scene_count=len(records), test_opened=False, updates=0))
    write(output/'run.json', dict(status='COMPLETED', exit_code=0, code=code, scenes=len(records),
          split=population.SPLIT, started=started.isoformat(), ended=datetime.now(timezone.utc).isoformat(),
          elapsed_seconds=time.monotonic()-wall, time_policy=args.time_policy, updates=0))


if __name__ == '__main__':
    main()
