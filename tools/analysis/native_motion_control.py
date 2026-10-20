"""Evaluate three fixed, untrained native-position controls on complete V only.

Fixed Q64/cap128/time-unit20 and shortest-arc reference alignment; no model,
training/test input, parameter search, learned gate or speed/course prediction.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.data.archive import MTADArchive, validate_manifest
from association.data.native import _record
from association.data.reference_states import iter_reference_states
from association.data.targets import build_scene_fusion_targets
from association.data.timed import prepare_timed_scene
from association.evaluation.native_motion import METHODS, REGIONS, position_metrics, predict_native_positions, summarize_positions


DATA = dict(length=64, max_observations=128, time_unit_seconds=20.)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def evaluate_native_motion(archive_path, manifest, alignment, output):
    """Preserve all frozen V scenes and fragments; reference enters after prediction."""
    validate_manifest(manifest)
    scene_ids = manifest['splits']['V_select']
    if not scene_ids or not set(scene_ids).issubset(alignment):
        raise ValueError('explicit alignment must cover the complete nonempty frozen V selection')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'predictions').mkdir()
    started = time.monotonic()
    run = dict(status='RUNNING', scope='complete frozen V_select native position controls',
        archive=str(Path(archive_path).resolve()), scene_ids=scene_ids, data=DATA,
        reference_longitude_policy='shortest_arc', optimizer_updates=0, test_opened=False,
        inference_uses_identity=False, inference_uses_reference=False,
        scope_note='full-fragment offline information, not causal online prediction',
        completed_scenes=0)
    _write_json(output/'run.json', run)
    rows, per_scene, per_fragment, index = [], [], [], []
    try:
        # The reference generator advances only after each scene's predictions exist.
        with MTADArchive(archive_path) as archive, closing(iter_reference_states(
                archive_path, manifest, split='V_select', scene_ids=scene_ids)) as references:
            reference_reads = []
            for sid in scene_ids:
                native = _record(archive.load_scene(sid), manifest['components'][sid], 'V_select', DATA['max_observations'])
                labels = archive.read_labels(sid)
                query = None
                if not native['tracks']:
                    if not labels:
                        raise ValueError('empty scene has no label bounds: '+sid)
                    start, stop = min(row['t_s'] for row in labels), max(row['t_e'] for row in labels)
                    if stop <= start:
                        raise ValueError('empty scene requires positive-duration label bounds: '+sid)
                    query = np.linspace(start, stop, DATA['length'])
                scene = prepare_timed_scene(native, **DATA, query_times=query)
                control = predict_native_positions(scene['native_values'], scene['native_times'],
                                                   scene['native_observed'], scene['grid'])
                reference = next(references)
                if reference['scene_id'] != sid:
                    raise ValueError('reference scene order changed')
                policy = alignment[sid]
                if not isinstance(policy, dict) or set(policy) != {'time_offset_seconds', 'max_gap_seconds'}:
                    raise ValueError('provide original explicit offset/gap only: '+sid)
                target_ids = list(dict.fromkeys('-'.join(row['mmsi'].split('-')[:2]) for row in labels))
                targets = build_scene_fusion_targets(scene, reference, target_identity_ids=target_ids,
                                                     longitude_policy='shortest_arc', **policy)
                n, q = len(scene['node_ids']), DATA['length']
                truth, valid = np.full((n, q, 4), np.nan), np.zeros((n, q, 4), bool)
                for node, identity in enumerate(scene['identities']):
                    truth[node], valid[node] = targets[identity]['states'], targets[identity]['valid']
                methods = summarize_positions(control['predictions'], truth, valid,
                                               control['regions'], control['nearest_seconds'])
                absent = [identity for identity in target_ids if identity not in set(scene['identities'])]
                per_scene.append(dict(scene_id=sid, component_id=scene['component_id'],
                    fragment_count=n, label_target_count=len(target_ids),
                    reference_targets_without_input=absent,
                    reference_unavailable_fragment_count=int((~valid[..., :2].all(-1).any(1)).sum()),
                    reference_only_valid_position_queries=sum(int(targets[identity]['valid'][..., :2].all(-1).sum()) for identity in absent),
                    native_point_count=int(control['native_point_counts'].sum()),
                    zero_query_support_fragments=int((~control['regions']['inside_support'].any(1)).sum()),
                    one_query_support_fragments=int((control['regions']['inside_support'].sum(1) == 1).sum()),
                    latitude_clamp_counts={key: int(value.sum()) for key, value in control['latitude_clamp_counts'].items()},
                    ambiguous_native_longitude_half_turns=int(control['native_half_turn_counts'].sum()),
                    reference_alignment=dict(**policy, longitude_policy='shortest_arc'),
                    quarantine_audit=scene['audit'], methods=methods))
                for node in range(n):
                    take = slice(node, node+1)
                    local = {key: position_metrics(value[take], truth[take], valid[take],
                                                   control['regions']['all'][take])
                             for key, value in control['predictions'].items()}
                    per_fragment.append(dict(scene_id=sid, node_id=str(scene['node_ids'][node]),
                        identity=str(scene['identities'][node]), source=str(scene['sources'][node]),
                        native_point_count=int(control['native_point_counts'][node]),
                        query_support_count=int(control['regions']['inside_support'][node].sum()),
                        latitude_clamp_counts={key: int(value[node]) for key, value in control['latitude_clamp_counts'].items()},
                        methods=local))
                filename = 'positions-'+sid+'.npz'
                np.savez_compressed(output/'predictions'/filename,
                    scene_id=np.asarray(sid), node_ids=np.asarray(scene['node_ids'], dtype=str),
                    identities=np.asarray(scene['identities'], dtype=str), sources=np.asarray(scene['sources'], dtype=str),
                    query_seconds=scene['grid'], native_values=scene['native_values'][..., :2],
                    native_times=scene['native_times'], native_observed=scene['native_observed'],
                    target=truth, target_valid=valid,
                    nearest_seconds=control['nearest_seconds'],
                    endpoint_slope_degrees_per_second=control['endpoint_slope_degrees_per_second'],
                    ols_slope_degrees_per_second=control['ols_slope_degrees_per_second'],
                    **control['predictions'], **{'region_'+key: value for key, value in control['regions'].items()})
                index.append(dict(scene_id=sid, path=filename, fragment_count=n))
                rows.append(dict(predictions=control['predictions'], target=truth, target_valid=valid,
                                 regions=control['regions'], nearest_seconds=control['nearest_seconds'],
                                 sources=np.asarray(scene['sources'], dtype=str)))
                reference_reads.append(dict(scene_id=sid, member=reference['member']))
                del reference
                run.update(completed_scenes=len(rows), elapsed_seconds=time.monotonic()-started)
                _write_json(output/'run.json', run)
            observation_reads = list(archive.members_read)
        pooled = dict(predictions={name: np.concatenate([row['predictions'][name] for row in rows]) for name in METHODS},
            target=np.concatenate([row['target'] for row in rows]),
            target_valid=np.concatenate([row['target_valid'] for row in rows]),
            regions={name: np.concatenate([row['regions'][name] for row in rows]) for name in REGIONS},
            nearest_seconds=np.concatenate([row['nearest_seconds'] for row in rows]))
        sources = np.concatenate([row['sources'] for row in rows])
        overall = summarize_positions(**pooled)
        by_source = {}
        for source in sorted(set(sources)):
            selected = sources == source
            by_source[str(source)] = summarize_positions(
                {name: value[selected] for name, value in pooled['predictions'].items()},
                pooled['target'][selected], pooled['target_valid'][selected],
                {name: value[selected] for name, value in pooled['regions'].items()}, pooled['nearest_seconds'][selected])
        report = dict(status='COMPLETED', scope=run['scope'], data=DATA, scene_count=len(rows),
            fragment_count=len(sources), query_count=len(sources)*DATA['length'],
            native_point_count=sum(row['native_point_count'] for row in per_scene),
            fragments_excluded_for_query_support=0, fit_fragment_count=len(sources), fallback_fragment_count=0,
            zero_query_support_fragments=sum(row['zero_query_support_fragments'] for row in per_scene),
            one_query_support_fragments=sum(row['one_query_support_fragments'] for row in per_scene),
            reference_unavailable_fragment_count=sum(row['reference_unavailable_fragment_count'] for row in per_scene),
            reference_only_valid_position_queries=sum(row['reference_only_valid_position_queries'] for row in per_scene),
            latitude_clamp_counts={name: sum(row['latitude_clamp_counts'][name] for row in per_scene) for name in METHODS},
            ambiguous_native_longitude_half_turns=sum(row['ambiguous_native_longitude_half_turns'] for row in per_scene),
            error_scope='point-pooled available position intersection over every input fragment; repeated identities retained; '
                        'reference-only target identities listed separately, never invented as input fragments',
            position_control='shared native shortest-arc linear interpolation within support; outside, original measured endpoint '
                             '+ signed real seconds times hold-zero / first-to-last slope / all selected native OLS slope',
            method_scope='fixed full-fragment offline controls; no learned parameters, validation fitting or online claim',
            optimizer_updates=0, test_opened=False, overall=overall, by_source=by_source,
            per_scene=per_scene, per_fragment=per_fragment,
            observation_members_read=observation_reads, reference_members_read=reference_reads,
            elapsed_seconds=time.monotonic()-started)
        _write_json(output/'predictions'/'index.json', index)
        _write_json(output/'native-motion.json', report)
        run.update(status='COMPLETED', fragment_count=len(sources), query_count=report['query_count'],
                   elapsed_seconds=time.monotonic()-started)
        _write_json(output/'run.json', run)
        return report
    except BaseException as exc:
        run.update(status='FAILED', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.monotonic()-started)
        _write_json(output/'run.json', run)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'manifest', 'alignment', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    manifest, alignment = (json.loads(Path(path).read_text(encoding='utf-8')) for path in (args.manifest, args.alignment))
    report = evaluate_native_motion(args.archive, manifest, alignment, args.output)
    print(json.dumps({key: report[key] for key in ('status', 'scene_count', 'fragment_count', 'query_count')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
