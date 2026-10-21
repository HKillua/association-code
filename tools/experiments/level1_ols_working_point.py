"""Compute frozen Level1 V-select pure-OLS threshold without starting model training.

Same score definition and exact scene macro-F1 selector as the association step0.
Only reads identity labels and native measurements; no reference CSV or T_test.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.data.motion_evidence import DATA, prepare_motion_scenes
from association.evaluation.association import evaluate_scenes, select_threshold
from association.evaluation.fusion_controls import rms_haversine_scores
from association.utils.runtime import read_json, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', nargs='+', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--chunk', type=int, default=4096)
    args = p.parse_args(argv)
    manifest = read_json(args.manifest)
    if manifest.get('dataset_level') != 'Level1' or manifest.get('split_unit') != 'scene':
        raise ValueError('exact Level1 scene-split manifest required')
    if args.chunk < 1:
        raise ValueError('chunk must be positive')
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        scenes = prepare_motion_scenes(args.archive, manifest, 'V_select',
                                      motion_observation_policy='canonical_all')
        if [str(scene['scene_id']) for scene in scenes] != manifest['splits']['V_select']:
            raise ValueError('V scene order differs from manifest')
        records = []
        for scene in scenes:
            pairs = np.asarray(scene['pairs'])
            scores = np.empty(len(pairs), np.float64)
            positions = scene['motion']['states'][..., :2]
            for start in range(0, len(pairs), args.chunk):
                rms_negative_m = rms_haversine_scores(positions, pairs[start:start+args.chunk])
                scores[start:start+args.chunk] = -np.log1p(-rms_negative_m/1000.)
            records.append(dict(scene_id=scene['scene_id'], component_id=scene['component_id'],
                                scores=scores, pairs=pairs, labels=scene['labels'], sources=scene['sources']))
        selection = select_threshold(records)
        wp = dict(scope='ols-working-point-v1', dataset_level='Level1', synthetic=False,
                  motion_observation_policy='canonical_all', data=dict(DATA),
                  training_scene_ids=manifest['splits']['T_core'],
                  validation_scene_ids=manifest['splits']['V_select'],
                  score_semantics='-log1p(rms_haversine_m/1000)',
                  threshold=selection['threshold'], selection_split='V_select',
                  selection_rule='pure OLS full V scene macro-F1, before any network update')
        write_json(destination/'ols-working-point.json', wp)
        write_json(destination/'baseline.json', dict(selection=selection,
                   evaluation=evaluate_scenes(records, selection['threshold']),
                   scene_ids=[s['scene_id'] for s in scenes],
                   test_opened=False, reference_read=False, optimizer_updates=0))
    except BaseException as e:
        write_json(destination/'failure.json', dict(error=f'{type(e).__name__}: {e}',
                                                 optimizer_updates=0, test_opened=False))
        raise
    print('COMPLETED:', destination/'ols-working-point.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
