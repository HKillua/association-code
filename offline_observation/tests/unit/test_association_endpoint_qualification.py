"""Synthetic software checks for the C148 fixed-budget diagnostic."""
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools.analysis import association_endpoint_qualification as c148


class EndpointQualificationTests(unittest.TestCase):
    def test_full_endpoints_and_zero_quota_preserve_the_raw_solution(self):
        edges = [(0, 2, 1.), (0, 3, 4.), (1, 2, 5.), (1, 3, 2.)]
        for quota, expected in ((2, [(0, 2), (1, 3)]), (0, [])):
            raw, result, eligible, _, _, residual = c148.compare_block([0, 1], [2, 3], edges, quota)
            self.assertEqual(raw, expected)
            self.assertEqual(result, raw)
            self.assertFalse(eligible)
            self.assertIsNone(residual)

    def test_raw_and_potential_objective_against_enumeration(self):
        rng = random.Random(72)
        strict_changes = 0
        left, right = [0, 3, 5], [1, 2, 4, 6]
        for _ in range(80):
            edges = [(i, j, rng.uniform(0, 5000)) for i in left for j in right if rng.random() < .7]
            feasible = [sorted(c148.pair(i, j) for i, j, _ in chunk) for chunk in combinations(edges, 2)
                        if len({v for i, j, _ in chunk for v in (i, j)}) == 4]
            if not feasible:
                continue
            raw, result, eligible, row, col, residual = c148.compare_block(left, right, edges, 2)
            distances = {c148.pair(i, j): Fraction(d) for i, j, d in edges}
            extra = {c148.pair(i, j): Fraction(float(row[left.index(i)])) + Fraction(float(col[right.index(j)]))
                     for i, j, _ in edges}

            def cost(choice, use_potential=False):
                return sum(distances[p] + (extra[p] if use_potential else 0) for p in choice)

            self.assertTrue(eligible)
            self.assertTrue(0 <= residual < .05)
            self.assertEqual(cost(raw), min(cost(p) for p in feasible))
            self.assertEqual(cost(result, True), min(cost(p, True) for p in feasible))
            if raw != result:
                strict_changes += 1
                self.assertNotEqual(c148.endpoints(raw), c148.endpoints(result))
        self.assertGreater(strict_changes, 0)

    def test_synthetic_run_freezes_every_scene_before_scoring_and_keeps_zeroes(self):
        query = np.arange(64, dtype=float)
        nodes = np.array(['n0', 'n1', 'n2'])
        sources = np.array(['9001', '9002', '9002'])
        values = np.tile([38., 120., 10., 0.], (6, 1))
        times = np.tile([0., 63.], 3)
        offsets = np.array([0, 2, 4, 6])
        support = np.ones((3, 64), dtype=bool)
        states = np.tile(values[0], (3, 64, 1))
        scene = dict(query=query, sources=sources, nodes=nodes, support=support, states=states,
                     values=values, native=(query, times, values[:, :2], offsets))

        class Archive:
            def __init__(self, paths):
                self.members_read = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def read_measurements(archive, motion_path, sid):
            archive.members_read.append(dict(scene_id=sid, kind='measurements'))
            return scene

        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            private = work / c148.DATA / 'private'
            for kind in ('motion', 'evaluation'):
                (private / kind / 'T_core').mkdir(parents=True)
            out = work / 'result'
            for sid in ('01', '02'):
                (private / 'motion' / 'T_core' / f'scene-{sid}.npz').write_bytes(b'motion stub')
                np.savez(private / 'evaluation' / 'T_core' / f'scene-{sid}.npz',
                         node_ids=nodes, query_seconds=query, identities=np.array(['t', 't', 'other']))

            real_read_identity = c148.read_identity

            def only_after_freeze(path, scene_nodes, scene_query):
                got = json.loads((out / 'freeze.json').read_text())
                self.assertEqual(list(got['prediction_sha256']), ['01', '02'])
                return real_read_identity(path, scene_nodes, scene_query)

            population = {}, ['archive stub'], {'A': '01', 'B': '02'}, {'A': 11, 'B': 2}, {}
            args = SimpleNamespace(output=out, runs=work, manifest=work / 'manifest.json')
            with (patch.object(c148, 'bind_population', return_value=population),
                  patch.object(c148, 'MTADArchive', Archive), patch.object(c148, 'validate_archive'),
                  patch.object(c148, 'build_scene', side_effect=read_measurements),
                  patch.object(c148, 'read_identity', side_effect=only_after_freeze)):
                c148.run(args)
            status = json.loads((out / 'run.json').read_text())
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual((status['status'], status['predictions'], status['scored']), ('COMPLETED', 2, 2))
            self.assertEqual(summary['full']['same_ship_missing_support_lost'], 0)
            self.assertEqual(summary['full']['eligible'], 2)
            self.assertEqual(summary['full']['changed'], 0)
            self.assertEqual(summary['without_largest_fit_component']['scenes'], 1)
            self.assertFalse(summary['qualifies_for_larger_fit_diagnostic'])


if __name__ == '__main__':
    unittest.main()
