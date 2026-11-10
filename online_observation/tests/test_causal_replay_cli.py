"""Check the 1 Hz entry point, archive permission boundary and queue accounting."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from online_observation import causal_replay, measurements
from online_observation.__main__ import main
from online_observation.stream import StreamState


HEADER = 'batch,source,time,lat,lon,vel,cou\n'


class CausalReplayCliTests(unittest.TestCase):
    def test_normal_entry_releases_only_current_rows_and_keeps_empty_ticks(self):
        with tempfile.TemporaryDirectory() as directory:
            measurements = Path(directory) / 'measurements.csv'
            output = Path(directory) / 'replay.jsonl'
            measurements.write_text(HEADER + 'future,B,3,0,0,10,90\n'
                                    'before,B,-0.1,0,0,10,90\n'
                                    'moving,A,1.2,0,0,10,90\n'
                                    'moving,A,2,0,1,10,90\n'
                                    'later,C,4.1,0,0,10,90\n', encoding='utf-8')
            args = ['causal-replay', '--measurements', str(measurements), '--start', '0', '--stop', '4',
                    '--stale-after-seconds', '3', '--output', str(output)]
            self.assertEqual(main(args), 0)
            original = output.read_text(encoding='utf-8')
            protocol, *ticks = map(json.loads, original.splitlines())
            self.assertEqual(protocol['arrival_clock'], 'ideal_measurement_time')
            self.assertFalse(protocol['warmup'])
            self.assertEqual(protocol['source_rows'], 5)
            self.assertEqual(protocol['scheduled_rows'], 3)
            self.assertEqual([row['result']['query_time'] for row in ticks], list(range(5)))
            self.assertEqual([row['received_rows'] for row in ticks], [0, 0, 2, 1, 0])
            self.assertEqual([len(row['result']['tracks']) for row in ticks], [0, 0, 1, 2, 2])
            moving = next(track for track in ticks[4]['result']['tracks'] if track['batch'] == 'moving')
            self.assertEqual(moving['observation_time'], 2)
            self.assertGreater(moving['longitude'], 1)

            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                main(args)
            self.assertEqual(result.exception.code, 2)
            self.assertEqual(output.read_text(encoding='utf-8'), original)

    def test_unschedulable_time_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            measurements = Path(directory) / 'bad.csv'
            output = Path(directory) / 'replay.jsonl'
            measurements.write_text(HEADER + 'x,A,nan,0,0,1,90\n', encoding='utf-8')
            with redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as result:
                main(['causal-replay', '--measurements', str(measurements), '--start', '0', '--stop', '0',
                      '--stale-after-seconds', '1', '--output', str(output)])
            self.assertEqual(result.exception.code, 2)
            self.assertIn('no schedulable time', stderr.getvalue())
            self.assertFalse(output.exists())

    def test_archive_only_opens_measurement_and_refuses_reclassified_test_scene(self):
        splits = dict(T_core=['0'], T_head=[], V_select=[], T_test=['1'])
        manifest = dict(dataset_level='Level1', all_scenes=['0', '1'], splits=splits,
                        caps={split: len(ids) for split, ids in splits.items()},
                        components={'0': 'a', '1': 'b'}, component_members={'a': ['0'], 'b': ['1']},
                        component_splits={'a': 'T_core', 'b': 'T_test'},
                        label_members={'0': {}, '1': {}}, development_scenes=['0'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path, roster = root / 'data.zip', root / 'manifest.json'
            roster.write_text(json.dumps(manifest), encoding='utf-8')
            with zipfile.ZipFile(archive_path, 'w') as archive:
                for scene in manifest['all_scenes']:
                    archive.writestr(f'Level1/关联表/关联结果-{scene}.csv', 'unreadable label')
                    archive.writestr(f'Level1/真实场景/场景-{scene}.csv', 'unreadable reference')
                    archive.writestr(f'Level1/量测场景/场景-{scene}.csv',
                                     (HEADER + '片段,A,0,0,0,1,90\n').encode('gb18030'))
            measurement_name = 'Level1/量测场景/场景-0.csv'
            with zipfile.ZipFile(archive_path) as archive:
                measurement_offset = archive.getinfo(measurement_name).header_offset
            payload = archive_path.read_bytes()
            header_end = payload.index(HEADER.encode('ascii'), measurement_offset) + len(HEADER)
            cuts = (0, measurement_offset + 11, header_end + 1, len(payload))
            parts = [root / f'part{i}.zip' for i in range(3)]
            for part, start, stop in zip(parts, cuts, cuts[1:]):
                part.write_bytes(payload[start:stop])
            opened = []
            read = zipfile.ZipFile.read

            def record_read(archive, member, *args, **kwargs):
                opened.append(member.filename)
                return read(archive, member, *args, **kwargs)

            with patch.object(measurements, '_LEVEL1_MANIFEST', roster), \
                    patch.object(zipfile.ZipFile, 'read', record_read):
                rows = measurements.read_archive_rows([archive_path], roster, '0')
                self.assertEqual(rows[0]['batch'], '片段')
                self.assertEqual(measurements.read_archive_rows(parts, roster, '0'), rows)
                self.assertEqual(opened, [measurement_name, measurement_name])
                with self.assertRaises((ValueError, zipfile.BadZipFile)):
                    measurements.read_archive_rows(parts[-1:], roster, '0')
                forged = root / 'alternate.json'
                alternate = {**manifest, 'splits': {**splits, 'T_core': ['1'], 'T_test': ['0']},
                             'component_splits': {'a': 'T_test', 'b': 'T_core'},
                             'development_scenes': ['1']}
                forged.write_text(json.dumps(alternate), encoding='utf-8')
                with self.assertRaises(ValueError):
                    measurements.read_archive_rows([archive_path], forged, '1')
                self.assertEqual(opened, [measurement_name, measurement_name])

    def test_public_entry_runs_without_offline_tree_or_site_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = Path(causal_replay.__file__).parent
            shutil.copytree(package, root / package.name,
                            ignore=shutil.ignore_patterns('tests', '__pycache__'))
            source = root / 'input.zip'
            output = root / 'output.jsonl'
            with zipfile.ZipFile(source, 'w') as archive:
                archive.writestr('Level1/量测场景/场景-0.csv', HEADER + 'x,A,0,0,0,1,90\n')
            roster = root / 'configs/data/mtad-level1-scene3000.json'
            roster.parent.mkdir(parents=True)
            roster.write_text(json.dumps(dict(
                dataset_level='Level1', all_scenes=['0'],
                splits=dict(T_core=['0'], T_head=[], V_select=[], T_test=[]),
            )), encoding='utf-8')
            isolated_entry = (
                'import runpy, sys\n'
                'sys.path.insert(0, sys.argv.pop(1))\n'
                'runpy.run_module("online_observation", run_name="__main__", alter_sys=True)\n'
            )
            completed = subprocess.run(
                [sys.executable, '-I', '-S', '-c', isolated_entry, str(root),
                 'causal-replay', '--archive', str(source), '--manifest', str(roster), '--scene', '0',
                 '--start', '0', '--stop', '1', '--stale-after-seconds', '2', '--output', str(output)],
                cwd=root, capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            _, *ticks = map(json.loads, output.read_text(encoding='utf-8').splitlines())
            self.assertEqual([row['result']['query_time'] for row in ticks], [0, 1])
            self.assertGreater(ticks[1]['result']['tracks'][0]['longitude'], 0)

    def test_virtual_serial_queue_accumulates_and_drains_without_changing_ticks(self):
        # Two 1.5s services accumulate a queue. Subsequent 0.2s services drain it.
        clock = iter((0, 1500000000, 1500000000, 3000000000, 3000000000, 3200000000,
                      3200000000, 3400000000, 3400000000, 3600000000))
        out = io.StringIO()
        causal_replay._write_ticks(out, StreamState(stale_after_seconds=10), [], 20, 24,
                                   clock_ns=lambda: next(clock))
        ticks = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual([row['result']['query_time'] for row in ticks], [20, 21, 22, 23, 24])
        for row, wait, latency in zip(ticks, (0., .5, 1., .2, 0.), (1.5, 2., 1.2, .4, .2)):
            self.assertAlmostEqual(row['timing']['modeled_queue_wait_seconds'], wait)
            self.assertAlmostEqual(row['timing']['modeled_result_latency_seconds'], latency)


if __name__ == '__main__':
    unittest.main()
