"""147 input lineage and measurement-only preparation, not a science result."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from association.data.archive import MTADArchive
from tests.fixtures.native_zip import make_native_zip_fixture
from tools.analysis import source_risk_full_benchmark as p


class FullBenchmarkTests(unittest.TestCase):
    def test_133_indirect_lineage_and_wrong_input_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = dict(source_system=root/'056', partition_run=root/'114')
            native = root/'127'
            native.mkdir()
            run = dict(status='COMPLETED', exit_code=0, **{k: str(v) for k, v in paths.items()})
            p.old.frozen.write(native/'run.json', run)
            pilot = dict(native_run=str(native), partition_run=str(paths['partition_run']))
            self.assertEqual(p.bind_lineage(paths, pilot, run), native)
            with self.assertRaisesRegex(ValueError, 'lineage'):
                p.bind_lineage(paths, pilot, dict(run, source_system=str(root/'other')))

    def test_native_loader_never_opens_reference_or_test(self):
        fixture = make_native_zip_fixture(self)
        read = zipfile.ZipFile.read
        def limited(archive, member, *args, **kwargs):
            name = member.filename if hasattr(member, 'filename') else str(member)
            if '真实场景' in name or name.endswith('-3.csv'):
                raise AssertionError('reference/test opened')
            return read(archive, member, *args, **kwargs)
        visible = dict(query_seconds=np.linspace(0., 30., 64))
        with patch.object(zipfile.ZipFile, 'read', limited), MTADArchive(fixture.path) as archive:
            native = p._record(archive.load_scene('2'), '2', 'V_select', 128)
            visible['node_ids'] = np.asarray(native['node_ids'], str)
            result = p.scene_motion(archive, fixture.manifest, '2', visible)
        self.assertEqual(result['states'].shape, (4, 64, 4))
        np.testing.assert_array_equal(result['query_seconds'], visible['query_seconds'])
        self.assertEqual(result['native_count'][result['sources'] == '9001'].tolist(), [4, 4])
        self.assertEqual(result['native_count'][result['sources'] == '9002'].tolist(), [2, 2])
