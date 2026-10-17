"""Evidence transport regressions: exact bytes, stale hashes, size and history."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.setup import collect_profile_evidence as collector
from tools.setup.collect_profile_evidence import LIMIT, collect


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'original'
        self.source.mkdir()
        self.acceptance = self.root / 'accept.log'
        self.output = self.root / 'new-evidence'
        self.profile = {'status': 'COMPLETED', 'peak_allocated_bytes': 123,
                        'peak_reserved_bytes': 256, 'phase_seconds': {'optimization': 10},
                        'step_phase_seconds': [{'encode_tracks': 2}], 'step_positive_pair_counts': [0]}
        self.bind()

    def bind(self):
        p = (json.dumps(self.profile, indent=4) + '\n').encode()
        (self.source / 'profile.json').write_bytes(p)
        h = hashlib.sha256(p).hexdigest()
        r = json.dumps({'profile_sha256': h}).encode()
        (self.source / 'run.json').write_bytes(r)
        self.acceptance.write_text(json.dumps({**self.profile, 'profile_sha256': h,
                                              'run_sha256': hashlib.sha256(r).hexdigest()}))

    def test_exact_originals_and_wrapper_without_execution(self):
        wrapper = self.root / 'wrapper.sh'
        wrapper.write_text('exit 99\n')
        result = collect(self.source, self.acceptance, self.output, [wrapper])
        self.assertEqual(result['status'], 'COLLECTED')
        self.assertTrue(result['profile_run_hash_link_verified'])
        for name in ('profile.json', 'run.json'):
            self.assertEqual((self.output / 'files' / name).read_bytes(), (self.source / name).read_bytes())
        self.assertEqual((self.output / 'files/attachment-01-wrapper.sh').read_text(), 'exit 99\n')

    def test_stale_hash_is_failure_with_original_preserved(self):
        original = self.source / 'profile.json'
        original.write_bytes(original.read_bytes() + b' ')
        before = original.read_bytes()
        result = collect(self.source, self.acceptance, self.output)
        self.assertEqual(result['status'], 'FAILED')
        self.assertEqual(original.read_bytes(), before)
        self.assertFalse((self.output / 'files/profile.json').exists())

    def test_missing_original_does_not_become_success(self):
        (self.source / 'run.json').unlink()
        result = collect(self.source, self.acceptance, self.output)
        self.assertEqual(result['status'], 'FAILED')
        self.assertTrue((self.output / 'collection.json').is_file())

    def test_large_original_returns_metadata_without_splitting(self):
        self.profile['large_field'] = 'x' * LIMIT
        self.bind()
        # Keep the independent acceptance summary small, as in the real task.
        accepted = json.loads(self.acceptance.read_text())
        accepted.pop('large_field')
        self.acceptance.write_text(json.dumps(accepted))
        result = collect(self.source, self.acceptance, self.output)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertFalse((self.output / 'files/profile.json').exists())
        row = next(r for r in result['files'] if r['name'] == 'profile.json')
        self.assertGreater(row['bytes'], LIMIT)
        self.assertEqual(row['status'], 'METADATA_ONLY_SIZE_LIMIT')

    def test_existing_output_is_never_overwritten(self):
        collect(self.source, self.acceptance, self.output)
        before = (self.output / 'collection.json').read_bytes()
        with self.assertRaises(FileExistsError):
            collect(self.source, self.acceptance, self.output)
        self.assertEqual((self.output / 'collection.json').read_bytes(), before)

    def test_summary_disagreement_is_rejected_even_with_correct_hashes(self):
        accepted = json.loads(self.acceptance.read_text())
        accepted['step_positive_pair_counts'] = [1]
        self.acceptance.write_text(json.dumps(accepted))
        self.assertEqual(collect(self.source, self.acceptance, self.output)['status'], 'FAILED')

    def test_growth_after_stat_cannot_bypass_size_limit(self):
        wrapper = self.root / 'wrapper.sh'
        wrapper.write_text('x')
        real_digest = collector.digest

        def grow_before_hash(path):
            if path == wrapper:
                wrapper.write_bytes(b'x' * (LIMIT + 1))
            return real_digest(path)

        with mock.patch.object(collector, 'digest', side_effect=grow_before_hash):
            result = collect(self.source, self.acceptance, self.output, [wrapper])
        self.assertEqual(result['status'], 'FAILED')
        self.assertFalse((self.output / 'files/attachment-01-wrapper.sh').exists())
        self.assertEqual(wrapper.stat().st_size, LIMIT + 1)


if __name__ == '__main__':
    unittest.main()
