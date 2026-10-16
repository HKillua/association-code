"""Contract regression tests using temporary Git repositories and fake packages."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools.setup import verification_contract as contract


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VerificationContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'repo'
        self.root.mkdir()
        for name, content in {
            'association/__init__.py': '', 'association/model.py': 'value = 1\n',
            'tests/test_model.py': '# fixture\n', 'configs/smoke.json': '{"steps": 1}\n',
            'tools/setup/run.sh': '#!/bin/sh\n', 'tools/setup/check.py': '# fixture\n',
            'pyproject.toml': '[project]\nname="mtad-association-pilot"\n', 'README.md': 'Initial documentation\n',
        }.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Contract Test')
        self.git('config', 'user.email', 'contract@example.invalid')
        self.git('add', '.')
        self.git('commit', '-qm', 'initial execution inputs')
        self.cpu_commit = self.git('rev-parse', 'HEAD').strip()
        self.cpu = self.root / 'receipts' / 'cpu'
        self.cpu.mkdir(parents=True)
        self.environment = {
            'schema_version': 1, 'python': '3.11.7', 'python_executable': '/field/bin/python',
            'prefix': '/field', 'base_prefix': '/base', 'numpy': '1.26.4',
            'torch': '2.5.1+cu121', 'cuda_build': '12.1',
            'package_sources': {'association': str(self.root / 'association/__init__.py'),
                                'numpy': '/field/numpy/__init__.py', 'torch': '/field/torch/__init__.py'},
            'pip_freeze': ['numpy==1.26.4', 'torch==2.5.1+cu121'],
            'sitecustomize': {'loaded': False, 'path': '/field/sitecustomize.py', 'exists': False, 'sha256': None},
            'thread_settings': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
            'torch_precision': {'float32_matmul_precision': 'highest',
                                'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': True},
        }
        self.imported = {name: deepcopy(self.environment[name]) for name in contract.IMPORT_ENVIRONMENT_FIELDS}
        self.imported.update(visible_gpu_count=0, code_sha256='fixture-code-hash')
        for name, content in {'setup_exit_code.txt': '0\n', 'setup_wrapper_exit_code.txt': '0\n',
                              'setup_completed.txt': 'completed\n', 'actual_commit.txt': self.cpu_commit + '\n'}.items():
            (self.cpu / name).write_text(content)
        for name in contract.REQUIRED_CHECKS:
            (self.cpu / (name + '_exit_code.txt')).write_text('0\n')
        write_json(self.cpu / 'source_files.json', contract.source_manifest(self.root))
        write_json(self.cpu / 'environment.json', self.environment)
        (self.cpu / 'pip_freeze.txt').write_text('\n'.join(self.environment['pip_freeze']) + '\n')
        write_json(self.cpu / 'import_check.json', self.imported)
        self.report = {'status': 'COMPLETED', 'actual_commit': self.cpu_commit,
                       'checks': {name: 0 for name in contract.REQUIRED_CHECKS},
                       'import_check': deepcopy(self.imported), 'environment_rechecked': True,
                       'source_files_sha256': digest(self.cpu / 'source_files.json'),
                       'environment_sha256': digest(self.cpu / 'environment.json')}
        self.save_report()

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.root), *args], capture_output=True,
                              text=True, check=True, timeout=10).stdout

    def save_report(self):
        write_json(self.cpu / 'validation.json', self.report)

    def validate(self, environment=None):
        return contract.validate_cpu_receipt(self.root, self.cpu,
                                             self.environment if environment is None else environment)

    def test_documentation_commit_preserves_cpu_reuse(self):
        (self.root / 'README.md').write_text('Updated documentation\n')
        (self.root / 'HandOver').mkdir()
        (self.root / 'HandOver/receipt.md').write_text('CPU verified\n')
        self.git('add', 'README.md', 'HandOver')
        self.git('commit', '-qm', 'record successful CPU verification')
        evidence = self.validate()
        self.assertEqual(evidence['cpu_actual_commit'], self.cpu_commit)
        self.assertEqual(evidence['current_commit'], self.git('rev-parse', 'HEAD').strip())
        self.assertNotEqual(evidence['current_commit'], self.cpu_commit)
        self.assertEqual(evidence['validation_sha256'], digest(self.cpu / 'validation.json'))
        self.assertEqual(evidence['source_files_sha256'], digest(self.cpu / 'source_files.json'))
        self.assertEqual(evidence['environment_sha256'], digest(self.cpu / 'environment.json'))

    def editable_metadata(self, root=None, version='0.1.0', editable=True):
        direct_url = {'url': (self.root if root is None else root).resolve().as_uri(),
                      'dir_info': {'editable': editable}}
        return types.SimpleNamespace(metadata={'Name': 'mtad-association-pilot'}, version=version,
                                     read_text=lambda name: json.dumps(direct_url) if name == 'direct_url.json' else None)

    def bind_freeze(self, lines):
        self.environment['pip_freeze'] = contract.normalize_freeze(self.root, lines)
        (self.cpu / 'pip_freeze.txt').write_text('\n'.join(lines) + '\n')
        write_json(self.cpu / 'environment.json', self.environment)
        self.report['environment_sha256'] = digest(self.cpu / 'environment.json')
        self.save_report()

    def test_verified_project_editable_head_changes_after_docs_commit_are_allowed(self):
        def project_line(commit):
            return ('-e git+ssh://git@github.com/Hkillua/association-code.git@' + commit +
                    '#egg=mtad_association_pilot')
        with mock.patch.object(contract.metadata, 'distribution', return_value=self.editable_metadata()):
            original = [project_line(self.cpu_commit), 'numpy==1.26.4', 'torch==2.5.1+cu121']
            self.bind_freeze(original)
            expected = f'-e {self.root.resolve().as_uri()}#egg=mtad-association-pilot==0.1.0'
            self.assertIn(expected, self.environment['pip_freeze'])
            (self.root / 'README.md').write_text('Document a completed CPU run\n')
            self.git('add', 'README.md')
            self.git('commit', '-qm', 'documentation-only receipt')
            current_commit = self.git('rev-parse', 'HEAD').strip()
            self.assertNotEqual(current_commit, self.cpu_commit)
            current = deepcopy(self.environment)
            current['pip_freeze'] = contract.normalize_freeze(self.root, [project_line(current_commit), *original[1:]])
            evidence = self.validate(current)
            self.assertEqual(evidence['cpu_actual_commit'], self.cpu_commit)
            self.assertEqual(evidence['current_commit'], current_commit)

    def test_external_editable_vcs_pin_is_preserved_and_change_is_rejected(self):
        before = '-e git+https://example.invalid/other.git@' + 'a' * 40 + '#egg=external_dependency'
        after = before.replace('a' * 40, 'b' * 40)
        with mock.patch.object(contract.metadata, 'distribution', return_value=self.editable_metadata()):
            self.bind_freeze([before, 'numpy==1.26.4', 'torch==2.5.1+cu121'])
            self.assertIn(before, self.environment['pip_freeze'])
            changed = deepcopy(self.environment)
            changed['pip_freeze'] = contract.normalize_freeze(self.root, [after, 'numpy==1.26.4', 'torch==2.5.1+cu121'])
            self.assertIn(after, changed['pip_freeze'])
            with self.assertRaisesRegex(ValueError, 'Environment changed'):
                self.validate(changed)

    def test_unverified_or_different_root_project_editable_retains_vcs_sha(self):
        line = '-e git+ssh://git@github.com/Hkillua/association-code.git@' + 'a' * 40 + '#egg=mtad_association_pilot'
        for distribution in (self.editable_metadata(Path(self.temporary.name)),
                             self.editable_metadata(editable=False)):
            with self.subTest(metadata=distribution):
                with mock.patch.object(contract.metadata, 'distribution', return_value=distribution):
                    self.assertEqual(contract.normalize_freeze(self.root, [line]), [line])
        with mock.patch.object(contract.metadata, 'distribution', side_effect=contract.metadata.PackageNotFoundError):
            self.assertEqual(contract.normalize_freeze(self.root, [line]), [line])
        wrong_path = '-e ' + Path(self.temporary.name).resolve().as_uri() + '#egg=mtad_association_pilot'
        with mock.patch.object(contract.metadata, 'distribution', return_value=self.editable_metadata()):
            self.assertEqual(contract.normalize_freeze(self.root, [wrong_path]), [wrong_path])

    def test_project_installed_metadata_version_drift_is_rejected(self):
        line = '-e git+ssh://git@github.com/Hkillua/association-code.git@' + self.cpu_commit + '#egg=mtad_association_pilot'
        with mock.patch.object(contract.metadata, 'distribution', return_value=self.editable_metadata()):
            self.bind_freeze([line, 'numpy==1.26.4', 'torch==2.5.1+cu121'])
        with mock.patch.object(contract.metadata, 'distribution', return_value=self.editable_metadata(version='0.2.0')):
            changed = deepcopy(self.environment)
            changed['pip_freeze'] = contract.normalize_freeze(self.root, [line, 'numpy==1.26.4', 'torch==2.5.1+cu121'])
            self.assertNotEqual(changed['pip_freeze'], self.environment['pip_freeze'])
            with self.assertRaisesRegex(ValueError, 'Environment changed'):
                self.validate(changed)

    def test_source_modification_addition_and_deletion_are_rejected(self):
        path = self.root / 'association/model.py'
        original = path.read_text()
        for operation in ('modify', 'add', 'delete'):
            with self.subTest(operation=operation):
                added = self.root / 'tools/setup/new_check.py'
                if operation == 'modify':
                    path.write_text('value = 2\n')
                elif operation == 'add':
                    added.write_text('# new execution file\n')
                else:
                    path.unlink()
                with self.assertRaisesRegex(ValueError, 'manifest changed'):
                    self.validate()
                path.write_text(original)
                if added.exists():
                    added.unlink()

    def test_configuration_and_dependency_pin_files_are_rejected(self):
        for name in ('configs/smoke.json', 'pyproject.toml'):
            with self.subTest(path=name):
                path = self.root / name
                original = path.read_text()
                path.write_text(original + '\n# changed\n')
                with self.assertRaisesRegex(ValueError, 'manifest changed'):
                    self.validate()
                path.write_text(original)

    def test_environment_package_pin_and_runtime_settings_changes_are_rejected(self):
        changes = [('torch', '2.6.0'), ('pip_freeze', ['numpy==2.0.0']),
                   ('thread_settings', {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '1'}),
                   ('torch_precision', {'float32_matmul_precision': 'high'}),
                   ('sitecustomize', {'loaded': True, 'path': '/field/sitecustomize.py', 'sha256': 'changed'})]
        for name, value in changes:
            with self.subTest(field=name):
                changed = deepcopy(self.environment)
                changed[name] = value
                with self.assertRaisesRegex(ValueError, 'Environment changed'):
                    self.validate(changed)

    def test_cpu_failure_and_missing_completion_are_rejected(self):
        for name in ('setup_exit_code.txt', 'setup_wrapper_exit_code.txt', 'tests_exit_code.txt'):
            with self.subTest(file=name):
                path = self.cpu / name
                path.write_text('1\n')
                with self.assertRaises(ValueError):
                    self.validate()
                path.write_text('0\n')
        (self.cpu / 'setup_completed.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or unreadable'):
            self.validate()

    def test_receipt_commit_status_and_check_inconsistencies_are_rejected(self):
        original = deepcopy(self.report)
        changes = [('actual_commit', 'f' * 40), ('status', 'RUNNING'),
                   ('checks', {'pip_check': 0}), ('checks', {**self.report['checks'], 'tests': False})]
        for name, value in changes:
            with self.subTest(field=name, value=value):
                self.report = {**deepcopy(original), name: value}
                self.save_report()
                with self.assertRaises(ValueError):
                    self.validate()
        self.report = original
        self.save_report()
        write_json(self.cpu / 'import_check.json', {**self.imported, 'numpy': '2.0'})
        with self.assertRaisesRegex(ValueError, 'import_check and validation report disagree'):
            self.validate()

    def test_import_receipt_cannot_disagree_with_bound_environment(self):
        changed = {**self.imported, 'prefix': '/another-environment'}
        write_json(self.cpu / 'import_check.json', changed)
        self.report['import_check'] = changed
        self.save_report()
        with self.assertRaisesRegex(ValueError, 'import_check disagrees with environment snapshot'):
            self.validate()

    def test_post_cpu_recheck_and_snapshot_hash_bindings_are_required(self):
        for name, value in (('environment_rechecked', False), ('source_files_sha256', '0' * 64),
                            ('environment_sha256', '0' * 64)):
            with self.subTest(field=name):
                original = self.report[name]
                self.report[name] = value
                self.save_report()
                with self.assertRaises(ValueError):
                    self.validate()
                self.report[name] = original
                self.save_report()

    def test_pip_freeze_evidence_must_match_normalized_snapshot(self):
        (self.cpu / 'pip_freeze.txt').write_text('\n torch==2.5.1+cu121 \n numpy==1.26.4\n')
        self.validate()
        (self.cpu / 'pip_freeze.txt').write_text('numpy==2.0\n')
        with self.assertRaisesRegex(ValueError, 'pip_freeze.txt disagrees'):
            self.validate()

    def test_legacy_receipt_requires_fresh_cpu_without_migration(self):
        (self.cpu / 'environment.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Fresh CPU verification required.*legacy ENV004'):
            self.validate()
        self.assertFalse((self.cpu / 'environment.json').exists())

    def test_manifest_scope_excludes_docs_and_bytecode_but_rejects_escaping_symlink(self):
        before = contract.source_manifest(self.root)
        (self.root / 'tests/__pycache__').mkdir()
        (self.root / 'tests/__pycache__/cached.pyc').write_bytes(b'ignored')
        (self.root / 'README.md').write_text('different docs')
        self.assertEqual(contract.source_manifest(self.root), before)
        outside = Path(self.temporary.name) / 'outside.py'
        outside.write_text('value = 3\n')
        (self.root / 'association/external.py').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'outside repository'):
            contract.source_manifest(self.root)

    def fake_packages(self):
        prefix = Path(self.temporary.name) / 'field'
        packages = {}
        for name in ('numpy', 'torch', 'association'):
            path = self.root / 'association/__init__.py' if name == 'association' else prefix / name / '__init__.py'
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text('# fake package\n')
            packages[name] = types.ModuleType(name)
            packages[name].__file__ = str(path)
        packages['numpy'].__version__ = '1.26.4'
        torch = packages['torch']
        torch.__version__ = '2.5.1+cu121'
        torch.version = types.SimpleNamespace(cuda='12.1')
        torch.get_float32_matmul_precision = lambda: 'highest'
        torch.backends = types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
            cudnn=types.SimpleNamespace(allow_tf32=True))
        # Any actual GPU inspection would fail this test.
        torch.cuda = mock.Mock(side_effect=AssertionError('GPU access is forbidden'))
        torch.cuda.is_available.side_effect = AssertionError('GPU access is forbidden')
        torch.cuda.device_count.side_effect = AssertionError('GPU access is forbidden')
        return prefix, packages

    def test_environment_capture_normalizes_freeze_without_gpu_or_network(self):
        prefix, packages = self.fake_packages()
        completed = subprocess.CompletedProcess([], 0, ' torch==2.5.1+cu121\n\nnumpy==1.26.4 \n', '')
        with mock.patch.dict(sys.modules, packages), mock.patch.object(sys, 'prefix', str(prefix)), \
                mock.patch.object(sys, 'base_prefix', '/base'), \
                mock.patch.object(sys, 'executable', str(prefix / 'bin/python')), \
                mock.patch.object(contract.sysconfig, 'get_path', return_value=str(prefix)), \
                mock.patch.object(contract.subprocess, 'run', return_value=completed) as run:
            sys.modules.pop('sitecustomize', None)
            environment = contract.capture_environment(self.root)
        self.assertEqual(environment['pip_freeze'], ['numpy==1.26.4', 'torch==2.5.1+cu121'])
        self.assertEqual(environment['sitecustomize'], {'loaded': False,
                         'path': str((prefix / 'sitecustomize.py').resolve()), 'exists': False, 'sha256': None})
        self.assertEqual(run.call_args.args[0], [str(prefix / 'bin/python'), '-m', 'pip',
                                               '--disable-pip-version-check', 'freeze', '--all'])
        self.assertEqual(run.call_args.kwargs['env']['PIP_NO_INDEX'], '1')
        self.assertEqual(run.call_args.kwargs['env']['PIP_CONFIG_FILE'], os.devnull)

    def test_capture_rejects_stale_project_and_external_numerical_package_sources(self):
        prefix, packages = self.fake_packages()
        outside = Path(self.temporary.name) / 'stale.py'
        outside.write_text('# stale source\n')
        for name in ('association', 'numpy', 'torch'):
            with self.subTest(package=name):
                original = packages[name].__file__
                packages[name].__file__ = str(outside)
                with mock.patch.dict(sys.modules, packages), mock.patch.object(sys, 'prefix', str(prefix)):
                    with self.assertRaisesRegex(ValueError, 'must be imported'):
                        contract.capture_environment(self.root)
                packages[name].__file__ = original


if __name__ == '__main__':
    unittest.main()
