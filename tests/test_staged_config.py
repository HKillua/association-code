"""Versioned configuration and CLI boundaries for staged training."""
import copy
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from association import cli
from association.config import load_config, validate_config, validate_manifest_config
from association.dataset import make_synthetic_archive, prepare
from association.profile import profile_reference
from association.reference import _check_configuration, load_reference, train_reference
from association.runtime import read_json


ROOT = Path(__file__).resolve().parents[1]


def staged_config(purpose='probe'):
    config = load_config(ROOT / 'configs' / 'smoke.json')
    ref = config['reference']
    for name in ('tcm_epochs', 'tam_epochs', 'warmup_epochs', 'joint_completion'):
        del ref[name]
    ref.update(schedule='staged_updates_v2', purpose=purpose, tcm_updates=8,
               tam_updates=4, joint_updates=2, validate_every_updates=2,
               completion_cache_mib=0, pair_batch_size=4, max_hours=.5)
    return config


class StagedConfigTests(unittest.TestCase):
    def test_legacy_defaults_preserve_config_and_v2_needs_no_epoch_fields(self):
        original = read_json(ROOT / 'configs' / 'smoke.json')
        self.assertEqual(validate_config(original), original)
        self.assertNotIn('schedule', original['reference'])
        for purpose in ('probe', 'fit'):
            config = staged_config(purpose)
            before = copy.deepcopy(config)
            self.assertEqual(validate_config(config), before)
            model, ref = _check_configuration(config, 'A0', 'ce')
            self.assertEqual(model.length, config['model']['length'])
            self.assertEqual(ref['purpose'], purpose)
            self.assertEqual(config, before)

    def test_unknown_schedule_and_invalid_staged_fields_are_rejected(self):
        invalid = [('schedule', 'staged_update_v2'), ('purpose', 'train'),
                   ('pair_batch_size', 1), ('pair_batch_size', 3), ('pair_batch_size', True),
                   ('tcm_updates', 0), ('tam_updates', 1.5), ('joint_updates', -1),
                   ('joint_updates', False), ('validate_every_updates', 0),
                   ('completion_cache_mib', -.1), ('completion_cache_mib', 1.5),
                   ('gradient_clip', 0), ('gradient_clip', float('nan')),
                   ('learning_rate', 0), ('max_hours', 0), ('weight_decay', -1),
                   ('completion_weight', float('inf'))]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                config = staged_config()
                config['reference'][key] = value
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_probe_caps_and_fit_bounds_are_distinct(self):
        for name, excessive in (('max_hours', .50001), ('tcm_updates', 9),
                                ('tam_updates', 5), ('joint_updates', 3)):
            with self.subTest(name=name):
                config = staged_config()
                config['reference'][name] = excessive
                with self.assertRaisesRegex(ValueError, 'probe'):
                    validate_config(config)
                config['reference']['purpose'] = 'fit'
                validate_config(config)
        config = staged_config()
        config['reference']['joint_updates'] = 0
        validate_config(config)

    def test_prepared_data_can_be_reused_without_rewriting_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = load_config(ROOT / 'configs' / 'smoke.json')
            archive = make_synthetic_archive(root / 'data.zip')
            prepare(archive, legacy, root / 'prepared', synthetic=True)
            audit = (root / 'prepared' / 'audit.json').read_bytes()
            args = SimpleNamespace(prepared=root / 'prepared', manifest=None)
            config = staged_config()
            manifest = cli._manifest(args, config)
            validate_manifest_config(manifest, config)
            self.assertEqual((root / 'prepared' / 'audit.json').read_bytes(), audit)
            config['caps']['T_core'] += 1
            with self.assertRaisesRegex(ValueError, 'seed/caps'):
                validate_manifest_config(manifest, config)
            config = staged_config()
            config['model']['length'] += 1
            with self.assertRaisesRegex(ValueError, 'length'):
                cli._manifest(args, config)

    def test_probe_rejected_by_reference_train_load_and_legacy_profile_before_data(self):
        config = staged_config()
        with patch('association.reference.validate_manifest_config', side_effect=AssertionError('data reached')):
            with self.assertRaisesRegex(ValueError, 'probe'):
                train_reference('missing.zip', {}, config, 'A0', 'ce', 20, 'cpu', 'missing-output')
            with self.assertRaisesRegex(ValueError, 'probe'):
                load_reference('missing-output', {}, config, 'A0', 'ce', 20, 'cpu')
        for purpose in ('probe', 'fit'):
            with self.subTest(purpose=purpose), patch('association.profile.validate_manifest_config',
                                                    side_effect=AssertionError('data reached')):
                with self.assertRaisesRegex(ValueError, 'legacy'):
                    profile_reference('missing.zip', {}, staged_config(purpose), 'A0', 'ce', 20,
                                      'cpu', 'missing-output')

    def test_fit_routes_to_staged_trainer_with_unchanged_arguments(self):
        module = ModuleType('association.staged')
        module.train_staged_reference = Mock(return_value={'marker': 'staged'})
        args = ('archive', {'manifest': 'marker'}, staged_config('fit'), 'A0', 'ce', 20, 'cpu', 'out')
        with patch.dict('sys.modules', {'association.staged': module}):
            self.assertEqual(train_reference(*args), {'marker': 'staged'})
        module.train_staged_reference.assert_called_once_with(*args)

    def _cli_args(self, command, config_path, output):
        args = [command, '--config', str(config_path), '--archive', 'missing.zip',
                '--prepared', 'missing-prepared', '--output', str(output), '--device', 'cpu']
        if command != 'run-pilot':
            args += ['--role', 'A0', '--branch', 'ce', '--seed', '20']
        return args

    def test_cli_rejects_mismatched_purposes_before_device_manifest_or_output(self):
        cases = [(command, staged_config()) for command in ('train-reference', 'run-pilot', 'profile')]
        cases += [('profile', staged_config('fit')), ('probe-training', staged_config('fit')),
                  ('probe-training', load_config(ROOT / 'configs' / 'smoke.json'))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (command, config) in enumerate(cases):
                with self.subTest(command=command, purpose=config['reference'].get('purpose')):
                    path = root / f'config-{index}.json'
                    path.write_text(json.dumps(config))
                    output = root / f'output-{index}'
                    with patch.object(cli, 'device_from_string', side_effect=AssertionError('device reached')), \
                         patch.object(cli, '_manifest', side_effect=AssertionError('data reached')), \
                         redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                        self.assertEqual(cli.main(self._cli_args(command, path, output)), 1)
                    self.assertFalse(output.exists())

    def test_cli_probe_routes_to_staged_trainer(self):
        module = ModuleType('association.staged')
        module.train_staged_reference = Mock()
        config = staged_config()
        manifest = {'manifest': 'marker'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            output = root / 'output'
            with patch.dict('sys.modules', {'association.staged': module}), \
                 patch.object(cli, '_manifest', return_value=manifest), \
                 patch.object(cli, 'device_from_string', return_value='cpu'), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(self._cli_args('probe-training', path, output)), 0)
            module.train_staged_reference.assert_called_once_with(
                'missing.zip', manifest, config, 'A0', 'ce', 20, 'cpu', str(output))


if __name__ == '__main__':
    unittest.main()
