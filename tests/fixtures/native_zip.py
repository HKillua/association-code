"""Native ZIP fixture with deliberately unreadable sealed-test members."""
import csv
import io
import tempfile
import zipfile
from pathlib import Path
from dataclasses import asdict
from association.utils import runtime
from association.models.tmtaff import ModelConfig
from association.models.fusion import PairFusionConfig
from association.data.dataset import load_timed_dataset

def csv_bytes(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')

def frozen_manifest():
    splits = {'T_core': ['0'], 'T_head': ['1'], 'V_select': ['2', '4'], 'T_test': ['3']}
    ids = ['0', '1', '2', '3', '4']
    return dict(splits=splits, all_scenes=ids, components={sid: sid for sid in ids},
        component_members={sid: [sid] for sid in ids},
        component_splits={sid: split for split, group in splits.items() for sid in group},
        label_members={sid: {} for sid in ids}, caps={split: len(group) for split, group in splits.items()},
        development_scenes=['0'])


class NativeZipFixture:
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / 'native.zip'
        self.manifest = frozen_manifest()
        self.data = dict(length=4, max_observations=4, time_unit_seconds=10.)
        self.alignment = {sid: dict(time_offset_seconds=1000., max_gap_seconds=10.)
                          for sid in ('0', '1', '2', '4')}
        self.config = dict(scope='TMTAFF-time-adapted-v1',
            model=asdict(ModelConfig(length=4, d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0)),
            fusion=asdict(PairFusionConfig(fc_hidden=8, length=4, d_model=8, nhead=2,
                num_layers=1, dim_feedforward=16, dropout=0)),
            data={key: self.data[key] for key in ('max_observations', 'time_unit_seconds')},
            train_splits=['T_core'], training=dict(
                updates=dict.fromkeys(('TCM', 'TAM_FROZEN', 'TFM_FROZEN', 'JOINT'), 1),
                learning_rates=dict.fromkeys(('TCM', 'TAM_FROZEN', 'TFM_FROZEN', 'JOINT'), .001),
                seed=9, branch='ce', batch_tracks=4, pair_batch_size=4, pair_microbatch=2,
                track_microbatch=2, completion_weight=.1, gradient_clip=1., max_hours=.1,
                validate_every_updates=1, completion_cache_mib=1, position_loss_mode='standard_smooth_l1',
                lr_schedule='constant', singleton_policy='observed', validation_batches=dict(tracks=4, pairs=4)))
        self.write_zip()

    def write_zip(self, *, empty_labels=False, zero_label_duration=False, missing_reference=None):
        with zipfile.ZipFile(self.path, 'w') as archive:
            for sid in self.manifest['all_scenes']:
                labels, values, refs = [], [], []
                if sid == '4':
                    if not empty_labels:
                        labels = [[5, 5 if zero_label_duration else 25, 'target4a-0-0', '9001', '0']]
                    refs = [['target4a-0', t+1000, 20., 120., 5., 45.] for t in (5, 15, 25)]
                else:
                    for name, source, batch, times in (
                            ('a', '9001', '0', (0, 10, 20, 30)), ('a', '9002', '1', (12, 18)),
                            ('b', '9001', '2', (0, 10, 20, 30)), ('b', '9002', '3', (0, 10))):
                        labels.append([times[0], times[-1], f'target{sid}{name}-0-{batch}', source, batch])
                        values.extend([[batch, source, t, 20+int(sid)+(.1 if name == 'b' else 0)+t*.0001,
                                        120+t*.0002, 5, 45] for t in times])
                    labels.append([0, 30, f'target{sid}c-0-4', '9003', '4'])  # quarantined: no measurement
                    for name in ('a', 'b', 'c'):
                        if (sid, name) == missing_reference:
                            continue
                        refs.extend([[f'target{sid}{name}-0', t+1000, 20+int(sid)+t*.0001,
                                      120+t*.0002, 5, 45] for t in (0, 10, 20, 30)])
                for directory, stem, header, rows in (
                        ('关联表', '关联结果', ['t_s', 't_e', 'mmsi', 'source', 'batch'], labels),
                        ('量测场景', '场景', ['batch', 'source', 'time', 'lat', 'lon', 'vel', 'cou'], values),
                        ('真实场景', '场景', ['MMSI', 'time', 'lat', 'lon', 'vel', 'cou'], refs)):
                    content = b'SEALED TEST MUST NOT BE READ' if sid == '3' else csv_bytes(header, rows)
                    archive.writestr(f'数据集level2/{directory}/{stem}-{sid}.csv', content)

    def load(self, **kwargs):
        arguments = dict(train_splits=['T_core'], data_config=self.data, alignment_by_scene=self.alignment)
        arguments.update(kwargs)
        return load_timed_dataset(self.path, self.manifest, **arguments)

    def cli_args(self, output):
        for filename, value in [('manifest.json', self.manifest), ('alignment.json', self.alignment), ('config.json', self.config)]:
            runtime.write_json(self.root / filename, value)
        return ['--archive', str(self.path), '--manifest', str(self.root/'manifest.json'),
                '--alignment', str(self.root/'alignment.json'), '--config', str(self.root/'config.json'),
                '--output', str(self.root/output), '--cpu-threads', '1']


def make_native_zip_fixture(test_case):
    fixture = NativeZipFixture()
    fixture.addCleanup = test_case.addCleanup
    fixture.setUp()
    return fixture
