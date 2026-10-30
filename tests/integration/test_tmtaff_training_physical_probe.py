import unittest
import numpy as np
from tests.fixtures.native_zip import make_native_zip_fixture
from association.data.archive import MTADArchive
from association.data.native import _record
from association.data.source_events import source_events
from association.data.timed import fit_timed_normalizer
from tools.analysis.native_cv_state_control import pack_native_events
from tools.analysis import tmtaff_training_physical_probe as p

class TrainingProbeInputsTests(unittest.TestCase):
    def test_rebuilt_cached_events_match_original_packed_native_units_and_queries(self):
        f=make_native_zip_fixture(self)
        with MTADArchive(f.path) as archive:
            raw=_record(archive.load_scene('0'),'0','T_core',128)
        legacy=p.prepare_timed_scene(raw,**p.DATA)
        cache=pack_native_events(legacy,source_events(legacy))
        expected=p.prepare_timed_scene(raw,**p.DATA,speed_unit_policy=p.SPEED_UNIT_POLICY)
        actual=p.timed_scene(cache,np.asarray(raw['identities'],str),'0')
        for key in ('native_values','native_times','native_observed','grid','observed','values'):
            np.testing.assert_array_equal(actual[key],expected[key])
        norm=fit_timed_normalizer([expected])
        for a,b in zip(p.normalize_scene(actual,norm),p.normalize_scene(expected,norm)):np.testing.assert_array_equal(a,b)
