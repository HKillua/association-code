import unittest

from online_observation.neighbor_witness_experiment import _parallel_motion


class NeighborWitnessExperimentTests(unittest.TestCase):
    def test_peer_parallelism_uses_arrived_positions_and_accepts_straight_motion(self):
        stamps = range(0, 141, 20)
        left = [(stamp, (4 * stamp, 0)) for stamp in stamps]
        same_course = [(stamp, (3 * stamp, 500)) for stamp in stamps]
        opposing = [(stamp, (-4 * stamp, 500)) for stamp in stamps]
        self.assertIsNone(_parallel_motion(left, same_course, 119))
        self.assertIs(_parallel_motion(left, same_course, 120), True)
        self.assertIs(_parallel_motion(left, opposing, 120), False)
