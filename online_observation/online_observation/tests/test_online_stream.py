import unittest

from online_observation.stream import StreamState


def row(time, *, source="9001", batch="0", lat="0", lon="0", vel="10", cou="90"):
    return dict(time=str(time), source=source, batch=batch, lat=lat, lon=lon, vel=vel, cou=cou)


class StreamTests(unittest.TestCase):
    def test_empty_cold_start_and_no_arrival_prediction(self):
        state = StreamState(stale_after_seconds=2)
        self.assertEqual(state.predict(0)["tracks"], [])
        state.ingest([row(1)], arrival_time=1)
        track = state.predict(1)["tracks"][0]
        self.assertTrue(track["appeared"])
        self.assertTrue(track["latest_observation_updated"])
        self.assertFalse(track["predicted"])
        self.assertEqual(track["longitude"], 0)
        self.assertAlmostEqual(track["speed_mps"], 5.144444444444444)
        current = state.predict(2)
        track = current["tracks"][0]
        self.assertEqual(current["query_time"], 2)
        self.assertFalse(track["appeared"])
        self.assertFalse(track["latest_observation_updated"])
        self.assertTrue(track["predicted"])
        self.assertAlmostEqual(track["longitude"], 0.0000462651)
        self.assertEqual(len(state.predict(3)["tracks"]), 1)
        self.assertEqual(state.predict(4)["disappeared"],
                         [dict(source="9001", batch="0", reason="stale")])
        state.ingest([row(5)], arrival_time=5)
        self.assertTrue(state.predict(5)["tracks"][0]["appeared"])

    def test_late_arrival_keeps_latest_event_and_does_not_reactivate_stale_track(self):
        state = StreamState(stale_after_seconds=2)
        state.ingest([row(2)], arrival_time=2)
        state.predict(2)
        state.ingest([row(1, lon="20")], arrival_time=3)
        track = state.predict(3)["tracks"][0]
        self.assertEqual(track["observation_time"], 2)
        self.assertEqual(track["arrival_time"], 2)
        self.assertFalse(track["latest_observation_updated"])
        state.predict(5)
        state.ingest([row(4)], arrival_time=8)
        self.assertEqual(state.predict(8)["tracks"], [])

    def test_duplicate_and_late_conflict_apply_from_arrival(self):
        state = StreamState(stale_after_seconds=5)
        first = row(0, cou="0")
        state.ingest([first, first, row(0, source="9002")], arrival_time=0)
        published = state.predict(0)
        self.assertEqual(published["audit"]["exact_duplicate_rows_removed"], 1)
        self.assertEqual(len(published["tracks"]), 2)
        state.ingest([row(0, cou="360")], arrival_time=1)
        following = state.predict(1)
        self.assertEqual(following["audit"]["quarantine_reasons"], {"conflicting_same_time": 1})
        self.assertEqual(following["disappeared"],
                         [dict(source="9001", batch="0", reason="quarantined")])
        state.ingest([row(2)], arrival_time=2)
        self.assertEqual([track["source"] for track in state.predict(2)["tracks"]], ["9002"])
        self.assertEqual(len(published["tracks"]), 2)

    def test_invalid_state_is_quarantined_when_observed(self):
        state = StreamState(stale_after_seconds=5)
        state.ingest([row(0, cou="360"), row(0, batch="1", cou="-1")], arrival_time=0)
        current = state.predict(0)
        self.assertEqual(current["tracks"][0]["course_degrees"], 0)
        self.assertEqual(current["audit"]["quarantine_reasons"], {"missing_course": 1})

    def test_forecast_from_pole_and_across_antimeridian(self):
        state = StreamState(stale_after_seconds=5)
        state.ingest([
            row(0, lat="90", lon="30", cou="0"),
            row(0, batch="1", lat="-90", lon="30", cou="0"),
            row(0, batch="2", lon="179.99999"),
        ], arrival_time=0)
        north, south, crossing = state.predict(1)["tracks"]
        self.assertAlmostEqual(north["latitude"], 89.9999537349)
        self.assertAlmostEqual(north["longitude"], -150)
        self.assertAlmostEqual(south["latitude"], -89.9999537349)
        self.assertAlmostEqual(south["longitude"], 30)
        self.assertAlmostEqual(crossing["longitude"], -179.9999637349)

    def test_clock_errors_reject_whole_batch(self):
        state = StreamState(stale_after_seconds=5)
        with self.assertRaises(ValueError):
            state.ingest([row(0), row(2)], arrival_time=1)
        self.assertEqual(state.predict(0)["audit"]["received_rows"], 0)
        with self.assertRaises(ValueError):
            state.ingest([row(0)], arrival_time=0)
        state.ingest([row(1)], arrival_time=2)
        with self.assertRaises(ValueError):
            state.predict(1)
        self.assertEqual(state.predict(2)["tracks"][0]["arrival_time"], 2)
        with self.assertRaises(ValueError):
            state.predict(2)


if __name__ == "__main__":
    unittest.main()
