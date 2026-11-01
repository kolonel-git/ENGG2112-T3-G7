"""Timing, censoring and sampling tests for the independent binary experiment."""

import unittest

import numpy as np
import pandas as pd

from wt_predmaint.short_horizon import (
    UniformSample,
    binary_targets,
    feature_windows,
    predict_history,
    split_mask,
    validate_grid,
)


def example(rows=40):
    index = pd.date_range("2022-01-01", periods=rows, freq="10min")
    return pd.DataFrame(
        {
            "wtc_sensor": np.arange(rows, dtype=float),
            "sensorObserved": True,
            "alarmOccurred": False,
        },
        index=index,
    )


class TimingTests(unittest.TestCase):
    def test_features_use_exactly_previous_twelve_rows(self):
        frame = example()
        features, complete = feature_windows(frame, ["wtc_sensor"])
        self.assertFalse(complete.iloc[11])
        self.assertTrue(complete.iloc[12])
        self.assertAlmostEqual(features.iloc[12]["wtc_sensor__mean"], 5.5)
        self.assertEqual(features.iloc[12]["wtc_sensor__last"], 11)
        self.assertEqual(features.iloc[12]["wtc_sensor__change"], 11)
        changed = frame.copy()
        changed.loc[changed.index[12] :, "wtc_sensor"] = 1e9
        new, _ = feature_windows(changed, ["wtc_sensor"])
        pd.testing.assert_series_equal(features.iloc[12], new.iloc[12])

    def test_alarm_targets_boundaries_and_nesting(self):
        frame = example()
        frame.loc[frame.index[14], "alarmOccurred"] = True
        targets = binary_targets(frame, [10, 20, 30])
        self.assertEqual(targets.iloc[12].tolist(), [0, 0, 1])
        self.assertEqual(targets.iloc[13].tolist(), [0, 1, 1])
        self.assertEqual(targets.iloc[14].tolist(), [1, 1, 1])
        self.assertEqual(targets.iloc[15].tolist(), [0, 0, 0])
        self.assertTrue(targets.iloc[-1, 1:].isna().all())
        # A future change affects labels, never the feature history.
        features, _ = feature_windows(frame, ["wtc_sensor"])
        altered = frame.copy()
        altered["alarmOccurred"] = ~altered.alarmOccurred
        new, _ = feature_windows(altered, ["wtc_sensor"])
        pd.testing.assert_frame_equal(features, new)

    def test_missing_observation_invalidates_history(self):
        frame = example()
        frame.loc[frame.index[5], "sensorObserved"] = False
        _, complete = feature_windows(frame, ["wtc_sensor"])
        self.assertFalse(complete.iloc[12])
        self.assertTrue(complete.iloc[18])

    def test_purge_excludes_overlapping_history_and_future(self):
        frame = example()
        _, complete = feature_windows(frame, ["wtc_sensor"])
        begin, end = frame.index[12], frame.index[36]
        keep = split_mask(frame.index, complete, str(begin), str(end))
        self.assertFalse(keep[23])
        self.assertTrue(keep[24])
        self.assertTrue(keep[33])
        self.assertFalse(keep[34])

    def test_broken_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Gap"):
            validate_grid(example().drop(example().index[5]))

    def test_uniform_sample_is_reproducible_and_bounded(self):
        frame = example(100).reset_index(names="TimeStamp")
        frame["turbine"] = "1"
        results = []
        for _ in range(2):
            sampler = UniformSample(15, 42)
            for offset in range(0, 100, 20):
                sampler.add(frame.iloc[offset : offset + 20])
            results.append(sampler.finish())
            self.assertEqual(sampler.seen, 100)
        self.assertEqual(len(results[0]), 15)
        pd.testing.assert_frame_equal(*results)

    def test_inference_needs_no_alarm_labels_or_future_rows(self):
        class CaptureModel:
            def predict_proba(self, x):
                self.features = x.copy()
                return np.array([[0.3, 0.7]])

        frame = example()
        features, _ = feature_windows(frame, ["wtc_sensor"])
        model = CaptureModel()
        bundle = {
            "model": model,
            "sensors": ["wtc_sensor"],
            "features": list(features),
            "threshold": 0.6,
            "horizon_minutes": 10,
        }
        history = frame.drop(columns="alarmOccurred").iloc[:12]
        result = predict_history(bundle, history, frame.index[12])
        self.assertEqual(result["alarm_prediction"], "yes")
        np.testing.assert_allclose(model.features, features.iloc[[12]].to_numpy())
        with self.assertRaisesRegex(ValueError, "12 observed"):
            predict_history(bundle, history.iloc[1:], frame.index[12])


if __name__ == "__main__":
    unittest.main()
