"""Small deterministic checks; these never read the project's reserved-year data."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv_pipeline as cv


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def entry(self, year, turbine="1", count=400):
        directory = self.root / "cache" / str(year) / f"Station_{turbine}"
        directory.mkdir(parents=True, exist_ok=True)
        times = pd.date_range(pd.Timestamp(year, 1, 1) + cv.LOOKBACK, periods=count, freq=cv.STEP)
        on = times[count // 2]
        target = ((times <= on) & (times > on - cv.HORIZON)).astype("int8")
        rng = np.random.default_rng(year + int(turbine))
        frame = pd.DataFrame(
            {
                "TimeStamp": times,
                "turbine": turbine,
                "target": target,
                "wtc_test__mean": (rng.normal(size=count) + target).astype("float32"),
            }
        )
        frame.to_parquet(directory / "features.parquet", index=False)
        pd.DataFrame(
            {
                "turbine": [turbine],
                "TimeOn": [on],
                "TimeOff": [on + cv.STEP],
                "codes": ["8230"],
                "records": [1],
            }
        ).to_parquet(directory / "episodes.parquet", index=False)
        return {
            "year": year,
            "turbine": turbine,
            "eligible_rows": count,
            "features": str(directory / "features.parquet"),
            "episodes": str(directory / "episodes.parquet"),
            "signature": {"year": year},
        }

    def test_real_preparation_history_labels_deduplication_and_active_episodes(self):
        grid = pd.date_range("2016-01-01", "2017-01-01", freq=cv.STEP, inclusive="left")
        frame = pd.DataFrame(
            {
                "wtc_test": np.arange(len(grid), dtype=float),
                "sensorObserved": True,
                "alarmOccurred": False,
                "alarmEvents": "[]",
            },
            index=grid,
        )
        frame.iloc[12, frame.columns.get_loc("wtc_test")] = 999
        events = [
            {
                "Alarmcode": "8210",
                "TimeOn": "2016-01-01 12:00:00",
                "TimeOff": "2016-01-01 12:20:00",
            },
            {
                "Alarmcode": "8210",
                "TimeOn": "2016-01-01 12:00:00",
                "TimeOff": "2016-01-01 12:20:00",
            },
            {
                "Alarmcode": "8230",
                "TimeOn": "2016-01-01 12:05:00",
                "TimeOff": "2016-01-01 12:30:00",
            },
            {"Alarmcode": "3130", "TimeOn": "2016-01-01 13:00:00", "TimeOff": None},
        ]
        frame.loc[pd.Timestamp("2016-01-01 12:00"), ["alarmOccurred", "alarmEvents"]] = [
            True,
            json.dumps(events),
        ]
        source = self.root / "source" / "2016" / "Station_1" / "data.parquet"
        source.parent.mkdir(parents=True)
        frame.to_parquet(source)
        entry = cv.prepare_one(source, self.root / "prepared", ["8210", "8230"], ["wtc_test"])
        features = pd.read_parquet(entry["features"]).set_index("TimeStamp")
        episodes = pd.read_parquet(entry["episodes"])
        self.assertEqual(features.loc["2016-01-01 02:00", "wtc_test__mean"], 5.5)
        self.assertEqual(features.loc["2016-01-01 06:00", "target"], 0)  # End is exclusive.
        self.assertEqual(features.loc["2016-01-01 06:10", "target"], 1)
        self.assertNotIn(pd.Timestamp("2016-01-01 12:10"), features.index)
        self.assertEqual(features.loc["2016-01-01 12:30", "target"], 0)
        self.assertEqual(entry["exact_duplicates_removed"], 1)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes.iloc[0].codes, "8210; 8230")
        self.assertTrue((features.index + cv.HORIZON <= pd.Timestamp("2017-01-01")).all())
        self.assertNotIn("alarmOccurred", features.columns)
        self.assertNotIn(pd.Timestamp("2016-01-01 01:50"), features.index)
        # A source gap invalidates its following two input hours without filling the gap.
        frame.loc["2016-01-01 00:20", "sensorObserved"] = False
        frame.to_parquet(source)
        revised = cv.prepare_one(source, self.root / "prepared", ["8210", "8230"], ["wtc_test"])
        self.assertNotEqual(entry["features"], revised["features"])
        revised_rows = pd.read_parquet(revised["features"])
        self.assertNotIn(pd.Timestamp("2016-01-01 02:20"), revised_rows.TimeStamp.tolist())
        self.assertIn(pd.Timestamp("2016-01-01 02:30"), revised_rows.TimeStamp.tolist())

    def test_shared_timestamp_purging(self):
        entry = self.entry(2023)
        frame = pd.read_parquet(entry["features"]).iloc[:4].copy()
        frame["TimeStamp"] = pd.to_datetime(
            ["2023-01-01 01:50", "2023-01-01 02:00", "2023-12-31 18:00", "2023-12-31 18:10"]
        )
        frame.to_parquet(entry["features"], index=False)
        selected = list(cv.iter_frames([entry], "2023-01-01", "2024-01-01"))[0][1]
        self.assertEqual(
            selected.TimeStamp.tolist(),
            [pd.Timestamp("2023-01-01 02:00"), pd.Timestamp("2023-12-31 18:00")],
        )

    def test_sampling_preserves_all_positives_and_turbine_strata(self):
        entries = [self.entry(year, turbine) for year in [2016, 2020] for turbine in ["1", "2"]]
        x, y, weights, audit = cv.training_sample(
            entries, "2016-01-01", "2021-01-01", ["wtc_test__mean"], max_negatives=50
        )
        self.assertEqual(int(y.sum()), 4 * 36)
        self.assertEqual(int((y == 0).sum()), 50)
        self.assertEqual(audit["turbines"], ["1", "2"])
        self.assertTrue(np.isfinite(weights).all())
        model = cv.fit_model({"name": "constant", "kind": "constant"}, x, y, weights, audit)
        self.assertAlmostEqual(model["prevalence"], 36 / 400)
        self.assertNotAlmostEqual(model["prevalence"], y.mean())

    def test_event_eligibility_lead_time_and_false_warning_episodes(self):
        times = pd.to_datetime(
            [
                "2021-01-01 06:00",
                "2021-01-01 07:00",
                "2021-01-01 08:00",
                "2021-01-01 08:10",
                "2021-01-02 15:00",
            ]
        )
        predictions = pd.DataFrame(
            {
                "TimeStamp": times,
                "turbine": "1",
                "target": [0, 1, 1, 1, 0],
                "score": [0, 0.9, 0.9, 0.9, 0.9],
            }
        )
        episodes = pd.DataFrame({"turbine": ["1"], "TimeOn": [pd.Timestamp("2021-01-01 12:00")]})
        result = cv.event_metrics(predictions, episodes, 0.5)
        self.assertEqual(result["eligible_alarm_episodes"], 1)
        self.assertEqual(result["detected_alarm_episodes"], 1)
        self.assertEqual(result["warning_episodes"], 3)
        self.assertEqual(result["false_warning_episodes"], 1)
        self.assertEqual(result["median_detected_lead_hours"], 5)

    def test_reserved_data_gate_and_development_year_guards(self):
        with (
            patch.object(
                cv, "discover", side_effect=AssertionError("Reserved discovery forbidden")
            ),
            patch.object(
                pd, "read_parquet", side_effect=AssertionError("Reserved data read forbidden")
            ),
        ):
            self.assertIsNone(cv.reserved_test({}, self.root, self.root, self.root, enabled=False))
        with self.assertRaises(ValueError):
            cv.run_cv([{"year": 2024}], [], [], [], self.root)
        with self.assertRaises(ValueError):
            cv.final_refit([{"year": 2025}], [], [], [], self.root, self.root, {})
        for year in [2016, 2024, 2025]:
            p = self.root / "source" / str(year) / "Station_1" / "data.parquet"
            p.parent.mkdir(parents=True)
            p.touch()
        paths, missing = cv.discover(self.root / "source", [2016, 2017])
        self.assertEqual([p.parent.parent.name for p in paths], ["2016"])
        self.assertEqual(missing, [2017])

    def test_budget_threshold_does_not_reward_permanent_warnings(self):
        times = pd.date_range("2021-01-01", periods=100, freq=cv.STEP)
        p = pd.DataFrame(
            {"TimeStamp": times, "turbine": "1", "target": [1] + [0] * 99, "score": [0.9] * 100}
        )
        e = pd.DataFrame({"turbine": ["1"], "TimeOn": [times[0]]})
        threshold = cv.choose_threshold(p, e, "false_warning_budget", 1, 0.05)
        self.assertGreater(threshold, 0.9)

    def test_three_fold_all_family_smoke_and_final_freeze(self):
        entries = [
            self.entry(year, turbine)
            for year in [2016, 2020, 2021, 2022, 2023]
            for turbine in ["1", "2"]
        ]
        candidates = [
            {"name": "constant", "kind": "constant"},
            {"name": "random", "kind": "random"},
            {"name": "logistic", "kind": "logistic"},
            {
                "name": "lightgbm",
                "kind": "lightgbm",
                "parameters": {"n_estimators": 5, "num_leaves": 3},
            },
            {"name": "forest", "kind": "forest", "parameters": {"n_estimators": 5, "max_depth": 3}},
        ]
        folds = [
            {
                "name": f"evaluate_{year}",
                "train_begin": "2016-01-01",
                "calibration_begin": f"{year - 1}-01-01",
                "evaluation_begin": f"{year}-01-01",
                "end": f"{year + 1}-01-01",
            }
            for year in [2021, 2022, 2023]
        ]
        calibration_years = []
        original = cv.choose_threshold

        def checked(predictions, *args):
            calibration_years.append(set(predictions.TimeStamp.dt.year.unique()))
            return original(predictions, *args)

        with patch.object(cv, "choose_threshold", side_effect=checked):
            table, summary = cv.run_cv(
                entries, ["wtc_test__mean"], candidates, folds, self.root / "cv", max_negatives=100
            )
        self.assertEqual(len(table), 15)
        self.assertEqual(len(summary), 5)
        self.assertTrue(all(years in [{2020}, {2021}, {2022}] for years in calibration_years))
        self.assertTrue((summary.folds == 3).all())
        self.assertTrue((table.eligible_alarm_episodes == 2).all())
        config = {
            "max_train_negatives": 100,
            "seed": 42,
            "threshold_policy": "max_f1",
            "false_warning_budget": 1,
            "alarm_codes": ["8230"],
            "sensors": ["wtc_test"],
        }
        bundle = cv.final_refit(
            entries,
            ["wtc_test__mean"],
            candidates,
            folds,
            self.root / "cv",
            self.root / "final",
            config,
        )
        self.assertTrue(bundle["frozen"])
        self.assertEqual(len(bundle["models"]), 5)
        self.assertIn(bundle["champion"], ["logistic", "lightgbm", "forest"])
        self.assertTrue(all(e["year"] <= 2023 for e in bundle["development_sources"]))
        self.assertTrue((self.root / "final" / "frozen_models.joblib").is_file())
        with patch.object(
            pd, "read_parquet", side_effect=AssertionError("Reserved data read forbidden")
        ):
            self.assertIsNone(
                cv.reserved_test(bundle, self.root, self.root, self.root / "test", enabled=False)
            )
        # Explicit enabled evaluation is checked on synthetic files only, never project holdouts.
        synthetic_test = [
            self.entry(year, turbine) for year in [2024, 2025] for turbine in ["1", "2"]
        ]
        for year in [2024, 2025]:
            for turbine in ["1", "2"]:
                path = (
                    self.root
                    / "synthetic_source"
                    / str(year)
                    / f"Station_{turbine}"
                    / "data.parquet"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
        before = bundle["thresholds"].copy()
        with (
            patch.object(cv, "prepare", return_value=synthetic_test),
            patch.object(
                cv,
                "choose_threshold",
                side_effect=AssertionError("Test must not retune thresholds"),
            ),
            patch.object(cv, "fit_model", side_effect=AssertionError("Test must not refit models")),
        ):
            results = cv.reserved_test(
                bundle,
                self.root / "synthetic_source",
                self.root / "unused",
                self.root / "test",
                enabled=True,
            )
        self.assertEqual(len(results), 5)
        self.assertEqual(before, bundle["thresholds"])
        self.assertTrue((self.root / "test" / "per_turbine_metrics.csv").is_file())
        with self.assertRaises(FileExistsError):
            cv.reserved_test(bundle, self.root, self.root, self.root / "test", enabled=True)


if __name__ == "__main__":
    unittest.main()
