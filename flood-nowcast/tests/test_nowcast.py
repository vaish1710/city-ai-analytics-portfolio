import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import mock_noaa  # noqa: E402
import noaa  # noqa: E402
from features import make_features  # noqa: E402
from live import latest_forecast  # noqa: E402
from model import HORIZONS, best_margin, chrono_split, event_scores, fit, forecast_series  # noqa: E402
from sim import make_series  # noqa: E402


class Shared(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = make_series(days=1095)
        cls.df = cls.full.drop(columns=["_true_surge"])
        cls.thr = float(cls.df["observed"].iloc[: int(len(cls.df) * 0.6)].quantile(0.99))
        cls.bundle, cls.res = fit(cls.df, cls.thr)
        cls.bundle["meta"] = {"source": "synthetic", "station": "TEST", "datum": "MLLW", "met": True}


class FeatureAndSplitTests(Shared):
    def test_features_never_use_the_future(self):
        t0 = self.df.index[5000]
        for h in HORIZONS:
            X1, _ = make_features(self.df, h)
            bad = self.df.copy()
            fut = bad.index > t0
            for c in ("observed", "wind_speed", "pressure"):
                bad.loc[fut, c] = 999.0           # corrupt every future observation
            X2, _ = make_features(bad, h)
            a, b = X1.loc[t0], X2.loc[t0]
            np.testing.assert_allclose(a.values, b.values, err_msg=f"features at t0 changed when the future changed (h={h})")

    def test_leak_test_is_sensitive_to_future_tide_predictions(self):
        # Predictions for future hours ARE allowed (published in advance). Changing them must change pred_fut,
        # which proves the test above would notice a real leak.
        t0 = self.df.index[5000]
        X1, _ = make_features(self.df, 3)
        bad = self.df.copy(); bad.loc[bad.index > t0, "predicted"] += 1.0
        X2, _ = make_features(bad, 3)
        self.assertNotEqual(X1.loc[t0, "pred_fut"], X2.loc[t0, "pred_fut"])

    def test_chronological_split_has_gap_and_no_overlap(self):
        tr, va, te = chrono_split(1000)
        self.assertLess(tr.stop, va.start); self.assertLess(va.stop, te.start)
        self.assertGreaterEqual(va.start - tr.stop, max(HORIZONS))
        self.assertGreaterEqual(te.start - va.stop, max(HORIZONS))


class MetricTests(unittest.TestCase):
    def test_event_scores_by_hand(self):
        obs = np.array([4.0, 4.0, 1.0, 1.0, 4.0]); score = np.array([4.2, 3.0, 3.9, 0.5, 2.0])
        r = event_scores(obs, score, 3.5, 0.0)       # events T,T,F,F,T ; alerts T,F,T,F,F
        self.assertEqual((r["hits"], r["misses"], r["false_alarms"]), (1, 2, 1))
        self.assertAlmostEqual(r["POD"], 1 / 3); self.assertAlmostEqual(r["FAR"], 1 / 2); self.assertAlmostEqual(r["CSI"], 1 / 4)
        r = event_scores(obs, score, 3.5, 0.5)       # alerts T,T,T,F,F
        self.assertEqual((r["hits"], r["misses"], r["false_alarms"]), (2, 1, 1))
        self.assertAlmostEqual(r["CSI"], 2 / 4)
        self.assertEqual(best_margin(obs, score, 3.5), 1.5)  # margin 1.5 also catches the last event: CSI 3/4

    def test_nans_are_ignored_not_counted(self):
        r = event_scores(np.array([4.0, np.nan]), np.array([4.0, 4.0]), 3.5, 0.0)
        self.assertEqual(r["hits"], 1); self.assertEqual(r["false_alarms"], 0)


class ModelTests(Shared):
    def test_model_beats_persistence_and_tide_only_on_synthetic(self):
        # Pipeline regression check on synthetic data (not evidence about real flooding).
        for h in HORIZONS:
            r = self.res[self.res.horizon_h == h].set_index("method")["rmse_ft"]
            self.assertLess(r["model"], r["persistence"], f"h={h}")
            self.assertLess(r["persistence"], r["tide_only"], f"h={h}")

    def test_margins_were_tuned_on_validation_not_test(self):
        # Re-deriving margins from the validation block must reproduce the stored margins.
        tr, va, te = chrono_split(len(self.df))
        h = 3
        X, _ = make_features(self.df, h)
        fut = self.df["observed"].shift(-h).values
        s = X["pred_fut"].values + X["resid_lag0"].values
        self.assertEqual(best_margin(fut[va], s[va], self.thr), self.bundle["margins"][(h, "persistence")])


class NoaaClientTests(Shared):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.srv, cls.url, cls.state = mock_noaa.start(cls.df)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        self.state["requests"].clear(); self.state.pop("force_error", None); self.state.pop("missing", None)

    def test_long_ranges_are_chunked_under_the_limit(self):
        b, e = self.df.index[0], self.df.index[0] + pd.Timedelta(days=100)
        out = noaa.fetch("TEST", "water_level", b, e, base_url=self.url)
        spans = [(pd.Timestamp(r["end_date"]) - pd.Timestamp(r["begin_date"])).days for r in self.state["requests"]]
        self.assertGreaterEqual(len(spans), 4)
        self.assertLessEqual(max(spans), 28)
        self.assertTrue(out.index.is_unique and out.index.is_monotonic_increasing)
        self.assertEqual(len(out), len(self.df["observed"].loc[b:e]))   # no row lost or duplicated at chunk edges

    def test_request_parameters(self):
        noaa.fetch("TEST", "water_level", self.df.index[0], self.df.index[0] + pd.Timedelta(days=2), base_url=self.url)
        noaa.fetch("TEST", "wind", self.df.index[0], self.df.index[0] + pd.Timedelta(days=2), base_url=self.url)
        wl, wind = self.state["requests"]
        self.assertEqual((wl["datum"], wl["time_zone"], wl["units"], wl["format"]), ("MLLW", "gmt", "english", "json"))
        self.assertNotIn("datum", wind)

    def test_errors(self):
        self.state["force_error"] = True
        with self.assertRaises(noaa.NoaaError):
            noaa.fetch("TEST", "water_level", self.df.index[0], self.df.index[0] + pd.Timedelta(days=1), base_url=self.url)
        self.state.pop("force_error")
        self.state["missing"] = {"wind", "air_pressure"}
        out = noaa.build_dataset("TEST", self.df.index[0], self.df.index[0] + pd.Timedelta(days=3), base_url=self.url)
        self.assertTrue(out["wind_speed"].isna().all() and out["pressure"].isna().all())
        self.assertTrue(out["observed"].notna().any())   # tide data still usable without met sensors

    def test_model_trains_without_met_features(self):
        no_met = self.df.drop(columns=["wind_speed", "pressure"]).iloc[:9000]
        bundle, res = fit(no_met, float(no_met["observed"].quantile(0.99)))
        self.assertFalse(any("wind" in c or "pressure" in c for c in bundle["features"][3]))
        self.assertEqual(len(res), len(HORIZONS) * 4)

    def test_live_forecast_matches_offline_forecast(self):
        now = self.df.index[-60]
        res = latest_forecast(self.bundle, "TEST", "MLLW", now=now, base_url=self.url)
        self.assertEqual(res["last_obs_utc"], str(now))
        for h in HORIZONS:
            offline = forecast_series(self.bundle, self.df, h)[now + pd.Timedelta(hours=h)]
            self.assertAlmostEqual(res["horizons"][h]["forecast_ft"], offline, places=1,
                                   msg="live path and offline path disagree: plumbing bug")
        self.assertIn(res["status"], ("none", "WATCH", "ALERT"))

    def test_stale_data_refuses_to_forecast(self):
        with self.assertRaises(RuntimeError):
            latest_forecast(self.bundle, "TEST", "MLLW", now=self.df.index[-1] + pd.Timedelta(days=5), base_url=self.url)


if __name__ == "__main__":
    unittest.main()
