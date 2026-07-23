import unittest

import numpy as np
import pandas as pd

from Week7.pricing_tool.market_data_cache import (
    CachedMarketDataResult,
    CacheStatus,
)
from Week7.pricing_tool.realtime_market_panel import (
    RealtimeMarketSnapshot,
    build_three_model_comparison,
    load_three_model_result,
    load_realtime_market_snapshot,
)
from Week7.pricing_tool.model_inference_service import ThreeModelPricingResult
from Week7.pricing_tool.tests.test_market_data_cache import make_market_bundle


class RealtimeSnapshotTests(unittest.TestCase):
    def test_cache_bundle_is_composed_with_task_3_3_features(self):
        bundle = make_market_bundle()
        cached = CachedMarketDataResult(
            bundle=bundle,
            status=CacheStatus(
                source="live",
                cached_at_utc=pd.Timestamp("2026-07-16T12:00:00Z"),
                age_seconds=0.0,
                is_stale=False,
                refresh_attempted=True,
                warning=None,
            ),
        )
        calls = []

        snapshot = load_realtime_market_snapshot(
            force_refresh=True,
            cache_loader=lambda **kwargs: calls.append(kwargs) or cached,
        )

        self.assertIsInstance(snapshot, RealtimeMarketSnapshot)
        self.assertEqual(calls, [{"force_refresh": True}])
        self.assertIs(snapshot.cache, cached)
        self.assertEqual(snapshot.features.linear_latest.shape, (1, 10))
        self.assertEqual(snapshot.features.lstm_sequence.shape, (20, 13))
        latest = bundle.merged.iloc[-1]
        self.assertEqual(snapshot.latest_close, float(latest["Close"]))
        self.assertEqual(snapshot.latest_vix, float(latest["VIX_Close"]))
        self.assertEqual(
            snapshot.latest_treasury_yield,
            float(latest["Treasury_Yield"]),
        )

    def test_three_model_result_composes_snapshot_features_and_artifacts(self):
        snapshot = self._make_snapshot()
        fake_artifacts = object()
        expected = self._make_result()
        calls = []

        result = load_three_model_result(
            snapshot,
            artifact_loader=lambda: fake_artifacts,
            predictor=lambda features, artifacts: calls.append((features, artifacts)) or expected,
        )

        self.assertIs(result, expected)
        self.assertEqual(calls, [(snapshot.features, fake_artifacts)])

    def test_comparison_table_has_three_models_and_bsm_differences(self):
        result = self._make_result()

        comparison = build_three_model_comparison(result)

        self.assertEqual(
            comparison["Model"].tolist(),
            ["LSTM + BSM", "Linear Regression", "BSM Baseline"],
        )
        self.assertEqual(comparison.shape, (3, 4))
        self.assertAlmostEqual(comparison.loc[0, "Difference vs BSM"], 1.25)
        self.assertAlmostEqual(comparison.loc[1, "Difference vs BSM"], -0.75)
        self.assertEqual(comparison.loc[2, "Difference vs BSM"], 0.0)
        self.assertTrue(np.isnan(comparison.loc[1, "Annual Volatility"]))

    def _make_snapshot(self):
        bundle = make_market_bundle()
        cached = CachedMarketDataResult(
            bundle=bundle,
            status=CacheStatus(
                source="fresh_cache",
                cached_at_utc=pd.Timestamp("2026-07-16T12:00:00Z"),
                age_seconds=60.0,
                is_stale=False,
                refresh_attempted=False,
                warning=None,
            ),
        )
        return load_realtime_market_snapshot(cache_loader=lambda **_: cached)

    @staticmethod
    def _make_result():
        return ThreeModelPricingResult(
            market_date=pd.Timestamp("2026-07-15"),
            spot=150.0,
            strike=150.0,
            maturity_years=1.0,
            risk_free_rate=0.045,
            lstm_call_price=12.25,
            linear_call_price=10.25,
            bsm_call_price=11.0,
            lstm_annual_volatility=0.22,
            baseline_annual_volatility=0.20,
            warnings=(),
        )


if __name__ == "__main__":
    unittest.main()
