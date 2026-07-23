import unittest
from dataclasses import replace
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from Week7.pricing_tool.feature_engineering_service import (
    FeatureEngineeringError,
    LINEAR_FEATURES,
    LSTM_FEATURES,
    RealTimeFeatureBundle,
    build_realtime_features,
    load_realtime_features,
)
from Week7.pricing_tool.market_data_service import MarketDataBundle


def make_market_bundle(periods: int = 45) -> MarketDataBundle:
    dates = pd.bdate_range("2026-04-01", periods=periods)
    step = np.arange(periods, dtype=float)
    close = 100.0 * np.exp(0.001 * step + 0.002 * np.sin(step / 2.5))
    volume = 1_000_000.0 + 10_000.0 * step + 20_000.0 * np.sin(step / 3.0)
    vix_close = 18.0 + 0.08 * step + 1.2 * np.sin(step / 3.5)
    treasury_yield = 4.0 + 0.01 * step + 0.03 * np.cos(step / 4.0)
    merged = pd.DataFrame(
        {
            "Date": dates,
            "Close": close,
            "Volume": volume,
            "VIX_Close": vix_close,
            "Treasury_Yield": treasury_yield,
        }
    )
    return MarketDataBundle(
        jpm=pd.DataFrame(),
        vix=pd.DataFrame(),
        treasury=pd.DataFrame(),
        merged=merged,
        metadata=pd.DataFrame(),
    )


class FeatureFormulaTests(unittest.TestCase):
    def test_historical_formulas_are_reproduced_exactly(self):
        market_bundle = make_market_bundle()

        features = build_realtime_features(market_bundle)

        source = market_bundle.merged.reset_index(drop=True)
        first = features.history.iloc[0]
        source_index = 20
        returns = np.log(source["Close"] / source["Close"].shift(1))
        expected_vix_window = source.loc[1:20, "VIX_Close"]

        self.assertAlmostEqual(
            first["Return"],
            np.log(source.loc[source_index, "Close"] / source.loc[source_index - 1, "Close"]),
        )
        self.assertAlmostEqual(first["MA5"], source.loc[16:20, "Close"].mean())
        self.assertAlmostEqual(first["MA20"], source.loc[1:20, "Close"].mean())
        self.assertAlmostEqual(first["MA5_Ratio"], first["Close"] / first["MA5"] - 1.0)
        self.assertAlmostEqual(first["MA20_Ratio"], first["Close"] / first["MA20"] - 1.0)
        self.assertAlmostEqual(first["RollingVol20"], returns.loc[1:20].std(ddof=1))
        self.assertAlmostEqual(
            first["VolumeChange"],
            source.loc[20, "Volume"] / source.loc[19, "Volume"] - 1.0,
        )
        self.assertAlmostEqual(
            first["RateMomentum"],
            source.loc[20, "Treasury_Yield"] - source.loc[15, "Treasury_Yield"],
        )
        self.assertAlmostEqual(
            first["VIX_Return"],
            np.log(source.loc[20, "VIX_Close"] / source.loc[19, "VIX_Close"]),
        )
        self.assertAlmostEqual(
            first["VIX_5D_Change"],
            source.loc[20, "VIX_Close"] / source.loc[15, "VIX_Close"] - 1.0,
        )
        self.assertAlmostEqual(
            first["VIX_20D_ZScore"],
            (source.loc[20, "VIX_Close"] - expected_vix_window.mean())
            / expected_vix_window.std(ddof=0),
        )

    def test_bundle_has_exact_model_feature_orders_and_shapes(self):
        features = build_realtime_features(make_market_bundle())

        self.assertIsInstance(features, RealTimeFeatureBundle)
        self.assertEqual(list(features.linear_latest.columns), LINEAR_FEATURES)
        self.assertEqual(list(features.lstm_sequence.columns), LSTM_FEATURES)
        self.assertEqual(features.linear_latest.shape, (1, 10))
        self.assertEqual(features.lstm_sequence.shape, (20, 13))
        self.assertEqual(len(features.history), 25)
        self.assertEqual(features.history.iloc[0]["Date"], make_market_bundle().merged.iloc[20]["Date"])
        self.assertEqual(features.latest_market_date, features.history.iloc[-1]["Date"])


class ThresholdAndLeakageTests(unittest.TestCase):
    def test_regime_boundaries_use_fixed_training_thresholds(self):
        low = 16.70829978942871
        high = 23.233899631500247
        cases = [(low - 1e-6, 0), (low, 1), (high, 1), (high + 1e-6, 2)]

        for latest_vix, expected_regime in cases:
            with self.subTest(latest_vix=latest_vix):
                market_bundle = make_market_bundle()
                changed = market_bundle.merged.copy()
                changed.loc[changed.index[-1], "VIX_Close"] = latest_vix

                result = build_realtime_features(replace(market_bundle, merged=changed))

                self.assertEqual(int(result.history.iloc[-1]["VIX_Regime"]), expected_regime)

    def test_changing_a_future_row_does_not_change_an_earlier_feature_row(self):
        market_bundle = make_market_bundle()
        before = build_realtime_features(market_bundle).history.iloc[5][MODEL_ASSERT_COLUMNS]
        changed = market_bundle.merged.copy()
        changed.loc[changed.index[-1], ["Close", "Volume", "VIX_Close"]] *= 2.0

        after = build_realtime_features(
            replace(market_bundle, merged=changed)
        ).history.iloc[5][MODEL_ASSERT_COLUMNS]

        pd.testing.assert_series_equal(before, after)


MODEL_ASSERT_COLUMNS = list(dict.fromkeys(LINEAR_FEATURES + LSTM_FEATURES))


class ValidationAndQualityTests(unittest.TestCase):
    def test_missing_required_column_is_rejected(self):
        market_bundle = make_market_bundle()
        changed = market_bundle.merged.drop(columns="Volume")

        with self.assertRaisesRegex(FeatureEngineeringError, "required columns"):
            build_realtime_features(replace(market_bundle, merged=changed))

    def test_duplicate_and_unordered_dates_are_rejected(self):
        market_bundle = make_market_bundle()
        duplicate = market_bundle.merged.copy()
        duplicate.loc[10, "Date"] = duplicate.loc[9, "Date"]
        descending = market_bundle.merged.iloc[::-1].reset_index(drop=True)

        with self.assertRaisesRegex(FeatureEngineeringError, "duplicate"):
            build_realtime_features(replace(market_bundle, merged=duplicate))
        with self.assertRaisesRegex(FeatureEngineeringError, "increasing"):
            build_realtime_features(replace(market_bundle, merged=descending))

    def test_invalid_market_numbers_are_rejected(self):
        for column, value in [
            ("Close", 0.0),
            ("VIX_Close", -1.0),
            ("Volume", np.nan),
            ("Treasury_Yield", np.inf),
        ]:
            with self.subTest(column=column, value=value):
                market_bundle = make_market_bundle()
                changed = market_bundle.merged.copy()
                changed.loc[25, column] = value

                with self.assertRaisesRegex(FeatureEngineeringError, "numeric"):
                    build_realtime_features(replace(market_bundle, merged=changed))

    def test_insufficient_history_and_invalid_window_are_rejected(self):
        with self.assertRaisesRegex(FeatureEngineeringError, "20 valid"):
            build_realtime_features(make_market_bundle(periods=39))
        with self.assertRaisesRegex(FeatureEngineeringError, "window"):
            build_realtime_features(make_market_bundle(), lstm_window=0)

    def test_zero_vix_variance_is_rejected(self):
        market_bundle = make_market_bundle()
        changed = market_bundle.merged.copy()
        changed["VIX_Close"] = 18.0

        with self.assertRaisesRegex(FeatureEngineeringError, "VIX.*standard deviation"):
            build_realtime_features(replace(market_bundle, merged=changed))

    def test_malformed_and_inconsistent_thresholds_are_rejected(self):
        bad_configs = [
            {"regime_low_q33": 17.0, "regime_high_q67": 16.0, "spike_return_q95": 0.1, "threshold_source": "train_only"},
            {"regime_low_q33": 16.0, "regime_high_q67": 23.0, "threshold_source": "train_only"},
            {"regime_low_q33": 16.0, "regime_high_q67": 23.0, "spike_return_q95": "bad", "threshold_source": "train_only"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, config in enumerate(bad_configs):
                with self.subTest(config=config):
                    path = Path(directory) / f"bad_{index}.json"
                    path.write_text(json.dumps(config), encoding="utf-8")
                    with self.assertRaisesRegex(FeatureEngineeringError, "threshold"):
                        build_realtime_features(make_market_bundle(), thresholds_path=path)

    def test_quality_summary_reports_the_feature_contract(self):
        result = build_realtime_features(make_market_bundle())

        self.assertEqual(
            list(result.quality.columns),
            [
                "RawRowCount",
                "ValidFeatureRowCount",
                "WarmupRowCount",
                "LatestMarketDate",
                "LinearFeatureCount",
                "LSTMWindowRows",
                "LSTMFeatureCount",
                "MissingModelValues",
                "InfiniteModelValues",
                "ThresholdSource",
            ],
        )
        quality = result.quality.iloc[0]
        self.assertEqual(int(quality["RawRowCount"]), 45)
        self.assertEqual(int(quality["ValidFeatureRowCount"]), 25)
        self.assertEqual(int(quality["WarmupRowCount"]), 20)
        self.assertEqual(int(quality["LinearFeatureCount"]), 10)
        self.assertEqual(int(quality["LSTMWindowRows"]), 20)
        self.assertEqual(int(quality["LSTMFeatureCount"]), 13)
        self.assertEqual(int(quality["MissingModelValues"]), 0)
        self.assertEqual(int(quality["InfiniteModelValues"]), 0)
        self.assertEqual(quality["ThresholdSource"], "train_only")
        self.assertEqual(quality["LatestMarketDate"], result.latest_market_date)

    def test_public_loader_calls_the_injected_market_data_loader_once(self):
        calls = []
        market_bundle = make_market_bundle()

        result = load_realtime_features(
            market_data_loader=lambda: calls.append("called") or market_bundle
        )

        self.assertEqual(calls, ["called"])
        self.assertIsInstance(result, RealTimeFeatureBundle)


if __name__ == "__main__":
    unittest.main()
