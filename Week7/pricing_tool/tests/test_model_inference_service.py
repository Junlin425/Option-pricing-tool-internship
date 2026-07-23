from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Week7.pricing_tool.feature_engineering_service import (
    LINEAR_FEATURES,
    LSTM_FEATURES,
    build_realtime_features,
)
from Week7.pricing_tool.model_inference_service import (
    LoadedModelArtifacts,
    ModelInferenceError,
    load_model_artifacts,
    predict_three_model_call_prices,
)
from Week7.pricing_tool.tests.test_market_data_cache import make_market_bundle


class RecordingLinearModel:
    n_features_in_ = 10

    def __init__(self, prediction=12.0):
        self.prediction = prediction
        self.last_input = None

    def predict(self, values):
        self.last_input = np.asarray(values, dtype=float).copy()
        return np.asarray([self.prediction], dtype=float)


class RecordingLstmModel:
    input_shape = (None, 20, 13)

    def __init__(self, scaled_delta=0.001):
        self.scaled_delta = scaled_delta
        self.last_input = None

    def predict(self, values, verbose=0):
        self.last_input = np.asarray(values, dtype=float).copy()
        return np.asarray([[self.scaled_delta]], dtype=float)


class RecordingScaler:
    def __init__(self, n_features_in, inverse_shift=0.0):
        self.n_features_in_ = n_features_in
        self.inverse_shift = inverse_shift
        self.last_input = None

    def transform(self, values):
        self.last_input = np.asarray(values, dtype=float).copy()
        return self.last_input

    def inverse_transform(self, values):
        self.last_input = np.asarray(values, dtype=float).copy()
        return self.last_input + self.inverse_shift


def make_artifacts(linear_prediction=12.0, scaled_delta=0.001):
    feature_ranges = {
        "linear": {
            feature: {"min": -1e9, "max": 1e9} for feature in LINEAR_FEATURES
        },
        "lstm": {
            feature: {"min": -1e9, "max": 1e9} for feature in LSTM_FEATURES
        },
    }
    return LoadedModelArtifacts(
        linear_model=RecordingLinearModel(linear_prediction),
        lstm_model=RecordingLstmModel(scaled_delta),
        x_scaler=RecordingScaler(13),
        delta_scaler=RecordingScaler(1),
        manifest={
            "schema_version": 1,
            "variant": "All VIX Features",
            "window_size": 20,
            "linear_features": LINEAR_FEATURES,
            "lstm_features": LSTM_FEATURES,
            "feature_ranges": feature_ranges,
        },
    )


class ThreeModelInferenceTests(unittest.TestCase):
    def setUp(self):
        self.features = build_realtime_features(make_market_bundle())

    def test_latest_date_prediction_uses_preceding_twenty_rows(self):
        artifacts = make_artifacts()

        result = predict_three_model_call_prices(self.features, artifacts)

        expected_lstm = self.features.history.iloc[-21:-1][LSTM_FEATURES].to_numpy(float)
        np.testing.assert_allclose(artifacts.x_scaler.last_input, expected_lstm)
        self.assertEqual(artifacts.lstm_model.last_input.shape, (1, 20, 13))
        expected_linear = self.features.history.iloc[-1][LINEAR_FEATURES].to_numpy(float)
        np.testing.assert_allclose(
            artifacts.linear_model.last_input,
            expected_linear.reshape(1, -1),
        )
        self.assertEqual(result.market_date, self.features.latest_market_date)
        self.assertEqual(result.spot, float(self.features.history.iloc[-1]["Close"]))
        self.assertEqual(result.strike, result.spot)
        self.assertEqual(result.maturity_years, 1.0)

    def test_lstm_delta_is_added_to_previous_volatility_then_annualized(self):
        artifacts = make_artifacts(scaled_delta=0.002)

        result = predict_three_model_call_prices(self.features, artifacts)

        expected_daily = float(self.features.history.iloc[-2]["RollingVol20"]) + 0.002
        self.assertAlmostEqual(
            result.lstm_annual_volatility,
            expected_daily * np.sqrt(252.0),
        )
        self.assertAlmostEqual(
            result.baseline_annual_volatility,
            float(self.features.history.iloc[-1]["RollingVol20"]) * np.sqrt(252.0),
        )
        self.assertTrue(np.isfinite(result.lstm_call_price))
        self.assertTrue(np.isfinite(result.bsm_call_price))

    def test_fewer_than_twenty_one_history_rows_is_rejected(self):
        short = replace(self.features, history=self.features.history.tail(20).copy())

        with self.assertRaisesRegex(ModelInferenceError, "21 valid feature rows"):
            predict_three_model_call_prices(short, make_artifacts())

    def test_missing_or_non_finite_history_feature_is_rejected(self):
        missing = replace(
            self.features,
            history=self.features.history.drop(columns=["VIX_Return"]),
        )
        with self.assertRaisesRegex(ModelInferenceError, "missing required columns"):
            predict_three_model_call_prices(missing, make_artifacts())

        invalid_history = self.features.history.copy()
        invalid_history.loc[invalid_history.index[-1], "Close"] = np.inf
        invalid = replace(self.features, history=invalid_history)
        with self.assertRaisesRegex(ModelInferenceError, "finite"):
            predict_three_model_call_prices(invalid, make_artifacts())

    def test_out_of_training_range_returns_warning_without_changing_input(self):
        artifacts = make_artifacts()
        live_close = float(self.features.history.iloc[-1]["Close"])
        artifacts.manifest["feature_ranges"]["linear"]["Close"] = {
            "min": live_close - 10.0,
            "max": live_close - 1.0,
        }

        result = predict_three_model_call_prices(self.features, artifacts)

        self.assertTrue(any("Close" in warning and "training range" in warning for warning in result.warnings))
        self.assertEqual(
            artifacts.linear_model.last_input[0, LINEAR_FEATURES.index("Close")],
            live_close,
        )

    def test_no_arbitrage_violation_is_warned_and_not_clipped(self):
        spot = float(self.features.history.iloc[-1]["Close"])
        artifacts = make_artifacts(linear_prediction=spot + 5.0)

        result = predict_three_model_call_prices(self.features, artifacts)

        self.assertEqual(result.linear_call_price, spot + 5.0)
        self.assertTrue(
            any("Linear Regression" in warning and "no-arbitrage" in warning for warning in result.warnings)
        )

    def test_non_positive_lstm_volatility_is_rejected(self):
        base = float(self.features.history.iloc[-2]["RollingVol20"])

        with self.assertRaisesRegex(ModelInferenceError, "positive volatility"):
            predict_three_model_call_prices(
                self.features,
                make_artifacts(scaled_delta=-(base + 0.001)),
            )


class ArtifactManifestLoadingTests(unittest.TestCase):
    def test_missing_manifest_error_includes_resolved_path_and_cause(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = str((Path(directory) / "manifest.json").resolve())

            with self.assertRaises(ModelInferenceError) as context:
                load_model_artifacts(directory)

        message = str(context.exception)
        self.assertIn(expected, message)
        self.assertIn("FileNotFoundError", message)

    def test_wrong_feature_contract_is_rejected_before_binary_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "schema_version": 1,
                "variant": "All VIX Features",
                "window_size": 20,
                "linear_features": list(reversed(LINEAR_FEATURES)),
                "lstm_features": LSTM_FEATURES,
                "feature_ranges": {"linear": {}, "lstm": {}},
                "hashes": {},
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ModelInferenceError, "Linear feature order"):
                load_model_artifacts(root)


if __name__ == "__main__":
    unittest.main()
