import unittest

import numpy as np
import pandas as pd

from Week7.pricing_tool.dashboard_service import (
    build_error_reference_table,
    build_sensitivity_curves,
    calculate_residual_reference,
    prepare_performance_dashboard,
)


class DashboardServiceTests(unittest.TestCase):
    def test_residual_reference_is_empirical_and_asymmetric(self):
        predictions = pd.DataFrame(
            {
                "Actual_CallPrice": [10.0, 12.0, 14.0, 18.0],
                "LSTM": [9.0, 11.0, 13.0, 15.0],
            }
        )

        result = calculate_residual_reference(
            predictions,
            actual_column="Actual_CallPrice",
            prediction_column="LSTM",
            lower_quantile=0.25,
            upper_quantile=0.75,
        )

        residuals = predictions["Actual_CallPrice"] - predictions["LSTM"]
        self.assertAlmostEqual(result["LowerResidual"], residuals.quantile(0.25))
        self.assertAlmostEqual(result["UpperResidual"], residuals.quantile(0.75))
        self.assertAlmostEqual(result["MAE"], np.abs(residuals).mean())
        self.assertAlmostEqual(result["RMSE"], np.sqrt(np.mean(residuals**2)))

    def test_error_table_adds_test_residuals_to_live_predictions(self):
        references = pd.DataFrame(
            [
                {
                    "Model": "LSTM + BSM",
                    "LowerResidual": -2.0,
                    "UpperResidual": 3.0,
                    "MAE": 1.0,
                    "RMSE": 1.5,
                },
                {
                    "Model": "Linear Regression",
                    "LowerResidual": -4.0,
                    "UpperResidual": 5.0,
                    "MAE": 2.0,
                    "RMSE": 3.0,
                },
            ]
        )

        result = build_error_reference_table(
            {"LSTM + BSM": 20.0, "Linear Regression": 18.0}, references
        )

        self.assertEqual(result["Model"].tolist(), ["LSTM + BSM", "Linear Regression"])
        self.assertEqual(result["Lower Reference"].tolist(), [18.0, 14.0])
        self.assertEqual(result["Upper Reference"].tolist(), [23.0, 23.0])

        low_price = build_error_reference_table(
            {"Linear Regression": 1.0}, references
        )
        self.assertEqual(low_price.loc[0, "Lower Reference"], 0.0)

    def test_performance_dashboard_selects_final_all_vix_models(self):
        metrics = pd.DataFrame(
            {
                "ModelFamily": ["Linear Regression", "LSTM + BSM", "Linear Regression"],
                "Variant": ["All VIX Features", "All VIX Features", "No VIX"],
                "Test_MAE": [2.7, 0.8, 3.0],
                "Test_RMSE": [4.3, 2.4, 4.5],
                "Test_R2": [0.77, 0.93, 0.70],
            }
        )
        predictions = pd.DataFrame(
            {
                "Date": ["2024-01-01"],
                "Actual_CallPrice": [10.0],
                "Linear_All_VIX_Features": [9.0],
                "LSTM_BSM_All_VIX_Features": [10.5],
                "Unused": [99.0],
            }
        )

        final_metrics, trend = prepare_performance_dashboard(metrics, predictions)

        self.assertEqual(final_metrics["Model"].tolist(), ["LSTM + BSM", "Linear Regression"])
        self.assertEqual(
            trend.columns.tolist(),
            ["Date", "Target CallPrice", "LSTM + BSM", "Linear Regression"],
        )
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(trend["Date"]))

    def test_sensitivity_curves_include_spot_volatility_and_rate(self):
        curves = build_sensitivity_curves(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            maturity=1.0,
            points=7,
        )

        self.assertEqual(set(curves), {"Spot", "Volatility", "Rate"})
        self.assertEqual(len(curves["Spot"]), 7)
        self.assertEqual(len(curves["Volatility"]), 7)
        self.assertEqual(len(curves["Rate"]), 7)
        self.assertTrue(curves["Spot"]["Call Price"].is_monotonic_increasing)
        self.assertTrue(curves["Volatility"]["Call Price"].is_monotonic_increasing)
        self.assertTrue(curves["Rate"]["Call Price"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
