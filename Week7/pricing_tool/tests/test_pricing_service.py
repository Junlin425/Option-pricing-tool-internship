import unittest

import numpy as np

from Week7.pricing_tool.pricing_service import (
    build_spot_price_curve,
    calculate_bsm_prices,
    calculate_from_percent_inputs,
)


class BsmPricingTests(unittest.TestCase):
    def test_classic_bsm_example(self):
        result = calculate_bsm_prices(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            maturity=1.0,
        )

        self.assertAlmostEqual(result["call_price"], 10.4506, places=4)
        self.assertAlmostEqual(result["put_price"], 5.5735, places=4)

    def test_put_call_parity_residual_is_near_zero(self):
        result = calculate_bsm_prices(100.0, 100.0, 0.05, 0.20, 1.0)

        self.assertAlmostEqual(result["parity_residual"], 0.0, places=10)

    def test_percent_adapter_matches_decimal_service_inputs(self):
        direct = calculate_bsm_prices(100.0, 100.0, 0.05, 0.20, 1.0)
        adapted = calculate_from_percent_inputs(
            100.0,
            100.0,
            rate_percent=5.0,
            volatility_percent=20.0,
            maturity=1.0,
        )

        for key in direct:
            self.assertAlmostEqual(adapted[key], direct[key], places=12)

    def test_non_positive_constrained_inputs_are_rejected(self):
        invalid_cases = [
            ((0.0, 100.0, 0.05, 0.20, 1.0), "spot"),
            ((100.0, -1.0, 0.05, 0.20, 1.0), "strike"),
            ((100.0, 100.0, 0.05, 0.0, 1.0), "volatility"),
            ((100.0, 100.0, 0.05, 0.20, 0.0), "maturity"),
        ]
        for values, field in invalid_cases:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    calculate_bsm_prices(*values)

    def test_non_finite_inputs_are_rejected_but_negative_rate_is_allowed(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "finite"):
                    calculate_bsm_prices(100.0, 100.0, invalid, 0.20, 1.0)

        result = calculate_bsm_prices(100.0, 100.0, -0.01, 0.20, 1.0)
        self.assertTrue(np.isfinite(list(result.values())).all())


class SpotCurveTests(unittest.TestCase):
    def test_curve_shape_order_and_option_price_direction(self):
        curve = build_spot_price_curve(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.20,
            maturity=1.0,
            points=5,
        )

        self.assertEqual(list(curve.columns), ["Spot", "Call Price", "Put Price"])
        self.assertEqual(len(curve), 5)
        self.assertAlmostEqual(curve["Spot"].iloc[0], 70.0)
        self.assertAlmostEqual(curve["Spot"].iloc[-1], 130.0)
        self.assertTrue(curve["Spot"].is_monotonic_increasing)
        self.assertTrue(curve["Call Price"].is_monotonic_increasing)
        self.assertTrue(curve["Put Price"].is_monotonic_decreasing)
        self.assertTrue(np.isfinite(curve.to_numpy()).all())

    def test_invalid_curve_configuration_is_rejected(self):
        invalid_settings = [
            {"lower_multiplier": 0.0},
            {"lower_multiplier": 1.3, "upper_multiplier": 0.7},
            {"points": 1},
            {"upper_multiplier": np.inf},
        ]
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    build_spot_price_curve(100, 100, 0.05, 0.20, 1.0, **settings)


if __name__ == "__main__":
    unittest.main()
