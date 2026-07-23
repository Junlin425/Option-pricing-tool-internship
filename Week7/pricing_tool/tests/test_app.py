import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from Week7.pricing_tool.model_inference_service import ThreeModelPricingResult
from Week7.pricing_tool.realtime_market_panel import load_realtime_market_snapshot
from Week7.pricing_tool.market_data_cache import CachedMarketDataResult, CacheStatus
from Week7.pricing_tool.tests.test_market_data_cache import make_market_bundle


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class PricingToolAppTests(unittest.TestCase):
    def test_default_page_runs_and_shows_known_prices(self):
        with patch(
            "Week7.pricing_tool.realtime_market_panel.load_realtime_market_snapshot"
        ) as realtime_loader:
            app = AppTest.from_file(str(APP_PATH)).run(timeout=20)

        self.assertFalse(app.exception)
        realtime_loader.assert_not_called()
        self.assertEqual(len(app.number_input), 5)
        self.assertEqual(len(app.button), 1)
        self.assertEqual(len(app.toggle), 1)
        self.assertEqual(app.toggle[0].label, "Enable automatic market updates")
        metric_values = [metric.value for metric in app.metric]
        self.assertIn("$10.4506", metric_values)
        self.assertIn("$5.5735", metric_values)
        self.assertTrue(app.success)
        tab_labels = [tab.label for tab in app.tabs]
        self.assertIn("Option Calculator", tab_labels)
        self.assertIn("Live Model Comparison", tab_labels)
        self.assertIn("Model Performance", tab_labels)
        page_text = " ".join(
            [item.value for item in app.header]
            + [item.value for item in app.subheader]
            + [item.value for item in app.caption]
            + [item.value for item in app.markdown]
        )
        self.assertIn("Sensitivity Analysis", page_text)
        self.assertIn("Test Prediction Trend", page_text)
        self.assertIn("MAE", page_text)

    def test_enabled_realtime_panel_renders_three_model_comparison(self):
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
        snapshot = load_realtime_market_snapshot(cache_loader=lambda **_: cached)
        result = ThreeModelPricingResult(
            market_date=snapshot.features.latest_market_date,
            spot=snapshot.latest_close,
            strike=snapshot.latest_close,
            maturity_years=1.0,
            risk_free_rate=snapshot.latest_treasury_yield / 100.0,
            lstm_call_price=12.25,
            linear_call_price=10.25,
            bsm_call_price=11.0,
            lstm_annual_volatility=0.22,
            baseline_annual_volatility=0.20,
            warnings=("Example model warning.",),
        )
        with (
            patch(
                "Week7.pricing_tool.realtime_market_panel.load_realtime_market_snapshot",
                return_value=snapshot,
            ),
            patch(
                "Week7.pricing_tool.realtime_market_panel.load_three_model_result",
                return_value=result,
            ),
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
            app.toggle[0].set_value(True)
            app.run(timeout=20)

        self.assertFalse(app.exception)
        text = " ".join(
            [item.value for item in app.header]
            + [item.value for item in app.subheader]
            + [item.value for item in app.caption]
            + [item.value for item in app.markdown]
        )
        self.assertIn("Live Three-Model CallPrice Comparison", text)
        self.assertIn("ATM", text)
        self.assertIn("1 year", text)
        metric_labels = [metric.label for metric in app.metric]
        self.assertIn("LSTM + BSM", metric_labels)
        self.assertIn("Linear Regression", metric_labels)
        self.assertIn("BSM Baseline", metric_labels)
        self.assertIn("Historical Error Reference", text)
        self.assertIn("not a confidence interval", text)
        self.assertTrue(any("research" in item.value.lower() for item in app.caption))
        self.assertTrue(app.warning)


if __name__ == "__main__":
    unittest.main()
