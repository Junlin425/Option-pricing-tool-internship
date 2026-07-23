import os
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from Week7.pricing_tool.e2e_validation import (
    REPORT_COLUMNS,
    EndToEndValidationError,
    EndToEndValidationResult,
    validate_end_to_end,
    write_end_to_end_report,
)
from Week7.pricing_tool.feature_engineering_service import build_realtime_features
from Week7.pricing_tool.market_data_cache import CachedMarketDataResult, CacheStatus
from Week7.pricing_tool.market_data_cache import load_cached_market_data
from Week7.pricing_tool.market_data_service import load_market_data
from Week7.pricing_tool.model_inference_service import ThreeModelPricingResult
from Week7.pricing_tool.tests.test_market_data_cache import make_market_bundle


class FakeResponse:
    def __init__(self, *, payload=None, text="", status_code=200):
        self._payload = payload
        self.text = text
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class RoutingFakeHttpClient:
    def __init__(self, alpha_payload, cboe_text, fred_payload):
        self.alpha_payload = alpha_payload
        self.cboe_text = cboe_text
        self.fred_payload = fred_payload
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if "alphavantage" in url:
            return FakeResponse(payload=self.alpha_payload)
        if "cboe" in url:
            return FakeResponse(text=self.cboe_text)
        if "stlouisfed" in url:
            return FakeResponse(payload=self.fred_payload)
        return FakeResponse(status_code=404)


def make_routing_client(periods: int = 45) -> RoutingFakeHttpClient:
    dates = pd.bdate_range("2026-04-01", periods=periods)
    alpha_series = {}
    cboe_rows = ["DATE,OPEN,HIGH,LOW,CLOSE"]
    fred_observations = []
    for index, date in enumerate(dates):
        date_iso = date.strftime("%Y-%m-%d")
        close = 100.0 + index + 0.1 * (index % 3)
        vix = 18.0 + 0.08 * index + 0.2 * (index % 4)
        alpha_series[date_iso] = {
            "1. open": f"{close - 0.5:.4f}",
            "2. high": f"{close + 1.0:.4f}",
            "3. low": f"{close - 1.0:.4f}",
            "4. close": f"{close:.4f}",
            "5. volume": str(1_000_000 + 10_000 * index),
        }
        cboe_rows.append(
            f"{date.strftime('%m/%d/%Y')},{vix - 0.2:.4f},{vix + 0.5:.4f},"
            f"{vix - 0.5:.4f},{vix:.4f}"
        )
        fred_observations.append(
            {"date": date_iso, "value": f"{4.0 + 0.01 * index:.4f}"}
        )
    return RoutingFakeHttpClient(
        {"Time Series (Daily)": alpha_series},
        "\n".join(cboe_rows),
        {"observations": fred_observations},
    )


def make_cached_result(source: str = "live") -> CachedMarketDataResult:
    cached_at = pd.Timestamp("2026-07-16T12:00:00Z")
    return CachedMarketDataResult(
        bundle=make_market_bundle(),
        status=CacheStatus(
            source=source,
            cached_at_utc=cached_at,
            age_seconds=0.0 if source == "live" else 1800.0,
            is_stale=source == "stale_fallback",
            refresh_attempted=source != "fresh_cache",
            warning="Safe stale warning." if source == "stale_fallback" else None,
        ),
    )


def fake_three_model_predictor(features, artifacts):
    latest = features.history.iloc[-1]
    return ThreeModelPricingResult(
        market_date=features.latest_market_date,
        spot=float(latest["Close"]),
        strike=float(latest["Close"]),
        maturity_years=1.0,
        risk_free_rate=float(latest["Treasury_Yield"]) / 100.0,
        lstm_call_price=12.0,
        linear_call_price=11.0,
        bsm_call_price=10.0,
        lstm_annual_volatility=0.22,
        baseline_annual_volatility=0.20,
        warnings=("Test extrapolation warning.",),
    )


def validate_with_fake_models(**kwargs):
    return validate_end_to_end(
        artifact_loader=lambda: SimpleNamespace(manifest={"schema_version": 1}),
        predictor=fake_three_model_predictor,
        **kwargs,
    )


class EndToEndValidatorTests(unittest.TestCase):
    def test_valid_pipeline_result_passes_data_and_model_checks(self):
        cached = make_cached_result()
        calls = []

        result = validate_with_fake_models(
            cache_loader=lambda **kwargs: calls.append(kwargs) or cached,
            force_refresh=True,
            run_at_utc=pd.Timestamp("2026-07-16T12:01:00Z"),
        )

        self.assertIsInstance(result, EndToEndValidationResult)
        self.assertEqual(calls, [{"force_refresh": True}])
        self.assertEqual(result.overall_status, "PASS")
        self.assertEqual(result.report.shape, (1, len(REPORT_COLUMNS)))
        self.assertEqual(list(result.report.columns), REPORT_COLUMNS)
        row = result.report.iloc[0]
        self.assertEqual(row["OverallStatus"], "PASS")
        self.assertEqual(int(row["PassedChecks"]), 29)
        self.assertEqual(int(row["TotalChecks"]), 29)
        self.assertEqual(row["FailedChecks"], "")
        self.assertEqual((int(row["LinearRows"]), int(row["LinearFeatures"])), (1, 10))
        self.assertEqual((int(row["LSTMRows"]), int(row["LSTMFeatures"])), (20, 13))
        self.assertEqual(float(row["LSTMCallPrice"]), 12.0)
        self.assertEqual(float(row["LinearCallPrice"]), 11.0)
        self.assertEqual(float(row["BSMBaselineCallPrice"]), 10.0)
        self.assertEqual(int(row["ModelWarningCount"]), 1)

    def test_invalid_linear_shape_returns_fail_with_stable_check_name(self):
        cached = make_cached_result()

        def bad_feature_builder(bundle):
            features = build_realtime_features(bundle)
            return replace(
                features,
                linear_latest=features.linear_latest.iloc[:, :9].copy(),
            )

        result = validate_with_fake_models(
            cache_loader=lambda **_: cached,
            feature_builder=bad_feature_builder,
            run_at_utc=pd.Timestamp("2026-07-16T12:01:00Z"),
        )

        self.assertEqual(result.overall_status, "FAIL")
        row = result.report.iloc[0]
        self.assertEqual(row["OverallStatus"], "FAIL")
        self.assertIn("linear_shape", row["FailedChecks"].split(";"))
        self.assertIn("linear_columns", row["FailedChecks"].split(";"))

    def test_report_writer_round_trips_one_row_and_uses_atomic_replace(self):
        result = validate_with_fake_models(
            cache_loader=lambda **_: make_cached_result(),
            run_at_utc=pd.Timestamp("2026-07-16T12:01:00Z"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.csv"

            returned_path = write_end_to_end_report(result, path)
            loaded = pd.read_csv(returned_path, keep_default_na=False)

            self.assertEqual(returned_path, path)
            self.assertEqual(loaded.shape, (1, len(REPORT_COLUMNS)))
            self.assertEqual(list(loaded.columns), REPORT_COLUMNS)
            self.assertEqual(loaded.loc[0, "OverallStatus"], "PASS")
            self.assertEqual(loaded.loc[0, "FailedChecks"], "")

    def test_failed_report_replace_preserves_existing_file(self):
        result = validate_with_fake_models(
            cache_loader=lambda **_: make_cached_result(),
            run_at_utc=pd.Timestamp("2026-07-16T12:01:00Z"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.csv"
            path.write_text("previous-report", encoding="utf-8")
            original = path.read_bytes()

            with patch(
                "Week7.pricing_tool.e2e_validation.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaises(EndToEndValidationError):
                    write_end_to_end_report(result, path)

            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            self.assertEqual(
                [item for item in Path(directory).iterdir() if item.name.startswith(".")],
                [],
            )

    def test_non_finite_model_output_returns_stable_failed_check(self):
        def invalid_predictor(features, artifacts):
            result = fake_three_model_predictor(features, artifacts)
            return replace(result, linear_call_price=float("nan"))

        result = validate_end_to_end(
            cache_loader=lambda **_: make_cached_result(),
            artifact_loader=lambda: object(),
            predictor=invalid_predictor,
            run_at_utc=pd.Timestamp("2026-07-16T12:01:00Z"),
        )

        self.assertEqual(result.overall_status, "FAIL")
        self.assertIn(
            "linear_call_price_finite",
            result.report.loc[0, "FailedChecks"].split(";"),
        )


class FullPipelineIntegrationTests(unittest.TestCase):
    NOW = pd.Timestamp("2026-07-16T12:00:00Z")

    def _controlled_market_loader(self, client):
        return lambda: load_market_data(http_client=client)

    def test_controlled_provider_responses_reach_live_cache_and_features(self):
        client = make_routing_client()
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.pkl"
            with patch(
                "Week7.pricing_tool.market_data_service.load_api_keys",
                return_value=SimpleNamespace(
                    alpha_vantage="fake-alpha-key",
                    fred="fake-fred-key",
                ),
            ):
                cached = load_cached_market_data(
                    force_refresh=True,
                    market_data_loader=self._controlled_market_loader(client),
                    cache_path=cache_path,
                    now_utc=self.NOW,
                )
            result = validate_with_fake_models(
                cache_loader=lambda **_: cached,
                run_at_utc=self.NOW,
            )

            self.assertEqual(len(client.calls), 3)
            self.assertEqual(cached.status.source, "live")
            self.assertTrue(cache_path.is_file())
            self.assertEqual(result.overall_status, "PASS")
            row = result.report.iloc[0]
            self.assertEqual(int(row["PassedChecks"]), 29)
            self.assertEqual(int(row["JPMRows"]), 45)
            self.assertEqual(int(row["MergedRows"]), 45)
            self.assertEqual((int(row["LinearRows"]), int(row["LinearFeatures"])), (1, 10))
            self.assertEqual((int(row["LSTMRows"]), int(row["LSTMFeatures"])), (20, 13))

    def test_immediate_second_run_uses_fresh_cache_without_provider(self):
        client = make_routing_client()
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.pkl"
            with patch(
                "Week7.pricing_tool.market_data_service.load_api_keys",
                return_value=SimpleNamespace(
                    alpha_vantage="fake-alpha-key",
                    fred="fake-fred-key",
                ),
            ):
                load_cached_market_data(
                    force_refresh=True,
                    market_data_loader=self._controlled_market_loader(client),
                    cache_path=cache_path,
                    now_utc=self.NOW,
                )

            def forbidden_loader():
                raise AssertionError("provider must not be called for fresh cache")

            cached = load_cached_market_data(
                market_data_loader=forbidden_loader,
                cache_path=cache_path,
                now_utc=self.NOW + pd.Timedelta(minutes=1),
            )
            result = validate_with_fake_models(
                cache_loader=lambda **_: cached,
                run_at_utc=self.NOW + pd.Timedelta(minutes=1),
            )

            self.assertEqual(cached.status.source, "fresh_cache")
            self.assertEqual(result.overall_status, "PASS")
            self.assertEqual(int(result.report.loc[0, "PassedChecks"]), 29)

    def test_failed_refresh_uses_stale_cache_and_still_passes_pipeline(self):
        client = make_routing_client()
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.pkl"
            with patch(
                "Week7.pricing_tool.market_data_service.load_api_keys",
                return_value=SimpleNamespace(
                    alpha_vantage="fake-alpha-key",
                    fred="fake-fred-key",
                ),
            ):
                load_cached_market_data(
                    force_refresh=True,
                    market_data_loader=self._controlled_market_loader(client),
                    cache_path=cache_path,
                    now_utc=self.NOW,
                )

            def failing_loader():
                raise RuntimeError("fake-secret-value")

            cached = load_cached_market_data(
                market_data_loader=failing_loader,
                cache_path=cache_path,
                now_utc=self.NOW + pd.Timedelta(hours=2),
            )
            result = validate_with_fake_models(
                cache_loader=lambda **_: cached,
                run_at_utc=self.NOW + pd.Timedelta(hours=2),
            )
            serialized = result.report.to_csv(index=False)

            self.assertEqual(cached.status.source, "stale_fallback")
            self.assertTrue(cached.status.is_stale)
            self.assertEqual(result.overall_status, "PASS")
            self.assertEqual(int(result.report.loc[0, "PassedChecks"]), 29)
            self.assertNotIn("fake-secret-value", serialized)


if __name__ == "__main__":
    unittest.main()
