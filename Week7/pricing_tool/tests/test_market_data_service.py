import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from Week7.pricing_tool.market_data_service import (
    MarketDataBundle,
    MarketDataError,
    align_market_data,
    build_metadata,
    fetch_jpm_daily,
    fetch_treasury_daily,
    fetch_vix_daily,
    load_market_data,
)


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


class FakeHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class ProviderAdapterTests(unittest.TestCase):
    def test_alpha_vantage_daily_data_is_standardized_and_sorted(self):
        payload = {
            "Time Series (Daily)": {
                "2026-07-15": {
                    "1. open": "301.00",
                    "2. high": "305.00",
                    "3. low": "299.00",
                    "4. close": "304.00",
                    "5. volume": "12000000",
                },
                "2026-07-14": {
                    "1. open": "298.00",
                    "2. high": "302.00",
                    "3. low": "297.00",
                    "4. close": "300.00",
                    "5. volume": "11000000",
                },
            }
        }
        client = FakeHttpClient(FakeResponse(payload=payload))

        frame = fetch_jpm_daily("fake-alpha-key", http_client=client)

        self.assertEqual(
            list(frame.columns),
            ["Date", "Open", "High", "Low", "Close", "Volume"],
        )
        self.assertEqual(frame["Date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-07-14", "2026-07-15"])
        self.assertEqual(frame["Close"].tolist(), [300.0, 304.0])
        self.assertEqual(frame["Volume"].tolist(), [11000000, 12000000])

    def test_alpha_provider_error_does_not_expose_supplied_key(self):
        fake_key = "fake-alpha-key-that-must-not-appear"
        client = FakeHttpClient(FakeResponse(payload={"Information": "rate limited"}))

        with self.assertRaises(MarketDataError) as caught:
            fetch_jpm_daily(fake_key, http_client=client)

        self.assertIn("Alpha Vantage", str(caught.exception))
        self.assertNotIn(fake_key, str(caught.exception))

    def test_cboe_csv_is_standardized(self):
        csv_text = (
            "DATE,OPEN,HIGH,LOW,CLOSE\n"
            "07/15/2026,15.82,16.50,15.20,16.34\n"
            "07/14/2026,16.10,16.20,15.40,15.67\n"
        )
        client = FakeHttpClient(FakeResponse(text=csv_text))

        frame = fetch_vix_daily(http_client=client)

        self.assertEqual(
            list(frame.columns),
            ["Date", "VIX_Open", "VIX_High", "VIX_Low", "VIX_Close"],
        )
        self.assertEqual(frame["Date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-07-14", "2026-07-15"])
        self.assertEqual(frame["VIX_Close"].tolist(), [15.67, 16.34])

    def test_fred_missing_marker_is_removed_and_units_are_preserved(self):
        payload = {
            "observations": [
                {"date": "2026-07-13", "value": "4.44"},
                {"date": "2026-07-14", "value": "."},
                {"date": "2026-07-15", "value": "4.56"},
            ]
        }
        client = FakeHttpClient(FakeResponse(payload=payload))

        frame = fetch_treasury_daily(
            "fake-fred-key",
            observation_start=date(2026, 7, 1),
            http_client=client,
        )

        self.assertEqual(
            list(frame.columns),
            ["Treasury_Observation_Date", "Treasury_Yield"],
        )
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame["Treasury_Yield"].tolist(), [4.44, 4.56])


def make_standard_frames(periods=25):
    dates = pd.bdate_range("2026-05-01", periods=periods)
    jpm = pd.DataFrame(
        {
            "Date": dates,
            "Open": 100.0,
            "High": 102.0,
            "Low": 99.0,
            "Close": 101.0,
            "Volume": 1_000_000,
        }
    )
    vix = pd.DataFrame(
        {
            "Date": dates,
            "VIX_Open": 18.0,
            "VIX_High": 19.0,
            "VIX_Low": 17.0,
            "VIX_Close": 18.5,
        }
    )
    treasury = pd.DataFrame(
        {
            "Treasury_Observation_Date": dates,
            "Treasury_Yield": 4.5,
        }
    )
    return jpm, vix, treasury


class AlignmentAndMetadataTests(unittest.TestCase):
    def test_treasury_uses_latest_observation_on_or_before_jpm_date(self):
        jpm, vix, treasury = make_standard_frames()
        missing_date = treasury.loc[10, "Treasury_Observation_Date"]
        treasury = treasury.drop(index=10).reset_index(drop=True)

        merged = align_market_data(jpm, vix, treasury)

        self.assertEqual(len(merged), 25)
        self.assertEqual(
            list(merged.columns),
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "VIX_Open",
                "VIX_High",
                "VIX_Low",
                "VIX_Close",
                "Treasury_Observation_Date",
                "Treasury_Yield",
                "Treasury_Staleness_Days",
            ],
        )
        target = merged.loc[merged["Date"].eq(missing_date)].iloc[0]
        self.assertLess(target["Treasury_Observation_Date"], target["Date"])
        self.assertGreater(target["Treasury_Staleness_Days"], 0)
        self.assertTrue(
            (merged["Treasury_Observation_Date"] <= merged["Date"]).all()
        )

    def test_vix_requires_an_exact_matching_date(self):
        jpm, vix, treasury = make_standard_frames()
        removed_date = vix.loc[5, "Date"]
        vix = vix.drop(index=5).reset_index(drop=True)

        merged = align_market_data(jpm, vix, treasury)

        self.assertEqual(len(merged), 24)
        self.assertNotIn(removed_date, merged["Date"].tolist())

    def test_fewer_than_twenty_complete_rows_are_rejected(self):
        jpm, vix, treasury = make_standard_frames(periods=19)

        with self.assertRaisesRegex(MarketDataError, "20"):
            align_market_data(jpm, vix, treasury)

    def test_treasury_observation_older_than_seven_days_is_rejected(self):
        jpm, vix, treasury = make_standard_frames(periods=20)
        treasury = treasury.iloc[[0]].copy()

        with self.assertRaisesRegex(MarketDataError, "stale"):
            align_market_data(jpm, vix, treasury)

    def test_metadata_has_provider_ranges_and_one_retrieval_time(self):
        jpm, vix, treasury = make_standard_frames()
        retrieved_at = pd.Timestamp("2026-07-16T12:00:00Z")

        metadata = build_metadata(jpm, vix, treasury, retrieved_at)

        self.assertEqual(
            list(metadata.columns),
            [
                "Dataset",
                "Provider",
                "RowCount",
                "FirstDate",
                "LatestDate",
                "RetrievedAtUTC",
            ],
        )
        self.assertEqual(metadata["Provider"].tolist(), ["Alpha Vantage", "Cboe", "FRED"])
        self.assertEqual(metadata["RowCount"].tolist(), [25, 25, 25])
        self.assertEqual(metadata["RetrievedAtUTC"].nunique(), 1)
        self.assertEqual(metadata.loc[0, "RetrievedAtUTC"], retrieved_at)

    def test_market_data_bundle_exposes_all_five_frames(self):
        jpm, vix, treasury = make_standard_frames()
        merged = align_market_data(jpm, vix, treasury)
        metadata = build_metadata(
            jpm,
            vix,
            treasury,
            pd.Timestamp("2026-07-16T12:00:00Z"),
        )

        bundle = MarketDataBundle(jpm, vix, treasury, merged, metadata)

        self.assertIs(bundle.jpm, jpm)
        self.assertIs(bundle.vix, vix)
        self.assertIs(bundle.treasury, treasury)
        self.assertIs(bundle.merged, merged)
        self.assertIs(bundle.metadata, metadata)


class MarketDataOrchestrationTests(unittest.TestCase):
    @patch("Week7.pricing_tool.market_data_service.fetch_treasury_daily")
    @patch("Week7.pricing_tool.market_data_service.fetch_vix_daily")
    @patch("Week7.pricing_tool.market_data_service.fetch_jpm_daily")
    @patch("Week7.pricing_tool.market_data_service.load_api_keys")
    def test_load_market_data_builds_all_bundle_frames(
        self,
        mock_load_keys,
        mock_fetch_jpm,
        mock_fetch_vix,
        mock_fetch_treasury,
    ):
        jpm, vix, treasury = make_standard_frames()
        mock_load_keys.return_value = SimpleNamespace(
            alpha_vantage="fake-alpha-key",
            fred="fake-fred-key",
        )
        mock_fetch_jpm.return_value = jpm
        mock_fetch_vix.return_value = vix
        mock_fetch_treasury.return_value = treasury
        client = FakeHttpClient(FakeResponse(payload={}))

        bundle = load_market_data(http_client=client)

        self.assertIsInstance(bundle, MarketDataBundle)
        self.assertIs(bundle.jpm, jpm)
        self.assertIs(bundle.vix, vix)
        self.assertIs(bundle.treasury, treasury)
        self.assertEqual(len(bundle.merged), 25)
        self.assertEqual(len(bundle.metadata), 3)
        mock_fetch_jpm.assert_called_once_with(
            "fake-alpha-key",
            http_client=client,
        )
        mock_fetch_vix.assert_called_once_with(http_client=client)
        treasury_call = mock_fetch_treasury.call_args
        self.assertEqual(treasury_call.args[0], "fake-fred-key")
        self.assertEqual(
            pd.Timestamp(treasury_call.kwargs["observation_start"]),
            jpm["Date"].min() - pd.Timedelta(days=10),
        )
        self.assertIs(treasury_call.kwargs["http_client"], client)


if __name__ == "__main__":
    unittest.main()
