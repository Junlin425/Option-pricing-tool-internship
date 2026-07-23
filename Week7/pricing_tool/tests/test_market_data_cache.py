import os
import pickle
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from Week7.pricing_tool.market_data_cache import (
    CACHE_SCHEMA_VERSION,
    CachedMarketDataResult,
    CachedMarketDataError,
    _read_valid_cache,
    _write_cache_atomically,
    load_cached_market_data,
)
from Week7.pricing_tool.market_data_service import (
    MarketDataBundle,
    align_market_data,
    build_metadata,
)


def make_market_bundle(periods: int = 45) -> MarketDataBundle:
    dates = pd.bdate_range("2026-04-01", periods=periods)
    step = np.arange(periods, dtype=float)
    close = 100.0 + step + 0.1 * np.sin(step)
    jpm = pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": (1_000_000 + 10_000 * step).astype("int64"),
        }
    )
    vix_close = 18.0 + 0.08 * step + np.sin(step / 3.0)
    vix = pd.DataFrame(
        {
            "Date": dates,
            "VIX_Open": vix_close - 0.2,
            "VIX_High": vix_close + 0.5,
            "VIX_Low": vix_close - 0.5,
            "VIX_Close": vix_close,
        }
    )
    treasury = pd.DataFrame(
        {
            "Treasury_Observation_Date": dates,
            "Treasury_Yield": 4.0 + 0.01 * step,
        }
    )
    merged = align_market_data(jpm, vix, treasury)
    metadata = build_metadata(
        jpm,
        vix,
        treasury,
        pd.Timestamp("2026-07-16T12:00:00Z"),
    )
    return MarketDataBundle(jpm, vix, treasury, merged, metadata)


class CachePersistenceTests(unittest.TestCase):
    def test_atomic_write_round_trips_a_valid_bundle(self):
        bundle = make_market_bundle()
        cached_at = pd.Timestamp("2026-07-16T12:00:00Z")
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "market_data_cache.pkl"

            _write_cache_atomically(bundle, cached_at, cache_path)
            loaded_bundle, loaded_at = _read_valid_cache(
                cache_path,
                now_utc=pd.Timestamp("2026-07-16T12:30:00Z"),
            )

            self.assertEqual(loaded_at, cached_at)
            pd.testing.assert_frame_equal(loaded_bundle.jpm, bundle.jpm)
            pd.testing.assert_frame_equal(loaded_bundle.vix, bundle.vix)
            pd.testing.assert_frame_equal(loaded_bundle.treasury, bundle.treasury)
            pd.testing.assert_frame_equal(loaded_bundle.merged, bundle.merged)
            pd.testing.assert_frame_equal(loaded_bundle.metadata, bundle.metadata)

    def test_corrupt_partial_and_wrong_version_caches_are_rejected(self):
        bundle = make_market_bundle()
        cached_at = pd.Timestamp("2026-07-16T12:00:00Z")
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            payloads = {
                "corrupt.pkl": b"not-a-pickle",
                "partial.pkl": pickle.dumps(
                    {"schema_version": CACHE_SCHEMA_VERSION, "cached_at_utc": "bad"}
                ),
                "wrong_version.pkl": pickle.dumps(
                    {
                        "schema_version": CACHE_SCHEMA_VERSION + 1,
                        "cached_at_utc": cached_at,
                        "latest_market_date": bundle.merged["Date"].max(),
                        "bundle": bundle,
                    }
                ),
            }
            for name, content in payloads.items():
                with self.subTest(name=name):
                    path = directory_path / name
                    path.write_bytes(content)
                    with self.assertRaises(CachedMarketDataError):
                        _read_valid_cache(path, now_utc=cached_at)

    def test_structurally_invalid_bundle_is_rejected(self):
        bundle = make_market_bundle()
        cached_at = pd.Timestamp("2026-07-16T12:00:00Z")
        invalid_bundles = [
            replace(bundle, merged=bundle.merged.iloc[:39].copy()),
            replace(
                bundle,
                merged=bundle.merged.assign(
                    Date=lambda frame: frame["Date"].where(frame.index != 10, frame.loc[9, "Date"])
                ),
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, invalid_bundle in enumerate(invalid_bundles):
                with self.subTest(index=index):
                    path = Path(directory) / f"invalid_{index}.pkl"
                    payload = {
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "cached_at_utc": cached_at,
                        "latest_market_date": invalid_bundle.merged["Date"].max(),
                        "bundle": invalid_bundle,
                    }
                    path.write_bytes(pickle.dumps(payload))
                    with self.assertRaises(CachedMarketDataError):
                        _read_valid_cache(path, now_utc=cached_at)

    def test_failed_replace_preserves_previous_cache_and_removes_temporary_file(self):
        bundle = make_market_bundle()
        cached_at = pd.Timestamp("2026-07-16T12:00:00Z")
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "market_data_cache.pkl"
            _write_cache_atomically(bundle, cached_at, cache_path)
            original_bytes = cache_path.read_bytes()

            with patch(
                "Week7.pricing_tool.market_data_cache.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaises(CachedMarketDataError):
                    _write_cache_atomically(
                        bundle,
                        cached_at + pd.Timedelta(hours=1),
                        cache_path,
                    )

            self.assertEqual(cache_path.read_bytes(), original_bytes)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            self.assertEqual(
                [path for path in Path(directory).iterdir() if path.name.startswith(".")],
                [],
            )


class CachePolicyTests(unittest.TestCase):
    NOW = pd.Timestamp("2026-07-16T12:00:00Z")

    def test_first_load_calls_provider_writes_cache_and_returns_live(self):
        calls = []
        bundle = make_market_bundle()
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.pkl"

            result = load_cached_market_data(
                market_data_loader=lambda: calls.append("called") or bundle,
                cache_path=cache_path,
                now_utc=self.NOW,
            )

            self.assertIsInstance(result, CachedMarketDataResult)
            self.assertEqual(calls, ["called"])
            self.assertEqual(result.status.source, "live")
            self.assertFalse(result.status.is_stale)
            self.assertTrue(result.status.refresh_attempted)
            self.assertEqual(result.status.age_seconds, 0.0)
            self.assertIsNone(result.status.warning)
            self.assertTrue(cache_path.is_file())

    def test_fresh_cache_avoids_provider_call(self):
        calls = []
        bundle = make_market_bundle()
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.pkl"
            cached_at = self.NOW - pd.Timedelta(minutes=30)
            _write_cache_atomically(bundle, cached_at, cache_path)

            result = load_cached_market_data(
                market_data_loader=lambda: calls.append("called") or bundle,
                cache_path=cache_path,
                now_utc=self.NOW,
            )

            self.assertEqual(calls, [])
            self.assertEqual(result.status.source, "fresh_cache")
            self.assertFalse(result.status.is_stale)
            self.assertFalse(result.status.refresh_attempted)
            self.assertEqual(result.status.age_seconds, 30 * 60)
            self.assertEqual(result.status.cached_at_utc, cached_at)

    def test_expired_cache_refreshes_and_force_refresh_bypasses_fresh_cache(self):
        bundle = make_market_bundle()
        cases = [
            (self.NOW - pd.Timedelta(minutes=61), False),
            (self.NOW - pd.Timedelta(minutes=30), True),
        ]
        for cached_at, force_refresh in cases:
            with self.subTest(cached_at=cached_at, force_refresh=force_refresh):
                calls = []
                with tempfile.TemporaryDirectory() as directory:
                    cache_path = Path(directory) / "cache.pkl"
                    _write_cache_atomically(bundle, cached_at, cache_path)

                    result = load_cached_market_data(
                        force_refresh=force_refresh,
                        market_data_loader=lambda: calls.append("called") or bundle,
                        cache_path=cache_path,
                        now_utc=self.NOW,
                    )

                    self.assertEqual(calls, ["called"])
                    self.assertEqual(result.status.source, "live")
                    _, updated_at = _read_valid_cache(cache_path, now_utc=self.NOW)
                    self.assertEqual(updated_at, self.NOW)

    def test_refresh_failure_uses_cache_within_seven_days(self):
        bundle = make_market_bundle()
        cases = [
            (self.NOW - pd.Timedelta(hours=2), False),
            (self.NOW - pd.Timedelta(minutes=30), True),
        ]
        for cached_at, force_refresh in cases:
            with self.subTest(cached_at=cached_at, force_refresh=force_refresh):
                with tempfile.TemporaryDirectory() as directory:
                    cache_path = Path(directory) / "cache.pkl"
                    _write_cache_atomically(bundle, cached_at, cache_path)

                    def failing_loader():
                        raise RuntimeError("fake-secret-value")

                    result = load_cached_market_data(
                        force_refresh=force_refresh,
                        market_data_loader=failing_loader,
                        cache_path=cache_path,
                        now_utc=self.NOW,
                    )

                    self.assertEqual(result.status.source, "stale_fallback")
                    self.assertTrue(result.status.is_stale)
                    self.assertTrue(result.status.refresh_attempted)
                    self.assertIsNotNone(result.status.warning)
                    self.assertNotIn("fake-secret-value", result.status.warning)

    def test_refresh_failure_rejects_missing_or_overly_old_cache(self):
        bundle = make_market_bundle()

        def failing_loader():
            raise RuntimeError("fake-secret-value")

        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "missing.pkl", Path(directory) / "old.pkl"]
            _write_cache_atomically(
                bundle,
                self.NOW - pd.Timedelta(days=8),
                paths[1],
            )
            for path in paths:
                with self.subTest(path=path.name):
                    with self.assertRaises(CachedMarketDataError) as caught:
                        load_cached_market_data(
                            market_data_loader=failing_loader,
                            cache_path=path,
                            now_utc=self.NOW,
                        )
                    self.assertNotIn("fake-secret-value", str(caught.exception))

    def test_invalid_ttl_configuration_is_rejected(self):
        invalid_cases = [
            (timedelta(0), timedelta(days=7)),
            (timedelta(minutes=60), timedelta(0)),
            (timedelta(days=2), timedelta(days=1)),
        ]
        for fresh_ttl, max_stale_age in invalid_cases:
            with self.subTest(fresh_ttl=fresh_ttl, max_stale_age=max_stale_age):
                with self.assertRaisesRegex(CachedMarketDataError, "TTL"):
                    load_cached_market_data(
                        fresh_ttl=fresh_ttl,
                        max_stale_age=max_stale_age,
                        market_data_loader=make_market_bundle,
                        now_utc=self.NOW,
                    )

    def test_cache_write_failure_returns_live_data_with_warning(self):
        bundle = make_market_bundle()
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "Week7.pricing_tool.market_data_cache._write_cache_atomically",
                side_effect=CachedMarketDataError("fake-secret-value"),
            ):
                result = load_cached_market_data(
                    market_data_loader=lambda: bundle,
                    cache_path=Path(directory) / "cache.pkl",
                    now_utc=self.NOW,
                )

            self.assertEqual(result.status.source, "live")
            self.assertIsNotNone(result.status.warning)
            self.assertNotIn("fake-secret-value", result.status.warning)


if __name__ == "__main__":
    unittest.main()
