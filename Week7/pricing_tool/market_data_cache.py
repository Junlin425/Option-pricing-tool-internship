"""Persistent market-data cache with explicit freshness and fallback status."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os
from pathlib import Path
import pickle
from typing import Callable, Literal
from uuid import uuid4

import numpy as np
import pandas as pd

try:
    from .market_data_service import MarketDataBundle, load_market_data
except ImportError:  # Supports direct execution from the pricing_tool folder.
    from market_data_service import MarketDataBundle, load_market_data


CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "market_data_cache.pkl"
DEFAULT_FRESH_TTL = timedelta(minutes=60)
DEFAULT_MAX_STALE_AGE = timedelta(days=7)
FUTURE_TIMESTAMP_TOLERANCE = timedelta(minutes=5)


class CachedMarketDataError(RuntimeError):
    """Raised when neither live nor acceptable cached market data is available."""


@dataclass(frozen=True)
class CacheStatus:
    """Describe where market data came from and whether it is stale."""

    source: Literal["live", "fresh_cache", "stale_fallback"]
    cached_at_utc: pd.Timestamp
    age_seconds: float
    is_stale: bool
    refresh_attempted: bool
    warning: str | None


@dataclass(frozen=True)
class CachedMarketDataResult:
    """A standardized market bundle plus transparent cache status."""

    bundle: MarketDataBundle
    status: CacheStatus


_FRAME_SCHEMAS = {
    "jpm": ["Date", "Open", "High", "Low", "Close", "Volume"],
    "vix": ["Date", "VIX_Open", "VIX_High", "VIX_Low", "VIX_Close"],
    "treasury": ["Treasury_Observation_Date", "Treasury_Yield"],
    "merged": [
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
    "metadata": [
        "Dataset",
        "Provider",
        "RowCount",
        "FirstDate",
        "LatestDate",
        "RetrievedAtUTC",
    ],
}


def _as_utc(value: object, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as error:
        raise CachedMarketDataError(f"Cache {field_name} is invalid.") from error
    if pd.isna(timestamp):
        raise CachedMarketDataError(f"Cache {field_name} is invalid.")
    if timestamp.tzinfo is None:
        raise CachedMarketDataError(f"Cache {field_name} must include a timezone.")
    return timestamp.tz_convert("UTC")


def _validate_dates(frame: pd.DataFrame, column: str, dataset: str) -> None:
    dates = pd.to_datetime(frame[column], errors="coerce")
    if dates.isna().any():
        raise CachedMarketDataError(f"Cached {dataset} dates are invalid.")
    if dates.duplicated().any():
        raise CachedMarketDataError(f"Cached {dataset} dates are duplicated.")
    if not dates.is_monotonic_increasing:
        raise CachedMarketDataError(f"Cached {dataset} dates are not increasing.")


def _validate_bundle(bundle: object) -> MarketDataBundle:
    if not isinstance(bundle, MarketDataBundle):
        raise CachedMarketDataError("Cached market bundle has an invalid type.")

    for field_name, required_columns in _FRAME_SCHEMAS.items():
        frame = getattr(bundle, field_name)
        if not isinstance(frame, pd.DataFrame):
            raise CachedMarketDataError(f"Cached {field_name} is not a DataFrame.")
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise CachedMarketDataError(f"Cached {field_name} is missing required columns.")

    if len(bundle.merged) < 40:
        raise CachedMarketDataError("Cached merged market data requires at least 40 rows.")
    _validate_dates(bundle.jpm, "Date", "JPM")
    _validate_dates(bundle.vix, "Date", "VIX")
    _validate_dates(bundle.treasury, "Treasury_Observation_Date", "Treasury")
    _validate_dates(bundle.merged, "Date", "merged market")

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "VIX_Open",
        "VIX_High",
        "VIX_Low",
        "VIX_Close",
        "Treasury_Yield",
        "Treasury_Staleness_Days",
    ]
    numeric = bundle.merged[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise CachedMarketDataError("Cached merged market numeric values are invalid.")
    return bundle


def _read_valid_cache(
    cache_path: str | Path,
    now_utc: pd.Timestamp,
) -> tuple[MarketDataBundle, pd.Timestamp]:
    """Read and structurally validate a trusted local cache file."""

    path = Path(cache_path)
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except Exception as error:
        raise CachedMarketDataError("Local market-data cache could not be read.") from error

    if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CachedMarketDataError("Local market-data cache version is invalid.")
    required = {"cached_at_utc", "latest_market_date", "bundle"}
    if not required.issubset(payload):
        raise CachedMarketDataError("Local market-data cache payload is incomplete.")

    now = _as_utc(now_utc, "current time")
    cached_at = _as_utc(payload["cached_at_utc"], "timestamp")
    if cached_at - now > FUTURE_TIMESTAMP_TOLERANCE:
        raise CachedMarketDataError("Local market-data cache timestamp is in the future.")

    bundle = _validate_bundle(payload["bundle"])
    try:
        recorded_latest = pd.Timestamp(payload["latest_market_date"]).normalize()
        actual_latest = pd.Timestamp(bundle.merged["Date"].max()).normalize()
    except Exception as error:
        raise CachedMarketDataError("Local market-data cache latest date is invalid.") from error
    if recorded_latest != actual_latest:
        raise CachedMarketDataError("Local market-data cache latest date is inconsistent.")
    return bundle, cached_at


def _write_cache_atomically(
    bundle: MarketDataBundle,
    cached_at_utc: pd.Timestamp,
    cache_path: str | Path,
) -> None:
    """Validate and atomically replace the single local cache snapshot."""

    valid_bundle = _validate_bundle(bundle)
    cached_at = _as_utc(cached_at_utc, "timestamp")
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cached_at_utc": cached_at,
        "latest_market_date": pd.Timestamp(valid_bundle.merged["Date"].max()).normalize(),
        "bundle": valid_bundle,
    }
    try:
        with temporary_path.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        _read_valid_cache(temporary_path, now_utc=cached_at)
        os.replace(temporary_path, path)
    except Exception as error:
        raise CachedMarketDataError("Local market-data cache could not be updated.") from error
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _validate_cache_policy(
    fresh_ttl: timedelta,
    max_stale_age: timedelta,
) -> tuple[float, float]:
    try:
        fresh_seconds = float(fresh_ttl.total_seconds())
        stale_seconds = float(max_stale_age.total_seconds())
    except (AttributeError, TypeError, ValueError) as error:
        raise CachedMarketDataError("Cache TTL values must be timedeltas.") from error
    if not np.isfinite([fresh_seconds, stale_seconds]).all():
        raise CachedMarketDataError("Cache TTL values must be finite.")
    if fresh_seconds <= 0 or stale_seconds <= 0 or stale_seconds < fresh_seconds:
        raise CachedMarketDataError(
            "Cache TTL values must be positive and max_stale_age must cover fresh_ttl."
        )
    return fresh_seconds, stale_seconds


def _status(
    source: Literal["live", "fresh_cache", "stale_fallback"],
    cached_at: pd.Timestamp,
    age_seconds: float,
    warning: str | None = None,
) -> CacheStatus:
    return CacheStatus(
        source=source,
        cached_at_utc=cached_at,
        age_seconds=float(max(age_seconds, 0.0)),
        is_stale=source == "stale_fallback",
        refresh_attempted=source != "fresh_cache",
        warning=warning,
    )


def load_cached_market_data(
    force_refresh: bool = False,
    fresh_ttl: timedelta = DEFAULT_FRESH_TTL,
    max_stale_age: timedelta = DEFAULT_MAX_STALE_AGE,
    market_data_loader: Callable[[], MarketDataBundle] = load_market_data,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    now_utc: pd.Timestamp | None = None,
) -> CachedMarketDataResult:
    """Return live, fresh cached, or explicitly stale fallback market data."""

    fresh_seconds, stale_seconds = _validate_cache_policy(fresh_ttl, max_stale_age)
    now = _as_utc(
        pd.Timestamp.now(tz="UTC") if now_utc is None else now_utc,
        "current time",
    )
    path = Path(cache_path)
    cached_bundle = None
    cached_at = None
    cache_age = None
    try:
        cached_bundle, cached_at = _read_valid_cache(path, now_utc=now)
        cache_age = max(float((now - cached_at).total_seconds()), 0.0)
    except CachedMarketDataError:
        pass

    if (
        cached_bundle is not None
        and cache_age is not None
        and cache_age <= fresh_seconds
        and not force_refresh
    ):
        return CachedMarketDataResult(
            bundle=cached_bundle,
            status=_status("fresh_cache", cached_at, cache_age),
        )

    try:
        live_bundle = _validate_bundle(market_data_loader())
    except Exception as provider_error:
        if (
            cached_bundle is not None
            and cached_at is not None
            and cache_age is not None
            and cache_age <= stale_seconds
        ):
            age_hours = cache_age / 3600.0
            warning = (
                "Real-time refresh failed. Using the last valid cache from "
                f"{cached_at.strftime('%Y-%m-%d %H:%M UTC')} "
                f"({age_hours:.1f} hours old)."
            )
            return CachedMarketDataResult(
                bundle=cached_bundle,
                status=_status(
                    "stale_fallback",
                    cached_at,
                    cache_age,
                    warning=warning,
                ),
            )
        raise CachedMarketDataError(
            "Real-time market data is unavailable and no acceptable cache exists."
        ) from provider_error

    warning = None
    try:
        _write_cache_atomically(live_bundle, now, path)
    except CachedMarketDataError:
        warning = "Live data loaded, but the local cache could not be updated."
    return CachedMarketDataResult(
        bundle=live_bundle,
        status=_status("live", now, 0.0, warning=warning),
    )
