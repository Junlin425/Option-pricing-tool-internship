"""Daily market-data adapters for the Week 7 pricing tool."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
from typing import Any

import pandas as pd
import requests

try:
    from .config import load_api_keys
except ImportError:  # Supports direct execution from the pricing_tool folder.
    from config import load_api_keys


ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_TIMEOUT_SECONDS = 30


class MarketDataError(RuntimeError):
    """Raised when a primary market-data provider cannot supply valid data."""


@dataclass(frozen=True)
class MarketDataBundle:
    """Standardized source tables, aligned data, and retrieval metadata."""

    jpm: pd.DataFrame
    vix: pd.DataFrame
    treasury: pd.DataFrame
    merged: pd.DataFrame
    metadata: pd.DataFrame


def _get_response(provider: str, http_client: Any, url: str, **kwargs: Any) -> Any:
    try:
        response = http_client.get(
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
        response.raise_for_status()
        return response
    except Exception as error:
        raise MarketDataError(f"{provider} request failed.") from error


def _finish_daily_frame(
    frame: pd.DataFrame,
    provider: str,
    date_column: str,
    numeric_columns: list[str],
) -> pd.DataFrame:
    frame = frame.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[[date_column, *numeric_columns]].isna().any().any():
        raise MarketDataError(f"{provider} returned invalid date or numeric values.")
    if frame[date_column].duplicated().any():
        raise MarketDataError(f"{provider} returned duplicate dates.")

    return frame.sort_values(date_column).reset_index(drop=True)


def fetch_jpm_daily(api_key: str, http_client: Any = requests) -> pd.DataFrame:
    """Return standardized JPM daily OHLCV data from Alpha Vantage."""

    response = _get_response(
        "Alpha Vantage",
        http_client,
        ALPHA_VANTAGE_URL,
        params={
            "function": "TIME_SERIES_DAILY",
            "symbol": "JPM",
            "outputsize": "compact",
            "apikey": api_key,
        },
    )
    try:
        payload = response.json()
    except Exception as error:
        raise MarketDataError("Alpha Vantage returned invalid JSON.") from error

    series = payload.get("Time Series (Daily)") if isinstance(payload, dict) else None
    if not isinstance(series, dict) or not series:
        raise MarketDataError("Alpha Vantage returned an unexpected response schema.")

    provider_columns = ["1. open", "2. high", "3. low", "4. close", "5. volume"]
    frame = pd.DataFrame.from_dict(series, orient="index")
    if not set(provider_columns).issubset(frame.columns):
        raise MarketDataError("Alpha Vantage response is missing required fields.")

    frame = (
        frame.reset_index(names="Date")
        .rename(
            columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume",
            }
        )
        [["Date", "Open", "High", "Low", "Close", "Volume"]]
    )
    frame = _finish_daily_frame(
        frame,
        "Alpha Vantage",
        "Date",
        ["Open", "High", "Low", "Close", "Volume"],
    )
    if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any() or (frame["Volume"] < 0).any():
        raise MarketDataError("Alpha Vantage returned invalid market values.")
    frame["Volume"] = frame["Volume"].astype("int64")
    return frame


def fetch_vix_daily(http_client: Any = requests) -> pd.DataFrame:
    """Return standardized VIX daily data from Cboe."""

    response = _get_response("Cboe", http_client, CBOE_VIX_URL)
    try:
        frame = pd.read_csv(StringIO(response.text))
    except Exception as error:
        raise MarketDataError("Cboe returned invalid CSV data.") from error

    required = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]
    if not set(required).issubset(frame.columns):
        raise MarketDataError("Cboe response is missing required fields.")

    frame = frame[required].rename(
        columns={
            "DATE": "Date",
            "OPEN": "VIX_Open",
            "HIGH": "VIX_High",
            "LOW": "VIX_Low",
            "CLOSE": "VIX_Close",
        }
    )
    frame = _finish_daily_frame(
        frame,
        "Cboe",
        "Date",
        ["VIX_Open", "VIX_High", "VIX_Low", "VIX_Close"],
    )
    if (frame[["VIX_Open", "VIX_High", "VIX_Low", "VIX_Close"]] <= 0).any().any():
        raise MarketDataError("Cboe returned invalid VIX values.")
    return frame


def fetch_treasury_daily(
    api_key: str,
    observation_start: date | pd.Timestamp,
    http_client: Any = requests,
) -> pd.DataFrame:
    """Return standardized daily DGS10 observations from FRED."""

    response = _get_response(
        "FRED",
        http_client,
        FRED_URL,
        params={
            "series_id": "DGS10",
            "api_key": api_key,
            "file_type": "json",
            "observation_start": pd.Timestamp(observation_start).date().isoformat(),
            "sort_order": "asc",
        },
    )
    try:
        payload = response.json()
    except Exception as error:
        raise MarketDataError("FRED returned invalid JSON.") from error

    observations = payload.get("observations") if isinstance(payload, dict) else None
    if not isinstance(observations, list) or not observations:
        raise MarketDataError("FRED returned an unexpected response schema.")

    frame = pd.DataFrame(observations)
    if not {"date", "value"}.issubset(frame.columns):
        raise MarketDataError("FRED response is missing required fields.")

    frame = frame[["date", "value"]].rename(
        columns={
            "date": "Treasury_Observation_Date",
            "value": "Treasury_Yield",
        }
    )
    frame["Treasury_Observation_Date"] = pd.to_datetime(
        frame["Treasury_Observation_Date"], errors="coerce"
    ).dt.normalize()
    frame["Treasury_Yield"] = pd.to_numeric(frame["Treasury_Yield"], errors="coerce")
    frame = frame.dropna(subset=["Treasury_Observation_Date", "Treasury_Yield"])
    if frame.empty:
        raise MarketDataError("FRED returned no valid DGS10 observations.")
    if frame["Treasury_Observation_Date"].duplicated().any():
        raise MarketDataError("FRED returned duplicate dates.")

    return frame.sort_values("Treasury_Observation_Date").reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, required: list[str], dataset: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise MarketDataError(f"{dataset} data is missing required standardized fields.")


def align_market_data(
    jpm: pd.DataFrame,
    vix: pd.DataFrame,
    treasury: pd.DataFrame,
    min_rows: int = 20,
    max_treasury_staleness_days: int = 7,
) -> pd.DataFrame:
    """Align VIX exactly and Treasury backward onto the JPM trading calendar."""

    jpm_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    vix_columns = ["Date", "VIX_Open", "VIX_High", "VIX_Low", "VIX_Close"]
    treasury_columns = ["Treasury_Observation_Date", "Treasury_Yield"]
    _require_columns(jpm, jpm_columns, "JPM")
    _require_columns(vix, vix_columns, "VIX")
    _require_columns(treasury, treasury_columns, "Treasury")

    jpm_sorted = jpm[jpm_columns].sort_values("Date").copy()
    vix_sorted = vix[vix_columns].sort_values("Date").copy()
    treasury_sorted = treasury[treasury_columns].sort_values(
        "Treasury_Observation_Date"
    ).copy()

    try:
        market = jpm_sorted.merge(
            vix_sorted,
            on="Date",
            how="inner",
            validate="one_to_one",
        )
        merged = pd.merge_asof(
            market.sort_values("Date"),
            treasury_sorted,
            left_on="Date",
            right_on="Treasury_Observation_Date",
            direction="backward",
        )
    except Exception as error:
        raise MarketDataError("Market data could not be aligned by date.") from error

    merged = merged.dropna(
        subset=["Treasury_Observation_Date", "Treasury_Yield"]
    ).reset_index(drop=True)
    merged["Treasury_Staleness_Days"] = (
        merged["Date"] - merged["Treasury_Observation_Date"]
    ).dt.days.astype("int64")

    if len(merged) < min_rows:
        raise MarketDataError(
            f"Aligned market data requires at least {min_rows} complete rows."
        )
    if (merged["Treasury_Staleness_Days"] > max_treasury_staleness_days).any():
        raise MarketDataError("FRED Treasury data is stale for at least one market date.")

    output_columns = [
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
    ]
    return merged[output_columns]


def build_metadata(
    jpm: pd.DataFrame,
    vix: pd.DataFrame,
    treasury: pd.DataFrame,
    retrieved_at_utc: pd.Timestamp,
) -> pd.DataFrame:
    """Describe provider coverage without including credentials."""

    timestamp = pd.Timestamp(retrieved_at_utc)
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
    definitions = [
        ("JPM", "Alpha Vantage", jpm, "Date"),
        ("VIX", "Cboe", vix, "Date"),
        ("Treasury", "FRED", treasury, "Treasury_Observation_Date"),
    ]
    records = []
    for dataset, provider, frame, date_column in definitions:
        if frame.empty:
            raise MarketDataError(f"{dataset} metadata cannot be built from empty data.")
        records.append(
            {
                "Dataset": dataset,
                "Provider": provider,
                "RowCount": len(frame),
                "FirstDate": frame[date_column].min(),
                "LatestDate": frame[date_column].max(),
                "RetrievedAtUTC": timestamp,
            }
        )

    return pd.DataFrame(
        records,
        columns=[
            "Dataset",
            "Provider",
            "RowCount",
            "FirstDate",
            "LatestDate",
            "RetrievedAtUTC",
        ],
    )


def load_market_data(http_client: Any = requests) -> MarketDataBundle:
    """Retrieve, standardize, align, and describe the three primary sources."""

    keys = load_api_keys()
    jpm = fetch_jpm_daily(keys.alpha_vantage, http_client=http_client)
    vix = fetch_vix_daily(http_client=http_client)
    observation_start = jpm["Date"].min() - pd.Timedelta(days=10)
    treasury = fetch_treasury_daily(
        keys.fred,
        observation_start=observation_start,
        http_client=http_client,
    )
    merged = align_market_data(jpm, vix, treasury)
    metadata = build_metadata(
        jpm,
        vix,
        treasury,
        pd.Timestamp.now(tz="UTC"),
    )

    return MarketDataBundle(
        jpm=jpm,
        vix=vix,
        treasury=treasury,
        merged=merged,
        metadata=metadata,
    )
