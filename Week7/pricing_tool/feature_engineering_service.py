"""Training-consistent real-time features for the Week 7 pricing tool."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from .market_data_service import MarketDataBundle, load_market_data
except ImportError:  # Supports direct execution from the pricing_tool folder.
    from market_data_service import MarketDataBundle, load_market_data


LINEAR_FEATURES = [
    "Close",
    "Treasury_Yield",
    "RollingVol20",
    "VolumeChange",
    "VIX_Close",
    "VIX_Return",
    "VIX_5D_Change",
    "VIX_20D_ZScore",
    "VIX_Regime",
    "VIX_Spike",
]

LSTM_FEATURES = [
    "RollingVol20",
    "Return",
    "MA5_Ratio",
    "MA20_Ratio",
    "VolumeChange",
    "RateMomentum",
    "Treasury_Yield",
    "VIX_Close",
    "VIX_Return",
    "VIX_5D_Change",
    "VIX_20D_ZScore",
    "VIX_Regime",
    "VIX_Spike",
]

MODEL_FEATURES = list(dict.fromkeys(LINEAR_FEATURES + LSTM_FEATURES))
REQUIRED_COLUMNS = ["Date", "Close", "Volume", "VIX_Close", "Treasury_Yield"]
QUALITY_COLUMNS = [
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
]
DEFAULT_THRESHOLD_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "vix_feature_thresholds.json"
)


class FeatureEngineeringError(RuntimeError):
    """Raised when live market data cannot produce valid model features."""


@dataclass(frozen=True)
class RealTimeFeatureBundle:
    """Feature history plus model-specific raw inputs and quality metadata."""

    history: pd.DataFrame
    linear_latest: pd.DataFrame
    lstm_sequence: pd.DataFrame
    latest_market_date: pd.Timestamp
    quality: pd.DataFrame


def load_vix_thresholds(path: str | Path | None = None) -> dict[str, object]:
    """Load the VIX classification thresholds fitted on historical training data."""

    threshold_path = Path(path) if path is not None else DEFAULT_THRESHOLD_PATH
    try:
        with threshold_path.open("r", encoding="utf-8") as stream:
            thresholds = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureEngineeringError("VIX threshold configuration could not be loaded.") from error

    required = ("regime_low_q33", "regime_high_q67", "spike_return_q95")
    if not isinstance(thresholds, dict) or any(key not in thresholds for key in required):
        raise FeatureEngineeringError("VIX threshold configuration is missing required values.")
    try:
        values = {key: float(thresholds[key]) for key in required}
    except (TypeError, ValueError) as error:
        raise FeatureEngineeringError("VIX threshold values must be numeric.") from error
    if not np.isfinite(list(values.values())).all():
        raise FeatureEngineeringError("VIX threshold values must be finite.")
    if values["regime_low_q33"] >= values["regime_high_q67"]:
        raise FeatureEngineeringError("VIX threshold ordering must satisfy low < high.")
    if thresholds.get("threshold_source") != "train_only":
        raise FeatureEngineeringError("VIX threshold source must be train_only.")
    return {**thresholds, **values}


def _validate_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FeatureEngineeringError(f"Market data is missing required columns: {missing}.")

    result = frame.copy().reset_index(drop=True)
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce").dt.normalize()
    if result["Date"].isna().any():
        raise FeatureEngineeringError("Market data contains invalid dates.")
    if result["Date"].duplicated().any():
        raise FeatureEngineeringError("Market data contains duplicate dates.")
    if not result["Date"].is_monotonic_increasing:
        raise FeatureEngineeringError("Market data dates must be strictly increasing.")

    numeric_columns = ["Close", "Volume", "VIX_Close", "Treasury_Yield"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    numeric_values = result[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise FeatureEngineeringError("Market numeric values must be finite and complete.")
    if (result["Close"] <= 0).any() or (result["VIX_Close"] <= 0).any():
        raise FeatureEngineeringError("Market numeric Close and VIX_Close values must be positive.")
    if (result["Volume"] < 0).any():
        raise FeatureEngineeringError("Market numeric Volume values must be non-negative.")
    return result


def build_realtime_features(
    market_bundle: MarketDataBundle,
    thresholds_path: str | Path | None = None,
    lstm_window: int = 20,
) -> RealTimeFeatureBundle:
    """Build raw Linear and LSTM features from a standardized market-data bundle."""

    if not isinstance(lstm_window, int) or lstm_window < 1:
        raise FeatureEngineeringError("lstm_window must be a positive integer.")
    frame = _validate_market_frame(market_bundle.merged)
    raw_row_count = len(frame)
    thresholds = load_vix_thresholds(thresholds_path)

    frame["Return"] = np.log(frame["Close"] / frame["Close"].shift(1))
    frame["MA5"] = frame["Close"].rolling(5).mean()
    frame["MA20"] = frame["Close"].rolling(20).mean()
    frame["MA5_Ratio"] = frame["Close"] / frame["MA5"] - 1.0
    frame["MA20_Ratio"] = frame["Close"] / frame["MA20"] - 1.0
    frame["RollingVol20"] = frame["Return"].rolling(20).std(ddof=1)
    frame["VolumeChange"] = frame["Volume"].pct_change()
    frame["RateMomentum"] = frame["Treasury_Yield"] - frame["Treasury_Yield"].shift(5)
    frame["VIX_Return"] = np.log(frame["VIX_Close"] / frame["VIX_Close"].shift(1))
    frame["VIX_5D_Change"] = frame["VIX_Close"].pct_change(periods=5)
    vix_rolling = frame["VIX_Close"].rolling(window=20, min_periods=20)
    vix_mean = vix_rolling.mean()
    vix_std = vix_rolling.std(ddof=0)
    if (vix_std.iloc[19:] <= 0).any():
        raise FeatureEngineeringError("VIX rolling standard deviation must be positive.")
    frame["VIX_20D_ZScore"] = (frame["VIX_Close"] - vix_mean) / vix_std

    low = float(thresholds["regime_low_q33"])
    high = float(thresholds["regime_high_q67"])
    spike = float(thresholds["spike_return_q95"])
    frame["VIX_Regime"] = np.select(
        [frame["VIX_Close"] < low, frame["VIX_Close"] > high],
        [0, 2],
        default=1,
    ).astype(int)
    frame["VIX_Spike"] = (frame["VIX_Return"] >= spike).astype(int)

    post_warmup = frame.iloc[20:]
    post_warmup_values = post_warmup[MODEL_FEATURES].to_numpy(dtype=float)
    if np.isnan(post_warmup_values).any() or np.isinf(post_warmup_values).any():
        raise FeatureEngineeringError("Derived model features contain invalid values after warm-up.")

    history = frame.dropna(subset=MODEL_FEATURES).reset_index(drop=True)
    if len(history) < lstm_window:
        raise FeatureEngineeringError(
            f"At least {lstm_window} valid feature rows are required for the LSTM window."
        )
    linear_latest = history.loc[[history.index[-1]], LINEAR_FEATURES].reset_index(drop=True)
    lstm_sequence = history.loc[:, LSTM_FEATURES].tail(lstm_window).reset_index(drop=True)
    latest_market_date = pd.Timestamp(history.iloc[-1]["Date"])
    model_values = history[MODEL_FEATURES].to_numpy(dtype=float)
    quality = pd.DataFrame(
        [
            {
                "RawRowCount": raw_row_count,
                "ValidFeatureRowCount": len(history),
                "WarmupRowCount": raw_row_count - len(history),
                "LatestMarketDate": latest_market_date,
                "LinearFeatureCount": len(LINEAR_FEATURES),
                "LSTMWindowRows": len(lstm_sequence),
                "LSTMFeatureCount": len(LSTM_FEATURES),
                "MissingModelValues": int(np.isnan(model_values).sum()),
                "InfiniteModelValues": int(np.isinf(model_values).sum()),
                "ThresholdSource": thresholds["threshold_source"],
            }
        ],
        columns=QUALITY_COLUMNS,
    )

    return RealTimeFeatureBundle(
        history=history,
        linear_latest=linear_latest,
        lstm_sequence=lstm_sequence,
        latest_market_date=latest_market_date,
        quality=quality,
    )


def load_realtime_features(
    market_data_loader: Callable[[], MarketDataBundle] = load_market_data,
    thresholds_path: str | Path | None = None,
    lstm_window: int = 20,
) -> RealTimeFeatureBundle:
    """Load current primary-source data and build training-consistent features."""

    return build_realtime_features(
        market_data_loader(),
        thresholds_path=thresholds_path,
        lstm_window=lstm_window,
    )
