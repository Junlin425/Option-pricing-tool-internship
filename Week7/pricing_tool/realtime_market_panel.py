"""Streamlit panel for cached real-time market data and Task 3.3 features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from Week7.pricing_tool.feature_engineering_service import (
    FeatureEngineeringError,
    RealTimeFeatureBundle,
    build_realtime_features,
)
from Week7.pricing_tool.market_data_cache import (
    CachedMarketDataError,
    CachedMarketDataResult,
    load_cached_market_data,
)
from Week7.pricing_tool.model_inference_service import (
    LoadedModelArtifacts,
    ModelInferenceError,
    ThreeModelPricingResult,
    load_model_artifacts,
    predict_three_model_call_prices,
)
from Week7.pricing_tool.dashboard_panel import render_live_error_reference


@dataclass(frozen=True)
class RealtimeMarketSnapshot:
    """Cache status, engineered features, and latest display values."""

    cache: CachedMarketDataResult
    features: RealTimeFeatureBundle
    latest_close: float
    latest_vix: float
    latest_treasury_yield: float


def load_realtime_market_snapshot(
    force_refresh: bool = False,
    cache_loader: Callable[..., CachedMarketDataResult] = load_cached_market_data,
    feature_builder: Callable[..., RealTimeFeatureBundle] = build_realtime_features,
) -> RealtimeMarketSnapshot:
    """Compose cached Task 3.2 data with Task 3.3 feature engineering."""

    cached = cache_loader(force_refresh=force_refresh)
    features = feature_builder(cached.bundle)
    latest = cached.bundle.merged.iloc[-1]
    return RealtimeMarketSnapshot(
        cache=cached,
        features=features,
        latest_close=float(latest["Close"]),
        latest_vix=float(latest["VIX_Close"]),
        latest_treasury_yield=float(latest["Treasury_Yield"]),
    )


def _format_age(age_seconds: float) -> str:
    if age_seconds < 60:
        return f"{age_seconds:.0f} seconds"
    if age_seconds < 3600:
        return f"{age_seconds / 60:.1f} minutes"
    return f"{age_seconds / 3600:.1f} hours"


def _render_snapshot(snapshot: RealtimeMarketSnapshot) -> None:
    status = snapshot.cache.status
    source_labels = {
        "live": "Live provider update",
        "fresh_cache": "Fresh local cache",
        "stale_fallback": "Stale fallback cache",
    }
    if status.source == "stale_fallback":
        st.warning(status.warning or "Using stale fallback market data.")
    else:
        st.success(f"Data source: {source_labels[status.source]}")
        if status.warning:
            st.warning(status.warning)

    st.caption(
        "Last successful update: "
        f"{status.cached_at_utc.strftime('%Y-%m-%d %H:%M UTC')} | "
        f"Cache age: {_format_age(status.age_seconds)} | "
        f"Latest market date: {snapshot.features.latest_market_date:%Y-%m-%d}"
    )

    market_columns = st.columns(3)
    market_columns[0].metric("JPM Close", f"${snapshot.latest_close:.2f}")
    market_columns[1].metric("VIX Close", f"{snapshot.latest_vix:.2f}")
    market_columns[2].metric(
        "US 10Y Treasury",
        f"{snapshot.latest_treasury_yield:.2f}%",
    )

    quality = snapshot.features.quality.iloc[0]
    st.caption(
        "Feature quality: "
        f"Linear {snapshot.features.linear_latest.shape}, "
        f"LSTM {snapshot.features.lstm_sequence.shape}, "
        f"missing={int(quality['MissingModelValues'])}, "
        f"infinite={int(quality['InfiniteModelValues'])}."
    )


@st.cache_resource(show_spinner="Loading deployed ML models...")
def _load_cached_model_artifacts() -> LoadedModelArtifacts:
    """Load verified model artifacts once per Streamlit process."""

    return load_model_artifacts()


def load_three_model_result(
    snapshot: RealtimeMarketSnapshot,
    artifact_loader: Callable[[], LoadedModelArtifacts] = _load_cached_model_artifacts,
    predictor: Callable[
        [RealTimeFeatureBundle, LoadedModelArtifacts],
        ThreeModelPricingResult,
    ] = predict_three_model_call_prices,
) -> ThreeModelPricingResult:
    """Apply the deployed models to one refreshed real-time snapshot."""

    artifacts = artifact_loader()
    return predictor(snapshot.features, artifacts)


def build_three_model_comparison(result: ThreeModelPricingResult) -> pd.DataFrame:
    """Build the stable three-row comparison table displayed by Streamlit."""

    return pd.DataFrame(
        [
            {
                "Model": "LSTM + BSM",
                "CallPrice": result.lstm_call_price,
                "Difference vs BSM": result.lstm_call_price - result.bsm_call_price,
                "Annual Volatility": result.lstm_annual_volatility,
            },
            {
                "Model": "Linear Regression",
                "CallPrice": result.linear_call_price,
                "Difference vs BSM": result.linear_call_price - result.bsm_call_price,
                "Annual Volatility": np.nan,
            },
            {
                "Model": "BSM Baseline",
                "CallPrice": result.bsm_call_price,
                "Difference vs BSM": 0.0,
                "Annual Volatility": result.baseline_annual_volatility,
            },
        ]
    )


def _render_three_model_comparison(result: ThreeModelPricingResult) -> None:
    st.subheader("Live Three-Model CallPrice Comparison")
    st.caption(
        f"JPM | {result.market_date:%Y-%m-%d} | European Call | "
        "ATM (K=S) | 1 year"
    )
    price_columns = st.columns(3)
    price_columns[0].metric("LSTM + BSM", f"${result.lstm_call_price:.4f}")
    price_columns[1].metric("Linear Regression", f"${result.linear_call_price:.4f}")
    price_columns[2].metric("BSM Baseline", f"${result.bsm_call_price:.4f}")
    st.caption(
        f"Common inputs: Spot = Strike = ${result.spot:.2f}; "
        f"risk-free rate = {result.risk_free_rate * 100:.2f}%; maturity = 1 year."
    )
    comparison = build_three_model_comparison(result)
    st.dataframe(
        comparison.style.format(
            {
                "CallPrice": "${:.4f}",
                "Difference vs BSM": "${:+.4f}",
                "Annual Volatility": lambda value: "N/A"
                if pd.isna(value)
                else f"{value:.2%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    for warning in result.warnings:
        st.warning(warning)
    render_live_error_reference(
        lstm_price=result.lstm_call_price,
        linear_price=result.linear_call_price,
    )
    st.caption(
        "Research prototype only: these model estimates are for comparative analysis "
        "and are not trading or investment advice."
    )


@st.fragment(run_every="60m")
def _render_automatic_market_updates() -> None:
    force_refresh = st.button(
        "Refresh now",
        key="refresh_market_data",
        width="stretch",
    )
    try:
        snapshot = load_realtime_market_snapshot(force_refresh=force_refresh)
    except (CachedMarketDataError, FeatureEngineeringError) as error:
        st.error(str(error))
        return
    _render_snapshot(snapshot)
    try:
        result = load_three_model_result(snapshot)
    except ModelInferenceError as error:
        st.subheader("Live Three-Model CallPrice Comparison")
        st.error(f"Three-model inference is unavailable: {error}")
        return
    _render_three_model_comparison(result)


def render_realtime_market_panel() -> None:
    """Render the opt-in 60-minute automatic market-data panel."""

    st.header("Real-Time Market Data")
    enabled = st.toggle(
        "Enable automatic market updates",
        value=False,
        help="When enabled, this section checks for updates every 60 minutes.",
    )
    if enabled:
        _render_automatic_market_updates()
    else:
        st.caption(
            "Enable this section to check the local cache and start 60-minute updates."
        )
