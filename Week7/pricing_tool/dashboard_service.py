"""Pure data preparation for the Week 8 Streamlit dashboard."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from Week7.pricing_tool.pricing_service import (
    build_spot_price_curve,
    calculate_bsm_prices,
    validate_bsm_inputs,
)


FINAL_PREDICTION_COLUMNS = {
    "LSTM + BSM": "LSTM_BSM_All_VIX_Features",
    "Linear Regression": "Linear_All_VIX_Features",
}


def calculate_residual_reference(
    predictions: pd.DataFrame,
    actual_column: str,
    prediction_column: str,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> dict[str, float]:
    """Return an empirical residual reference from held-out predictions."""

    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("residual quantiles must satisfy 0 <= lower < upper <= 1")
    missing = [
        column
        for column in (actual_column, prediction_column)
        if column not in predictions.columns
    ]
    if missing:
        raise ValueError(f"prediction data is missing columns: {missing}")
    actual = pd.to_numeric(predictions[actual_column], errors="raise")
    predicted = pd.to_numeric(predictions[prediction_column], errors="raise")
    residuals = actual - predicted
    if residuals.empty or not np.isfinite(residuals).all():
        raise ValueError("residuals must be non-empty and finite")
    return {
        "LowerResidual": float(residuals.quantile(lower_quantile)),
        "UpperResidual": float(residuals.quantile(upper_quantile)),
        "MAE": float(residuals.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(residuals)))),
        "Coverage": float(upper_quantile - lower_quantile),
        "TestRows": int(len(residuals)),
    }


def build_residual_references(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate final-model 90% empirical residual references."""

    rows = []
    for model, column in FINAL_PREDICTION_COLUMNS.items():
        rows.append(
            {
                "Model": model,
                **calculate_residual_reference(
                    predictions,
                    "Actual_CallPrice",
                    column,
                ),
            }
        )
    return pd.DataFrame(rows)


def build_error_reference_table(
    live_prices: Mapping[str, float],
    references: pd.DataFrame,
) -> pd.DataFrame:
    """Apply empirical test residuals to current point estimates."""

    required = {"Model", "LowerResidual", "UpperResidual", "MAE", "RMSE"}
    missing = sorted(required.difference(references.columns))
    if missing:
        raise ValueError(f"residual reference is missing columns: {missing}")
    rows = []
    reference_lookup = references.set_index("Model")
    for model, estimate in live_prices.items():
        if model not in reference_lookup.index:
            raise ValueError(f"no residual reference is available for {model}")
        row = reference_lookup.loc[model]
        numeric_estimate = float(estimate)
        if not np.isfinite(numeric_estimate):
            raise ValueError("live model estimates must be finite")
        rows.append(
            {
                "Model": model,
                "Point Estimate": numeric_estimate,
                "Lower Reference": max(
                    0.0,
                    numeric_estimate + float(row["LowerResidual"]),
                ),
                "Upper Reference": numeric_estimate + float(row["UpperResidual"]),
                "Test MAE": float(row["MAE"]),
                "Test RMSE": float(row["RMSE"]),
            }
        )
    return pd.DataFrame(rows)


def prepare_performance_dashboard(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the final all-VIX metrics and prediction trend."""

    metric_required = {
        "ModelFamily",
        "Variant",
        "Test_MAE",
        "Test_RMSE",
        "Test_R2",
    }
    prediction_required = {"Date", "Actual_CallPrice", *FINAL_PREDICTION_COLUMNS.values()}
    missing_metrics = sorted(metric_required.difference(metrics.columns))
    missing_predictions = sorted(prediction_required.difference(predictions.columns))
    if missing_metrics:
        raise ValueError(f"metrics data is missing columns: {missing_metrics}")
    if missing_predictions:
        raise ValueError(f"prediction data is missing columns: {missing_predictions}")

    selected = metrics.loc[metrics["Variant"].eq("All VIX Features")].copy()
    selected = selected.loc[selected["ModelFamily"].isin(FINAL_PREDICTION_COLUMNS)]
    selected["Model"] = selected["ModelFamily"]
    order = pd.Categorical(
        selected["Model"],
        categories=["LSTM + BSM", "Linear Regression"],
        ordered=True,
    )
    selected = selected.assign(_order=order).sort_values("_order")
    final_metrics = selected[
        ["Model", "Test_MAE", "Test_RMSE", "Test_R2"]
    ].reset_index(drop=True)
    if len(final_metrics) != 2:
        raise ValueError("final all-VIX metrics must contain both deployed ML models")

    trend = predictions[
        ["Date", "Actual_CallPrice", *FINAL_PREDICTION_COLUMNS.values()]
    ].copy()
    trend["Date"] = pd.to_datetime(trend["Date"], errors="raise")
    trend = trend.rename(
        columns={
            "Actual_CallPrice": "Target CallPrice",
            **{column: model for model, column in FINAL_PREDICTION_COLUMNS.items()},
        }
    ).sort_values("Date").reset_index(drop=True)
    return final_metrics, trend


def build_sensitivity_curves(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    points: int = 41,
) -> dict[str, pd.DataFrame]:
    """Build BSM Call/Put curves for spot, volatility and rate changes."""

    validate_bsm_inputs(spot, strike, rate, volatility, maturity)
    if not isinstance(points, (int, np.integer)) or points < 2:
        raise ValueError("points must be an integer greater than or equal to two")

    spot_curve = build_spot_price_curve(
        spot, strike, rate, volatility, maturity, points=points
    )
    volatility_values = np.linspace(max(float(volatility) * 0.5, 1e-4), float(volatility) * 1.5, points)
    volatility_rows = []
    for value in volatility_values:
        prices = calculate_bsm_prices(spot, strike, rate, value, maturity)
        volatility_rows.append(
            {
                "Annual Volatility (%)": value * 100.0,
                "Call Price": prices["call_price"],
                "Put Price": prices["put_price"],
            }
        )

    rate_values = np.linspace(float(rate) - 0.02, float(rate) + 0.02, points)
    rate_rows = []
    for value in rate_values:
        prices = calculate_bsm_prices(spot, strike, value, volatility, maturity)
        rate_rows.append(
            {
                "Risk-free Rate (%)": value * 100.0,
                "Call Price": prices["call_price"],
                "Put Price": prices["put_price"],
            }
        )
    return {
        "Spot": spot_curve,
        "Volatility": pd.DataFrame(volatility_rows),
        "Rate": pd.DataFrame(rate_rows),
    }
