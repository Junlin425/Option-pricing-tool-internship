"""Streamlit presentation helpers for the Week 8 model dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from Week7.pricing_tool.dashboard_service import (
    build_error_reference_table,
    build_residual_references,
    prepare_performance_dashboard,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "Week7" / "results"


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load saved final metrics and test predictions."""

    metrics = pd.read_csv(RESULT_DIR / "task_1_3_model_metrics.csv")
    predictions = pd.read_csv(
        RESULT_DIR / "task_1_3_test_predictions.csv",
        parse_dates=["Date"],
    )
    return metrics, predictions


def render_live_error_reference(lstm_price: float, linear_price: float) -> None:
    """Show empirical test-residual ranges around current ML estimates."""

    _, predictions = load_dashboard_data()
    references = build_residual_references(predictions)
    table = build_error_reference_table(
        {
            "LSTM + BSM": lstm_price,
            "Linear Regression": linear_price,
        },
        references,
    )
    st.subheader("Historical Error Reference")
    st.dataframe(
        table.style.format(
            {
                "Point Estimate": "${:.4f}",
                "Lower Reference": "${:.4f}",
                "Upper Reference": "${:.4f}",
                "Test MAE": "{:.4f}",
                "Test RMSE": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "The lower and upper values add the 5th and 95th percentiles of held-out "
        "test residuals to the current estimate. This is a historical error reference, "
        "not a confidence interval or guarantee."
    )


def render_model_performance_dashboard() -> None:
    """Render final test metrics, prediction trend and residual information."""

    st.header("Model Performance")
    metrics, predictions = load_dashboard_data()
    final_metrics, trend = prepare_performance_dashboard(metrics, predictions)
    references = build_residual_references(predictions)

    st.subheader("Final Test Metrics")
    st.caption("MAE and RMSE measure price error; R² measures explained variation.")
    columns = st.columns(2)
    for column, (_, row) in zip(columns, final_metrics.iterrows()):
        column.markdown(f"**{row['Model']}**")
        metric_columns = column.columns(3)
        metric_columns[0].metric("MAE", f"{row['Test_MAE']:.3f}")
        metric_columns[1].metric("RMSE", f"{row['Test_RMSE']:.3f}")
        metric_columns[2].metric("R²", f"{row['Test_R2']:.3f}")

    st.caption(
        "Metrics use the untouched 15 December 2023 to 30 December 2024 test set. "
        "CallPrice is a BSM-generated research target rather than an observed option quote."
    )
    st.dataframe(
        final_metrics.style.format(
            {"Test_MAE": "{:.4f}", "Test_RMSE": "{:.4f}", "Test_R2": "{:.4f}"}
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Test Prediction Trend")
    st.line_chart(
        trend.set_index("Date"),
        x_label="Test Date",
        y_label="CallPrice",
    )

    st.subheader("Residual Reference")
    st.dataframe(
        references[
            ["Model", "LowerResidual", "UpperResidual", "MAE", "RMSE", "TestRows"]
        ].style.format(
            {
                "LowerResidual": "{:+.4f}",
                "UpperResidual": "{:+.4f}",
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Residual = target CallPrice − model prediction. The 5th and 95th percentiles "
        "describe past test errors and are not formal confidence bounds."
    )
