"""Streamlit interface for option pricing and final model comparison."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Week7.pricing_tool.dashboard_panel import render_model_performance_dashboard
from Week7.pricing_tool.dashboard_service import build_sensitivity_curves
from Week7.pricing_tool.pricing_service import calculate_from_percent_inputs
from Week7.pricing_tool.realtime_market_panel import render_realtime_market_panel


st.set_page_config(
    page_title="Option Pricing and Model Comparison",
    layout="wide",
)


def render_option_calculator() -> None:
    """Render the manual BSM calculator and sensitivity charts."""

    st.header("Option Calculator")
    st.write(
        "Enter European option parameters to calculate BSM Call and Put prices. "
        "The charts below show how the result changes when one input is varied."
    )

    with st.form("bsm_pricing_form"):
        first_row = st.columns(3)
        spot = first_row[0].number_input(
            "Spot Price (S)",
            value=100.0,
            step=1.0,
            format="%.2f",
            help="Current underlying price; must be greater than zero.",
        )
        strike = first_row[1].number_input(
            "Strike Price (K)",
            value=100.0,
            step=1.0,
            format="%.2f",
            help="Option exercise price; must be greater than zero.",
        )
        maturity = first_row[2].number_input(
            "Maturity (years)",
            value=1.0,
            step=0.25,
            format="%.2f",
            help="Time to maturity in years; must be greater than zero.",
        )

        second_row = st.columns(2)
        rate_percent = second_row[0].number_input(
            "Risk-free Rate (%)",
            value=5.0,
            step=0.25,
            format="%.2f",
            help="Enter 5 for 5 percent.",
        )
        volatility_percent = second_row[1].number_input(
            "Annual Volatility (%)",
            value=20.0,
            step=1.0,
            format="%.2f",
            help="Enter 20 for 20 percent annual volatility.",
        )
        st.form_submit_button(
            "Calculate Option Prices",
            type="primary",
            use_container_width=True,
        )

    try:
        result = calculate_from_percent_inputs(
            spot=spot,
            strike=strike,
            rate_percent=rate_percent,
            volatility_percent=volatility_percent,
            maturity=maturity,
        )
        curves = build_sensitivity_curves(
            spot=spot,
            strike=strike,
            rate=rate_percent / 100.0,
            volatility=volatility_percent / 100.0,
            maturity=maturity,
        )
    except ValueError as error:
        st.error(str(error))
        return

    st.subheader("Pricing Results")
    call_column, put_column = st.columns(2)
    call_column.metric("European Call Price", f"${result['call_price']:.4f}")
    put_column.metric("European Put Price", f"${result['put_price']:.4f}")
    if abs(result["parity_residual"]) <= 1e-10:
        st.success(
            "Put-Call Parity passed: residual = "
            f"{result['parity_residual']:.3e}"
        )
    else:
        st.warning(
            "Put-Call Parity residual is larger than expected: "
            f"{result['parity_residual']:.3e}"
        )

    st.subheader("Sensitivity Analysis")
    sensitivity_tabs = st.tabs(["Spot Price", "Volatility", "Interest Rate"])
    with sensitivity_tabs[0]:
        st.line_chart(
            curves["Spot"].set_index("Spot")[["Call Price", "Put Price"]],
            x_label="Spot Price",
            y_label="Option Price",
        )
    with sensitivity_tabs[1]:
        st.line_chart(
            curves["Volatility"].set_index("Annual Volatility (%)")[
                ["Call Price", "Put Price"]
            ],
            x_label="Annual Volatility (%)",
            y_label="Option Price",
        )
    with sensitivity_tabs[2]:
        st.line_chart(
            curves["Rate"].set_index("Risk-free Rate (%)")[
                ["Call Price", "Put Price"]
            ],
            x_label="Risk-free Rate (%)",
            y_label="Option Price",
        )

    with st.expander("BSM formulas and assumptions"):
        st.latex(
            r"d_1=\frac{\ln(S/K)+(r+\frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},"
            r"\qquad d_2=d_1-\sigma\sqrt{T}"
        )
        st.latex(
            r"C=SN(d_1)-Ke^{-rT}N(d_2),\qquad "
            r"P=Ke^{-rT}N(-d_2)-SN(-d_1)"
        )
        st.markdown(
            "Assumptions: European exercise, no dividends, constant rate and "
            "volatility, frictionless markets and lognormal underlying prices."
        )


st.title("Option Pricing and Model Comparison")
st.caption("Week 8 tool finalisation")
st.write(
    "Use the calculator for a manual BSM price, compare the three deployed models "
    "with current market inputs, and review their saved test performance."
)

calculator_tab, live_tab, performance_tab = st.tabs(
    ["Option Calculator", "Live Model Comparison", "Model Performance"]
)
with calculator_tab:
    render_option_calculator()
with live_tab:
    render_realtime_market_panel()
with performance_tab:
    try:
        render_model_performance_dashboard()
    except (OSError, ValueError) as error:
        st.error(f"Saved performance results could not be loaded: {error}")
