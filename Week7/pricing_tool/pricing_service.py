"""Pure Black-Scholes-Merton calculations for the Streamlit pricing tool."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def validate_bsm_inputs(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> None:
    """Validate the numerical domain required by the BSM equations."""

    values = {
        "spot": spot,
        "strike": strike,
        "rate": rate,
        "volatility": volatility,
        "maturity": maturity,
    }
    for name, value in values.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be numeric and finite") from error
        if not np.isfinite(numeric):
            raise ValueError(f"{name} must be finite")

    for name in ("spot", "strike", "volatility", "maturity"):
        if float(values[name]) <= 0:
            raise ValueError(f"{name} must be greater than zero")


def calculate_bsm_prices(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> dict[str, float]:
    """Calculate European Call and Put prices without dividends."""

    validate_bsm_inputs(spot, strike, rate, volatility, maturity)
    spot, strike, rate, volatility, maturity = map(
        float, (spot, strike, rate, volatility, maturity)
    )
    sqrt_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike)
        + (rate + 0.5 * volatility**2) * maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    discounted_strike = strike * np.exp(-rate * maturity)
    call_price = spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    put_price = discounted_strike * norm.cdf(-d2) - spot * norm.cdf(-d1)
    parity_residual = (call_price - put_price) - (spot - discounted_strike)

    return {
        "call_price": float(call_price),
        "put_price": float(put_price),
        "d1": float(d1),
        "d2": float(d2),
        "parity_residual": float(parity_residual),
    }


def calculate_from_percent_inputs(
    spot: float,
    strike: float,
    rate_percent: float,
    volatility_percent: float,
    maturity: float,
) -> dict[str, float]:
    """Convert user-facing percentages and calculate BSM prices."""

    try:
        rate = float(rate_percent) / 100.0
        volatility = float(volatility_percent) / 100.0
    except (TypeError, ValueError) as error:
        raise ValueError("rate_percent and volatility_percent must be numeric") from error
    return calculate_bsm_prices(spot, strike, rate, volatility, maturity)


def build_spot_price_curve(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    lower_multiplier: float = 0.7,
    upper_multiplier: float = 1.3,
    points: int = 61,
) -> pd.DataFrame:
    """Calculate Call and Put prices across a range of underlying prices."""

    validate_bsm_inputs(spot, strike, rate, volatility, maturity)
    try:
        lower = float(lower_multiplier)
        upper = float(upper_multiplier)
    except (TypeError, ValueError) as error:
        raise ValueError("curve multipliers must be numeric and finite") from error
    if not np.isfinite([lower, upper]).all():
        raise ValueError("curve multipliers must be finite")
    if lower <= 0:
        raise ValueError("lower_multiplier must be greater than zero")
    if upper <= lower:
        raise ValueError("upper_multiplier must be greater than lower_multiplier")
    if not isinstance(points, (int, np.integer)) or points < 2:
        raise ValueError("points must be an integer greater than or equal to two")

    spot_values = np.linspace(float(spot) * lower, float(spot) * upper, points)
    rows = []
    for spot_value in spot_values:
        result = calculate_bsm_prices(
            spot_value,
            strike,
            rate,
            volatility,
            maturity,
        )
        rows.append(
            {
                "Spot": float(spot_value),
                "Call Price": result["call_price"],
                "Put Price": result["put_price"],
            }
        )
    return pd.DataFrame(rows)
