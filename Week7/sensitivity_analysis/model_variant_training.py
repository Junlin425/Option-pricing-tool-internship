"""Reusable model-variant utilities for Week 7 Task 1.3."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


VIX_FEATURES = [
    "VIX_Close",
    "VIX_Return",
    "VIX_5D_Change",
    "VIX_20D_ZScore",
    "VIX_Regime",
    "VIX_Spike",
]


def get_feature_variants(model_family: str) -> dict[str, list[str]]:
    """Return the three matched VIX treatments for one model family."""

    if model_family == "Linear Regression":
        base = ["Close", "Treasury_Yield", "RollingVol20", "VolumeChange"]
    elif model_family == "LSTM + BSM":
        base = [
            "RollingVol20",
            "Return",
            "MA5_Ratio",
            "MA20_Ratio",
            "VolumeChange",
            "RateMomentum",
            "Treasury_Yield",
        ]
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    return {
        "No VIX": base.copy(),
        "VIX Close Only": base + ["VIX_Close"],
        "All VIX Features": base + VIX_FEATURES,
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate the common MAE, RMSE, and R-squared metrics."""

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_sequences(
    X_history,
    y_history,
    X_current,
    y_current,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use previous-period context and predict every row in the current period."""

    X_history = np.asarray(X_history, dtype=float)
    y_history = np.asarray(y_history, dtype=float).reshape(-1)
    X_current = np.asarray(X_current, dtype=float)
    y_current = np.asarray(y_current, dtype=float).reshape(-1)
    if window < 1 or len(X_history) < window or len(y_history) < window:
        raise ValueError("history must contain at least window observations")
    if len(X_current) != len(y_current):
        raise ValueError("current features and target must have the same length")

    X_all = np.vstack([X_history[-window:], X_current])
    y_all = np.concatenate([y_history[-window:], y_current])
    sequences, previous_vol, truth, delta = [], [], [], []
    for i in range(len(y_current)):
        end = i + window
        sequences.append(X_all[i:end])
        previous_vol.append(y_all[end - 1])
        truth.append(y_all[end])
        delta.append(y_all[end] - y_all[end - 1])
    return tuple(
        np.asarray(values)
        for values in (sequences, previous_vol, truth, delta)
    )


def call_prices_from_volatility(
    data: pd.DataFrame,
    daily_volatility,
    maturity: float = 1.0,
) -> np.ndarray:
    """Convert predicted daily volatility to ATM Black-Scholes call prices."""

    daily_volatility = np.asarray(daily_volatility, dtype=float).reshape(-1)
    if len(data) != len(daily_volatility):
        raise ValueError("data and daily_volatility must have the same length")
    if maturity <= 0:
        raise ValueError("maturity must be positive")

    spot = data["Close"].to_numpy(dtype=float)
    strike = spot
    rate = data["Treasury_Yield"].to_numpy(dtype=float) / 100.0
    sigma = np.maximum(daily_volatility, 1e-8) * np.sqrt(252.0)
    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * maturity) / (
        sigma * sqrt_t
    )
    d2 = d1 - sigma * sqrt_t
    prices = spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)
    return np.asarray(prices, dtype=float)


def _add_ratio_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["MA5_Ratio"] = result["Close"] / result["MA5"] - 1.0
    result["MA20_Ratio"] = result["Close"] / result["MA20"] - 1.0
    return result


def _result_row(
    model_family: str,
    variant: str,
    validation_true,
    validation_prediction,
    test_true,
    test_prediction,
) -> dict[str, object]:
    validation_metrics = regression_metrics(validation_true, validation_prediction)
    test_metrics = regression_metrics(test_true, test_prediction)
    return {
        "ModelFamily": model_family,
        "Variant": variant,
        **{f"Validation_{name}": value for name, value in validation_metrics.items()},
        **{f"Test_{name}": value for name, value in test_metrics.items()},
    }


def train_linear_variants(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit the three aligned Linear Regression feature variants."""

    variants = get_feature_variants("Linear Regression")
    results = []
    predictions = {}
    for variant, features in variants.items():
        model = LinearRegression()
        model.fit(train[features], train["CallPrice"])
        validation_prediction = model.predict(validation[features])
        test_prediction = model.predict(test[features])
        results.append(
            _result_row(
                "Linear Regression",
                variant,
                validation["CallPrice"],
                validation_prediction,
                test["CallPrice"],
                test_prediction,
            )
        )
        predictions[variant] = np.asarray(test_prediction, dtype=float)
    return pd.DataFrame(results), predictions


def train_lstm_variants(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    seed: int = 42,
    window_size: int = 20,
    max_epochs: int = 100,
    batch_size: int = 32,
    verbose: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """Fit three constant-architecture LSTM variants and convert them via BSM."""

    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense, Input, LSTM

    if max_epochs < 1 or batch_size < 1:
        raise ValueError("max_epochs and batch_size must be positive")
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    train = _add_ratio_features(train)
    validation = _add_ratio_features(validation)
    test = _add_ratio_features(test)
    target = "RollingVol20"
    delta_scaler = StandardScaler().fit(
        train[target].diff().dropna().to_numpy().reshape(-1, 1)
    )

    results = []
    volatility_results = []
    predictions = {}
    for variant, features in get_feature_variants("LSTM + BSM").items():
        tf.keras.backend.clear_session()
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)

        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(train[features])
        X_validation = x_scaler.transform(validation[features])
        X_test = x_scaler.transform(test[features])

        X_train_seq, _, _, train_delta = make_sequences(
            X_train[:window_size],
            train[target].iloc[:window_size].to_numpy(),
            X_train[window_size:],
            train[target].iloc[window_size:].to_numpy(),
            window_size,
        )
        X_validation_seq, validation_base, validation_vol_true, validation_delta = make_sequences(
            X_train,
            train[target].to_numpy(),
            X_validation,
            validation[target].to_numpy(),
            window_size,
        )
        X_test_seq, test_base, test_vol_true, _ = make_sequences(
            X_validation,
            validation[target].to_numpy(),
            X_test,
            test[target].to_numpy(),
            window_size,
        )

        train_delta_scaled = delta_scaler.transform(train_delta.reshape(-1, 1)).ravel()
        validation_delta_scaled = delta_scaler.transform(
            validation_delta.reshape(-1, 1)
        ).ravel()

        model = Sequential(
            [
                Input(shape=(window_size, len(features))),
                LSTM(16),
                Dense(1, kernel_initializer="zeros", bias_initializer="zeros"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
        )
        history = model.fit(
            X_train_seq,
            train_delta_scaled,
            validation_data=(X_validation_seq, validation_delta_scaled),
            epochs=max_epochs,
            batch_size=batch_size,
            shuffle=False,
            verbose=verbose,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss", patience=12, restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-5,
                ),
            ],
        )

        validation_delta_prediction = delta_scaler.inverse_transform(
            model.predict(X_validation_seq, verbose=0)
        ).ravel()
        test_delta_prediction = delta_scaler.inverse_transform(
            model.predict(X_test_seq, verbose=0)
        ).ravel()
        validation_volatility = np.maximum(
            validation_base + validation_delta_prediction, 1e-8
        )
        test_volatility = np.maximum(test_base + test_delta_prediction, 1e-8)
        validation_call_prediction = call_prices_from_volatility(
            validation, validation_volatility
        )
        test_call_prediction = call_prices_from_volatility(test, test_volatility)

        row = _result_row(
            "LSTM + BSM",
            variant,
            validation["CallPrice"],
            validation_call_prediction,
            test["CallPrice"],
            test_call_prediction,
        )
        row["Epochs"] = len(history.history["loss"])
        row["Best_Validation_Loss"] = float(min(history.history["val_loss"]))
        results.append(row)

        volatility_row = _result_row(
            "LSTM Volatility",
            variant,
            validation_vol_true,
            validation_volatility,
            test_vol_true,
            test_volatility,
        )
        volatility_results.append(volatility_row)
        predictions[variant] = np.asarray(test_call_prediction, dtype=float)

    return pd.DataFrame(results), predictions, pd.DataFrame(volatility_results)
