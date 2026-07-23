"""Validated runtime inference for the deployed Week 7 pricing models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from Week7.model_deployment.artifact_training import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactGenerationError,
    ArtifactPaths,
    DEFAULT_OUTPUT_DIR,
    MODEL_VARIANT,
    verify_artifact_hashes,
)
from Week7.pricing_tool.feature_engineering_service import (
    LINEAR_FEATURES,
    LSTM_FEATURES,
    RealTimeFeatureBundle,
)
from Week7.pricing_tool.pricing_service import calculate_bsm_prices


WINDOW_SIZE = 20
MATURITY_YEARS = 1.0
ANNUALIZATION_DAYS = 252.0


class ModelInferenceError(RuntimeError):
    """Raised when deployed artifacts or current model inputs are invalid."""


@dataclass(frozen=True)
class LoadedModelArtifacts:
    """Loaded model objects plus their validated deployment manifest."""

    linear_model: object
    lstm_model: object
    x_scaler: object
    delta_scaler: object
    manifest: dict[str, object]


@dataclass(frozen=True)
class ThreeModelPricingResult:
    """Three CallPrice estimates for one common latest ATM one-year contract."""

    market_date: pd.Timestamp
    spot: float
    strike: float
    maturity_years: float
    risk_free_rate: float
    lstm_call_price: float
    linear_call_price: float
    bsm_call_price: float
    lstm_annual_volatility: float
    baseline_annual_volatility: float
    warnings: tuple[str, ...]


def _validate_feature_ranges(manifest: Mapping[str, object]) -> None:
    ranges = manifest.get("feature_ranges")
    if not isinstance(ranges, Mapping):
        raise ModelInferenceError("Artifact manifest is missing feature ranges.")
    for family, features in (("linear", LINEAR_FEATURES), ("lstm", LSTM_FEATURES)):
        family_ranges = ranges.get(family)
        if not isinstance(family_ranges, Mapping):
            raise ModelInferenceError(f"Artifact manifest is missing {family} feature ranges.")
        for feature in features:
            bounds = family_ranges.get(feature)
            if not isinstance(bounds, Mapping) or not {"min", "max"}.issubset(bounds):
                raise ModelInferenceError(
                    f"Artifact manifest is missing the training range for {family} {feature}."
                )
            try:
                lower = float(bounds["min"])
                upper = float(bounds["max"])
            except (TypeError, ValueError) as error:
                raise ModelInferenceError("Artifact feature ranges must be numeric.") from error
            if not np.isfinite([lower, upper]).all() or lower > upper:
                raise ModelInferenceError("Artifact feature ranges must be finite and ordered.")


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ModelInferenceError("Artifact schema version is not supported.")
    if manifest.get("variant") != MODEL_VARIANT:
        raise ModelInferenceError("Artifact model variant is not All VIX Features.")
    if list(manifest.get("linear_features", [])) != LINEAR_FEATURES:
        raise ModelInferenceError("Linear feature order does not match the deployed contract.")
    if list(manifest.get("lstm_features", [])) != LSTM_FEATURES:
        raise ModelInferenceError("LSTM feature order does not match the deployed contract.")
    if manifest.get("window_size") != WINDOW_SIZE:
        raise ModelInferenceError("LSTM window length does not match the deployed contract.")
    _validate_feature_ranges(manifest)


def _feature_count(value: object, name: str) -> int:
    try:
        return int(getattr(value, "n_features_in_"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ModelInferenceError(f"{name} does not expose a valid feature count.") from error


def load_model_artifacts(
    artifact_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> LoadedModelArtifacts:
    """Verify and load the saved model bundle once for runtime inference."""

    paths = ArtifactPaths.from_directory(artifact_dir)
    try:
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        resolved_path = paths.manifest.resolve()
        raise ModelInferenceError(
            "Model artifact manifest could not be loaded from "
            f"{resolved_path} ({type(error).__name__}: {error})."
        ) from error
    if not isinstance(manifest, dict):
        raise ModelInferenceError("Model artifact manifest must be a JSON object.")
    _validate_manifest(manifest)
    try:
        verify_artifact_hashes(paths, manifest)
    except ArtifactGenerationError as error:
        raise ModelInferenceError(str(error)) from error

    try:
        from tensorflow.keras.models import load_model

        linear_model = joblib.load(paths.linear)
        lstm_model = load_model(paths.lstm, compile=False)
        x_scaler = joblib.load(paths.x_scaler)
        delta_scaler = joblib.load(paths.delta_scaler)
    except Exception as error:
        raise ModelInferenceError("Model artifacts could not be loaded.") from error

    if _feature_count(linear_model, "Linear model") != len(LINEAR_FEATURES):
        raise ModelInferenceError("Linear model feature count is invalid.")
    if _feature_count(x_scaler, "LSTM input scaler") != len(LSTM_FEATURES):
        raise ModelInferenceError("LSTM input scaler feature count is invalid.")
    if _feature_count(delta_scaler, "LSTM delta scaler") != 1:
        raise ModelInferenceError("LSTM delta scaler feature count is invalid.")
    try:
        input_shape = tuple(lstm_model.input_shape)
    except (AttributeError, TypeError) as error:
        raise ModelInferenceError("LSTM model input shape is unavailable.") from error
    if input_shape[-2:] != (WINDOW_SIZE, len(LSTM_FEATURES)):
        raise ModelInferenceError("LSTM model input shape is invalid.")
    return LoadedModelArtifacts(
        linear_model=linear_model,
        lstm_model=lstm_model,
        x_scaler=x_scaler,
        delta_scaler=delta_scaler,
        manifest=manifest,
    )


def _training_range_warnings(
    linear_input: pd.DataFrame,
    lstm_input: pd.DataFrame,
    manifest: Mapping[str, object],
) -> list[str]:
    warnings: list[str] = []
    ranges = manifest["feature_ranges"]
    for family, frame, features in (
        ("Linear", linear_input, LINEAR_FEATURES),
        ("LSTM", lstm_input, LSTM_FEATURES),
    ):
        range_key = family.lower()
        for feature in features:
            lower = float(ranges[range_key][feature]["min"])
            upper = float(ranges[range_key][feature]["max"])
            values = frame[feature].to_numpy(dtype=float)
            if (values < lower).any() or (values > upper).any():
                warnings.append(
                    f"{family} feature {feature} is outside its training range "
                    f"[{lower:.6g}, {upper:.6g}]."
                )
    return warnings


def _append_no_arbitrage_warning(
    warnings: list[str],
    model_name: str,
    price: float,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
) -> None:
    lower = max(0.0, spot - strike * np.exp(-rate * maturity))
    upper = spot
    tolerance = 1e-9
    if price < lower - tolerance or price > upper + tolerance:
        warnings.append(
            f"{model_name} CallPrice {price:.6f} violates the no-arbitrage "
            f"interval [{lower:.6f}, {upper:.6f}]; the raw prediction is shown."
        )


def predict_three_model_call_prices(
    features: RealTimeFeatureBundle,
    artifacts: LoadedModelArtifacts,
) -> ThreeModelPricingResult:
    """Price one latest JPM ATM one-year Call with all three deployed paths."""

    _validate_manifest(artifacts.manifest)
    history = features.history.reset_index(drop=True).copy()
    if len(history) < WINDOW_SIZE + 1:
        raise ModelInferenceError("At least 21 valid feature rows are required for inference.")
    required = ["Date", *dict.fromkeys(LINEAR_FEATURES + LSTM_FEATURES)]
    missing = [column for column in required if column not in history.columns]
    if missing:
        raise ModelInferenceError(f"Feature history is missing required columns: {missing}.")
    numeric = history[required[1:]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ModelInferenceError("Feature history values must be finite.")

    latest = history.iloc[-1]
    linear_frame = history.loc[[history.index[-1]], LINEAR_FEATURES]
    lstm_frame = history.iloc[-21:-1].loc[:, LSTM_FEATURES]
    warnings = _training_range_warnings(linear_frame, lstm_frame, artifacts.manifest)

    linear_prediction = np.asarray(
        artifacts.linear_model.predict(linear_frame),
        dtype=float,
    ).reshape(-1)
    if linear_prediction.size != 1 or not np.isfinite(linear_prediction).all():
        raise ModelInferenceError("Linear Regression must return one finite CallPrice.")
    linear_call_price = float(linear_prediction[0])

    try:
        scaled_sequence = np.asarray(
            artifacts.x_scaler.transform(lstm_frame),
            dtype=float,
        ).reshape(1, WINDOW_SIZE, len(LSTM_FEATURES))
        scaled_delta = np.asarray(
            artifacts.lstm_model.predict(scaled_sequence, verbose=0),
            dtype=float,
        ).reshape(-1, 1)
        delta = np.asarray(
            artifacts.delta_scaler.inverse_transform(scaled_delta),
            dtype=float,
        ).reshape(-1)
    except Exception as error:
        raise ModelInferenceError("LSTM inference failed.") from error
    if delta.size != 1 or not np.isfinite(delta).all():
        raise ModelInferenceError("LSTM must return one finite volatility change.")

    predicted_daily_volatility = float(history.iloc[-2]["RollingVol20"]) + float(delta[0])
    baseline_daily_volatility = float(latest["RollingVol20"])
    if predicted_daily_volatility <= 0 or baseline_daily_volatility <= 0:
        raise ModelInferenceError("BSM paths require positive volatility inputs.")
    annualizer = np.sqrt(ANNUALIZATION_DAYS)
    lstm_annual_volatility = predicted_daily_volatility * annualizer
    baseline_annual_volatility = baseline_daily_volatility * annualizer

    spot = float(latest["Close"])
    strike = spot
    rate = float(latest["Treasury_Yield"]) / 100.0
    if spot <= 0 or not np.isfinite([spot, rate]).all():
        raise ModelInferenceError("Latest spot and risk-free rate must be valid.")
    lstm_call_price = calculate_bsm_prices(
        spot,
        strike,
        rate,
        lstm_annual_volatility,
        MATURITY_YEARS,
    )["call_price"]
    bsm_call_price = calculate_bsm_prices(
        spot,
        strike,
        rate,
        baseline_annual_volatility,
        MATURITY_YEARS,
    )["call_price"]
    prices = [lstm_call_price, linear_call_price, bsm_call_price]
    if not np.isfinite(prices).all():
        raise ModelInferenceError("All three CallPrice outputs must be finite.")

    for model_name, price in (
        ("LSTM + BSM", lstm_call_price),
        ("Linear Regression", linear_call_price),
        ("BSM Baseline", bsm_call_price),
    ):
        _append_no_arbitrage_warning(
            warnings,
            model_name,
            price,
            spot,
            strike,
            rate,
            MATURITY_YEARS,
        )

    market_date = pd.Timestamp(latest["Date"]).normalize()
    if pd.isna(market_date):
        raise ModelInferenceError("Latest market date must be valid.")
    return ThreeModelPricingResult(
        market_date=market_date,
        spot=spot,
        strike=strike,
        maturity_years=MATURITY_YEARS,
        risk_free_rate=rate,
        lstm_call_price=float(lstm_call_price),
        linear_call_price=linear_call_price,
        bsm_call_price=float(bsm_call_price),
        lstm_annual_volatility=float(lstm_annual_volatility),
        baseline_annual_volatility=float(baseline_annual_volatility),
        warnings=tuple(warnings),
    )
