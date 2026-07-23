"""End-to-end validation for the Week 7 Stage 3 real-time data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Literal
from uuid import uuid4

import numpy as np
import pandas as pd

from Week7.pricing_tool.feature_engineering_service import (
    LINEAR_FEATURES,
    LSTM_FEATURES,
    RealTimeFeatureBundle,
    build_realtime_features,
)
from Week7.pricing_tool.market_data_cache import (
    CachedMarketDataResult,
    load_cached_market_data,
)
from Week7.pricing_tool.model_inference_service import (
    LoadedModelArtifacts,
    ThreeModelPricingResult,
    load_model_artifacts,
    predict_three_model_call_prices,
)


DEFAULT_REPORT_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "task_3_6_end_to_end_report.csv"
)
REPORT_COLUMNS = [
    "RunAtUTC",
    "OverallStatus",
    "DataSource",
    "IsStale",
    "RefreshAttempted",
    "CachedAtUTC",
    "CacheAgeSeconds",
    "LatestMarketDate",
    "JPMRows",
    "VIXRows",
    "TreasuryRows",
    "MergedRows",
    "ValidFeatureRows",
    "LinearRows",
    "LinearFeatures",
    "LSTMRows",
    "LSTMFeatures",
    "MissingModelValues",
    "InfiniteModelValues",
    "ThresholdSource",
    "LSTMCallPrice",
    "LinearCallPrice",
    "BSMBaselineCallPrice",
    "LSTMAnnualVolatility",
    "BaselineAnnualVolatility",
    "ModelWarningCount",
    "PassedChecks",
    "TotalChecks",
    "FailedChecks",
]


class EndToEndValidationError(RuntimeError):
    """Raised when a completed validation report cannot be persisted."""


@dataclass(frozen=True)
class EndToEndValidationResult:
    """One-row validation report and its aggregate pass/fail status."""

    report: pd.DataFrame
    overall_status: Literal["PASS", "FAIL"]


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _dates_valid(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns or frame.empty:
        return False
    dates = pd.to_datetime(frame[column], errors="coerce")
    return bool(
        not dates.isna().any()
        and not dates.duplicated().any()
        and dates.is_monotonic_increasing
    )


def validate_end_to_end(
    cache_loader: Callable[..., CachedMarketDataResult] = load_cached_market_data,
    feature_builder: Callable[..., RealTimeFeatureBundle] = build_realtime_features,
    artifact_loader: Callable[[], LoadedModelArtifacts] = load_model_artifacts,
    predictor: Callable[
        [RealTimeFeatureBundle, LoadedModelArtifacts],
        ThreeModelPricingResult,
    ] = predict_three_model_call_prices,
    force_refresh: bool = False,
    run_at_utc: pd.Timestamp | None = None,
) -> EndToEndValidationResult:
    """Run market data through deployed inference and summarize 29 checks."""

    cached = cache_loader(force_refresh=force_refresh)
    features = feature_builder(cached.bundle)
    bundle = cached.bundle
    status = cached.status
    run_at = _utc_timestamp(
        pd.Timestamp.now(tz="UTC") if run_at_utc is None else run_at_utc
    )

    merged_dates = pd.to_datetime(bundle.merged.get("Date"), errors="coerce")
    vix_dates = pd.to_datetime(bundle.vix.get("Date"), errors="coerce")
    treasury_observation = pd.to_datetime(
        bundle.merged.get("Treasury_Observation_Date"), errors="coerce"
    )
    staleness = pd.to_numeric(
        bundle.merged.get("Treasury_Staleness_Days"), errors="coerce"
    )
    try:
        cached_at = _utc_timestamp(status.cached_at_utc)
        cache_time_valid = bool(
            np.isfinite(float(status.age_seconds))
            and float(status.age_seconds) >= 0
        )
    except (TypeError, ValueError):
        cached_at = pd.NaT
        cache_time_valid = False

    quality = features.quality.iloc[0]
    latest_merged_date = pd.Timestamp(bundle.merged["Date"].max()).normalize()
    artifacts = None
    model_result = None
    try:
        artifacts = artifact_loader()
        model_result = predictor(features, artifacts)
    except Exception:
        artifacts = None
        model_result = None

    latest_feature_row = features.history.iloc[-1]
    manifest = getattr(artifacts, "manifest", None)
    artifact_manifest_valid = bool(
        isinstance(manifest, dict) and manifest.get("schema_version") == 1
    )
    if model_result is None:
        model_date_consistent = False
        model_contract_consistent = False
        lstm_call_price = np.nan
        linear_call_price = np.nan
        bsm_call_price = np.nan
        lstm_annual_volatility = np.nan
        baseline_annual_volatility = np.nan
        model_warning_count = 0
    else:
        model_date_consistent = bool(
            pd.Timestamp(model_result.market_date).normalize()
            == pd.Timestamp(features.latest_market_date).normalize()
        )
        model_contract_consistent = bool(
            np.isclose(model_result.spot, float(latest_feature_row["Close"]))
            and np.isclose(model_result.strike, model_result.spot)
            and np.isclose(model_result.maturity_years, 1.0)
            and np.isclose(
                model_result.risk_free_rate,
                float(latest_feature_row["Treasury_Yield"]) / 100.0,
            )
        )
        lstm_call_price = float(model_result.lstm_call_price)
        linear_call_price = float(model_result.linear_call_price)
        bsm_call_price = float(model_result.bsm_call_price)
        lstm_annual_volatility = float(model_result.lstm_annual_volatility)
        baseline_annual_volatility = float(model_result.baseline_annual_volatility)
        model_warning_count = len(model_result.warnings)
    checks = {
        "jpm_non_empty": not bundle.jpm.empty,
        "vix_non_empty": not bundle.vix.empty,
        "treasury_non_empty": not bundle.treasury.empty,
        "merged_minimum_rows": len(bundle.merged) >= 40,
        "jpm_dates_valid": _dates_valid(bundle.jpm, "Date"),
        "vix_dates_valid": _dates_valid(bundle.vix, "Date"),
        "treasury_dates_valid": _dates_valid(
            bundle.treasury, "Treasury_Observation_Date"
        ),
        "merged_dates_valid": _dates_valid(bundle.merged, "Date"),
        "vix_exact_alignment": bool(
            not merged_dates.isna().any()
            and set(merged_dates).issubset(set(vix_dates.dropna()))
        ),
        "treasury_not_future": bool(
            not treasury_observation.isna().any()
            and not merged_dates.isna().any()
            and (treasury_observation <= merged_dates).all()
        ),
        "treasury_staleness_valid": bool(
            not staleness.isna().any()
            and staleness.between(0, 7, inclusive="both").all()
        ),
        "cache_source_valid": status.source
        in {"live", "fresh_cache", "stale_fallback"},
        "cache_time_valid": cache_time_valid,
        "latest_date_consistent": pd.Timestamp(features.latest_market_date).normalize()
        == latest_merged_date,
        "linear_shape": features.linear_latest.shape == (1, 10),
        "lstm_shape": features.lstm_sequence.shape == (20, 13),
        "linear_columns": list(features.linear_latest.columns) == LINEAR_FEATURES,
        "lstm_columns": list(features.lstm_sequence.columns) == LSTM_FEATURES,
        "missing_model_values_zero": int(quality["MissingModelValues"]) == 0,
        "infinite_model_values_zero": int(quality["InfiniteModelValues"]) == 0,
        "threshold_source_train_only": quality["ThresholdSource"] == "train_only",
        "artifact_manifest_valid": artifact_manifest_valid,
        "three_model_date_consistent": model_date_consistent,
        "three_model_contract_consistent": model_contract_consistent,
        "lstm_call_price_finite": bool(np.isfinite(lstm_call_price)),
        "linear_call_price_finite": bool(np.isfinite(linear_call_price)),
        "bsm_call_price_finite": bool(np.isfinite(bsm_call_price)),
        "lstm_volatility_positive": bool(
            np.isfinite(lstm_annual_volatility) and lstm_annual_volatility > 0
        ),
        "baseline_volatility_positive": bool(
            np.isfinite(baseline_annual_volatility)
            and baseline_annual_volatility > 0
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not bool(passed)]
    overall_status: Literal["PASS", "FAIL"] = "PASS" if not failed_checks else "FAIL"
    linear_rows, linear_features = features.linear_latest.shape
    lstm_rows, lstm_features = features.lstm_sequence.shape
    report = pd.DataFrame(
        [
            {
                "RunAtUTC": run_at,
                "OverallStatus": overall_status,
                "DataSource": status.source,
                "IsStale": bool(status.is_stale),
                "RefreshAttempted": bool(status.refresh_attempted),
                "CachedAtUTC": cached_at,
                "CacheAgeSeconds": float(status.age_seconds),
                "LatestMarketDate": latest_merged_date,
                "JPMRows": len(bundle.jpm),
                "VIXRows": len(bundle.vix),
                "TreasuryRows": len(bundle.treasury),
                "MergedRows": len(bundle.merged),
                "ValidFeatureRows": len(features.history),
                "LinearRows": linear_rows,
                "LinearFeatures": linear_features,
                "LSTMRows": lstm_rows,
                "LSTMFeatures": lstm_features,
                "MissingModelValues": int(quality["MissingModelValues"]),
                "InfiniteModelValues": int(quality["InfiniteModelValues"]),
                "ThresholdSource": quality["ThresholdSource"],
                "LSTMCallPrice": lstm_call_price,
                "LinearCallPrice": linear_call_price,
                "BSMBaselineCallPrice": bsm_call_price,
                "LSTMAnnualVolatility": lstm_annual_volatility,
                "BaselineAnnualVolatility": baseline_annual_volatility,
                "ModelWarningCount": model_warning_count,
                "PassedChecks": len(checks) - len(failed_checks),
                "TotalChecks": len(checks),
                "FailedChecks": ";".join(failed_checks),
            }
        ],
        columns=REPORT_COLUMNS,
    )
    return EndToEndValidationResult(report=report, overall_status=overall_status)


def write_end_to_end_report(
    result: EndToEndValidationResult,
    output_path: str | Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Atomically write and verify the one-row end-to-end CSV report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as stream:
            result.report.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        loaded = pd.read_csv(temporary_path, keep_default_na=False)
        if loaded.shape != (1, len(REPORT_COLUMNS)) or list(loaded.columns) != REPORT_COLUMNS:
            raise EndToEndValidationError("End-to-end report verification failed.")
        os.replace(temporary_path, path)
    except Exception as error:
        raise EndToEndValidationError("End-to-end report could not be written.") from error
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
    return path
