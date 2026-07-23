"""Reproduce and publish the Week 7 All-VIX model artifacts offline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tempfile
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from Week7.sensitivity_analysis.model_variant_training import (
    call_prices_from_volatility,
    get_feature_variants,
    make_sequences,
    regression_metrics,
)


ARTIFACT_SCHEMA_VERSION = 1
MODEL_VARIANT = "All VIX Features"
WINDOW_SIZE = 20
SEED = 42
LINEAR_ABSOLUTE_TOLERANCE = 1e-9
LSTM_ABSOLUTE_TOLERANCE = {"MAE": 0.05, "RMSE": 0.05, "R2": 0.01}
METRIC_COLUMNS = [
    "Validation_MAE",
    "Validation_RMSE",
    "Validation_R2",
    "Test_MAE",
    "Test_RMSE",
    "Test_R2",
]
WEEK7_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = WEEK7_ROOT / "data"
DEFAULT_OUTPUT_DIR = WEEK7_ROOT / "model_artifacts"
DEFAULT_REFERENCE_METRICS = WEEK7_ROOT / "results" / "task_1_3_model_metrics.csv"


class ArtifactGenerationError(RuntimeError):
    """Raised when reproducible deployment artifacts cannot be published."""


@dataclass(frozen=True)
class ArtifactPaths:
    """Stable paths for the deployed models, scalers, and manifest."""

    linear: Path
    lstm: Path
    x_scaler: Path
    delta_scaler: Path
    manifest: Path

    @classmethod
    def from_directory(cls, directory: str | Path) -> "ArtifactPaths":
        root = Path(directory)
        return cls(
            linear=root / "linear_all_vix.joblib",
            lstm=root / "lstm_all_vix.keras",
            x_scaler=root / "lstm_x_scaler.joblib",
            delta_scaler=root / "lstm_delta_scaler.joblib",
            manifest=root / "manifest.json",
        )

    @property
    def binary_paths(self) -> tuple[Path, Path, Path, Path]:
        return self.linear, self.lstm, self.x_scaler, self.delta_scaler


def _metric_tolerance(model_family: str, metric_column: str) -> float:
    if model_family == "Linear Regression":
        return LINEAR_ABSOLUTE_TOLERANCE
    if model_family != "LSTM + BSM":
        raise ArtifactGenerationError(f"Unsupported model family: {model_family}.")
    metric_name = metric_column.rsplit("_", 1)[-1]
    try:
        return LSTM_ABSOLUTE_TOLERANCE[metric_name]
    except KeyError as error:
        raise ArtifactGenerationError(f"Unsupported metric: {metric_column}.") from error


def verify_metric_reproduction(
    model_family: str,
    observed: Mapping[str, float],
    reference: Mapping[str, float],
) -> None:
    """Reject artifacts whose metrics do not reproduce the stored Week 7 result."""

    for metric_column, reference_value in reference.items():
        if metric_column not in observed:
            raise ArtifactGenerationError(
                f"{model_family} is missing reproduced metric {metric_column}."
            )
        try:
            actual = float(observed[metric_column])
            expected = float(reference_value)
        except (TypeError, ValueError) as error:
            raise ArtifactGenerationError(
                f"{model_family} {metric_column} must be numeric."
            ) from error
        if not np.isfinite([actual, expected]).all():
            raise ArtifactGenerationError(
                f"{model_family} {metric_column} must be finite."
            )
        tolerance = _metric_tolerance(model_family, metric_column)
        if abs(actual - expected) > tolerance:
            raise ArtifactGenerationError(
                f"{model_family} {metric_column} differs from the stored reference "
                f"by more than {tolerance}."
            )


def load_reference_metrics(path: str | Path = DEFAULT_REFERENCE_METRICS) -> dict[str, dict[str, float]]:
    """Load the two stored All-VIX metric rows used as the publication gate."""

    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ArtifactGenerationError("Reference metrics could not be loaded.") from error
    required = {"ModelFamily", "Variant", *METRIC_COLUMNS}
    if not required.issubset(frame.columns):
        raise ArtifactGenerationError("Reference metrics are missing required columns.")
    selected = frame.loc[
        (frame["Variant"] == MODEL_VARIANT)
        & frame["ModelFamily"].isin(["Linear Regression", "LSTM + BSM"])
    ]
    if selected["ModelFamily"].tolist().count("Linear Regression") != 1 or selected[
        "ModelFamily"
    ].tolist().count("LSTM + BSM") != 1:
        raise ArtifactGenerationError("Reference metrics must contain one row per deployed model.")
    result: dict[str, dict[str, float]] = {}
    for _, row in selected.iterrows():
        family = str(row["ModelFamily"])
        values = {column: float(row[column]) for column in METRIC_COLUMNS}
        if not np.isfinite(list(values.values())).all():
            raise ArtifactGenerationError("Reference metrics must be finite.")
        result[family] = values
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_hashes(paths: ArtifactPaths, manifest: Mapping[str, object]) -> None:
    """Verify every binary artifact against its manifest SHA-256 hash."""

    hashes = manifest.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ArtifactGenerationError("Artifact manifest is missing hashes.")
    for path in paths.binary_paths:
        expected = hashes.get(path.name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ArtifactGenerationError(f"Artifact hash is missing for {path.name}.")
        if not path.is_file():
            raise ArtifactGenerationError(f"Artifact file is missing: {path.name}.")
        if _sha256_file(path) != expected:
            raise ArtifactGenerationError(f"Artifact hash mismatch for {path.name}.")


def reset_inherited_artifact_permissions(paths: ArtifactPaths) -> None:
    """Ensure published files inherit permissions from the model-artifact folder."""

    published_paths = [*paths.binary_paths, paths.manifest]
    if platform.system() == "Windows":
        for path in published_paths:
            completed = subprocess.run(
                ["icacls", str(path), "/reset"],
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise ArtifactGenerationError(
                    f"Artifact permissions could not be reset for {path.name}."
                )
        return
    for path in published_paths:
        try:
            path.chmod(0o644)
        except OSError as error:
            raise ArtifactGenerationError(
                f"Artifact permissions could not be reset for {path.name}."
            ) from error


def _load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for name in ("train", "validation", "test"):
        path = data_dir / f"{name}_vix_features.csv"
        try:
            frame = pd.read_csv(path, parse_dates=["Date"])
        except (OSError, pd.errors.ParserError, ValueError) as error:
            raise ArtifactGenerationError(f"{name} data could not be loaded.") from error
        if frame.empty or frame["Date"].isna().any():
            raise ArtifactGenerationError(f"{name} data must contain valid dated rows.")
        frames.append(frame)
    train, validation, test = frames
    if not (
        train["Date"].max() < validation["Date"].min()
        and validation["Date"].max() < test["Date"].min()
    ):
        raise ArtifactGenerationError("Train, validation, and test dates must not overlap.")
    return train, validation, test


def _add_ratio_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["MA5_Ratio"] = result["Close"] / result["MA5"] - 1.0
    result["MA20_Ratio"] = result["Close"] / result["MA20"] - 1.0
    return result


def _metric_row(validation_true, validation_prediction, test_true, test_prediction):
    validation = regression_metrics(validation_true, validation_prediction)
    test = regression_metrics(test_true, test_prediction)
    return {
        **{f"Validation_{name}": value for name, value in validation.items()},
        **{f"Test_{name}": value for name, value in test.items()},
    }


def _train_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    max_epochs: int,
    verbose: int,
):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense, Input, LSTM

    if max_epochs < 1:
        raise ArtifactGenerationError("max_epochs must be positive.")
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    linear_features = get_feature_variants("Linear Regression")[MODEL_VARIANT]
    linear_model = LinearRegression()
    linear_model.fit(train[linear_features], train["CallPrice"])
    linear_metrics = _metric_row(
        validation["CallPrice"],
        linear_model.predict(validation[linear_features]),
        test["CallPrice"],
        linear_model.predict(test[linear_features]),
    )

    train_lstm = _add_ratio_features(train)
    validation_lstm = _add_ratio_features(validation)
    test_lstm = _add_ratio_features(test)
    lstm_features = get_feature_variants("LSTM + BSM")[MODEL_VARIANT]
    target = "RollingVol20"
    delta_scaler = StandardScaler().fit(
        train_lstm[target].diff().dropna().to_numpy().reshape(-1, 1)
    )
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(train_lstm[lstm_features])
    X_validation = x_scaler.transform(validation_lstm[lstm_features])
    X_test = x_scaler.transform(test_lstm[lstm_features])

    X_train_seq, _, _, train_delta = make_sequences(
        X_train[:WINDOW_SIZE],
        train_lstm[target].iloc[:WINDOW_SIZE].to_numpy(),
        X_train[WINDOW_SIZE:],
        train_lstm[target].iloc[WINDOW_SIZE:].to_numpy(),
        WINDOW_SIZE,
    )
    X_validation_seq, validation_base, _, validation_delta = make_sequences(
        X_train,
        train_lstm[target].to_numpy(),
        X_validation,
        validation_lstm[target].to_numpy(),
        WINDOW_SIZE,
    )
    X_test_seq, test_base, _, _ = make_sequences(
        X_validation,
        validation_lstm[target].to_numpy(),
        X_test,
        test_lstm[target].to_numpy(),
        WINDOW_SIZE,
    )
    train_delta_scaled = delta_scaler.transform(train_delta.reshape(-1, 1)).ravel()
    validation_delta_scaled = delta_scaler.transform(
        validation_delta.reshape(-1, 1)
    ).ravel()

    tf.keras.backend.clear_session()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    lstm_model = Sequential(
        [
            Input(shape=(WINDOW_SIZE, len(lstm_features))),
            LSTM(16),
            Dense(1, kernel_initializer="zeros", bias_initializer="zeros"),
        ]
    )
    lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )
    history = lstm_model.fit(
        X_train_seq,
        train_delta_scaled,
        validation_data=(X_validation_seq, validation_delta_scaled),
        epochs=max_epochs,
        batch_size=32,
        shuffle=False,
        verbose=verbose,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-5,
            ),
        ],
    )
    validation_delta_prediction = delta_scaler.inverse_transform(
        lstm_model.predict(X_validation_seq, verbose=0)
    ).ravel()
    test_delta_prediction = delta_scaler.inverse_transform(
        lstm_model.predict(X_test_seq, verbose=0)
    ).ravel()
    validation_volatility = np.maximum(
        validation_base + validation_delta_prediction,
        1e-8,
    )
    test_volatility = np.maximum(test_base + test_delta_prediction, 1e-8)
    validation_call_prediction = call_prices_from_volatility(
        validation_lstm,
        validation_volatility,
    )
    test_call_prediction = call_prices_from_volatility(test_lstm, test_volatility)
    lstm_metrics = _metric_row(
        validation_lstm["CallPrice"],
        validation_call_prediction,
        test_lstm["CallPrice"],
        test_call_prediction,
    )
    lstm_metrics["Epochs"] = len(history.history["loss"])
    lstm_metrics["Best_Validation_Loss"] = float(min(history.history["val_loss"]))
    return (
        linear_model,
        lstm_model,
        x_scaler,
        delta_scaler,
        linear_metrics,
        lstm_metrics,
        linear_features,
        lstm_features,
        train_lstm,
    )


def _date_ranges(train, validation, test) -> dict[str, dict[str, str]]:
    return {
        name: {
            "start": frame["Date"].min().strftime("%Y-%m-%d"),
            "end": frame["Date"].max().strftime("%Y-%m-%d"),
            "rows": int(len(frame)),
        }
        for name, frame in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
    }


def _feature_ranges(
    train: pd.DataFrame,
    linear_features: list[str],
    lstm_features: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        family: {
            feature: {
                "min": float(train[feature].min()),
                "max": float(train[feature].max()),
            }
            for feature in features
        }
        for family, features in (
            ("linear", linear_features),
            ("lstm", lstm_features),
        )
    }


def _package_versions() -> dict[str, str]:
    packages = {
        "tensorflow": "tensorflow",
        "scikit_learn": "scikit-learn",
        "numpy": "numpy",
        "pandas": "pandas",
        "joblib": "joblib",
    }
    versions = {"python": platform.python_version()}
    for key, distribution in packages.items():
        versions[key] = importlib.metadata.version(distribution)
    return versions


def _publish_staged_artifacts(staged: ArtifactPaths, final: ArtifactPaths) -> None:
    final.manifest.parent.mkdir(parents=True, exist_ok=True)
    backup_root = staged.manifest.parent / "backup"
    backup_root.mkdir()
    replaced: list[Path] = []
    backed_up: dict[Path, Path] = {}
    try:
        for destination in (*final.binary_paths, final.manifest):
            if destination.exists():
                backup = backup_root / destination.name
                shutil.copy2(destination, backup)
                backed_up[destination] = backup
        for source, destination in zip(staged.binary_paths, final.binary_paths):
            os.replace(source, destination)
            replaced.append(destination)
        os.replace(staged.manifest, final.manifest)
        replaced.append(final.manifest)
        reset_inherited_artifact_permissions(final)
    except (OSError, ArtifactGenerationError) as error:
        for destination in reversed(replaced):
            backup = backed_up.get(destination)
            if backup is not None and backup.exists():
                os.replace(backup, destination)
            else:
                destination.unlink(missing_ok=True)
        raise ArtifactGenerationError("Artifact publication failed.") from error


def generate_model_artifacts(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    reference_metrics_path: str | Path = DEFAULT_REFERENCE_METRICS,
    max_epochs: int = 100,
    verbose: int = 0,
) -> dict[str, object]:
    """Train, verify, and atomically publish the two models and LSTM scalers."""

    train, validation, test = _load_splits(Path(data_dir))
    reference = load_reference_metrics(reference_metrics_path)
    (
        linear_model,
        lstm_model,
        x_scaler,
        delta_scaler,
        linear_metrics,
        lstm_metrics,
        linear_features,
        lstm_features,
        train_lstm,
    ) = _train_models(train, validation, test, max_epochs=max_epochs, verbose=verbose)
    verify_metric_reproduction("Linear Regression", linear_metrics, reference["Linear Regression"])
    verify_metric_reproduction("LSTM + BSM", lstm_metrics, reference["LSTM + BSM"])

    output_root = Path(output_dir)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent,
        prefix=".model_artifacts_",
    ) as directory:
        staged = ArtifactPaths.from_directory(directory)
        joblib.dump(linear_model, staged.linear)
        lstm_model.save(staged.lstm)
        joblib.dump(x_scaler, staged.x_scaler)
        joblib.dump(delta_scaler, staged.delta_scaler)
        hashes = {path.name: _sha256_file(path) for path in staged.binary_paths}
        manifest: dict[str, object] = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "variant": MODEL_VARIANT,
            "seed": SEED,
            "window_size": WINDOW_SIZE,
            "linear_features": linear_features,
            "lstm_features": lstm_features,
            "targets": {
                "linear": "CallPrice",
                "lstm": "one-day change in RollingVol20",
            },
            "assumptions": {
                "underlying": "JPM",
                "option_type": "European Call",
                "strike": "ATM K=S",
                "maturity_years": 1.0,
                "annualization_days": 252,
            },
            "date_ranges": _date_ranges(train, validation, test),
            "feature_ranges": _feature_ranges(
                train_lstm,
                linear_features,
                lstm_features,
            ),
            "metrics": {
                "Linear Regression": linear_metrics,
                "LSTM + BSM": lstm_metrics,
            },
            "versions": _package_versions(),
            "hashes": hashes,
        }
        staged.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        verify_artifact_hashes(staged, manifest)
        _publish_staged_artifacts(staged, ArtifactPaths.from_directory(output_root))
    return manifest


def main() -> None:
    manifest = generate_model_artifacts()
    print(f"Published model artifacts to: {DEFAULT_OUTPUT_DIR}")
    for family in ("Linear Regression", "LSTM + BSM"):
        metrics = manifest["metrics"][family]
        print(
            f"{family}: validation RMSE={metrics['Validation_RMSE']:.6f}, "
            f"test RMSE={metrics['Test_RMSE']:.6f}, test R2={metrics['Test_R2']:.6f}"
        )


if __name__ == "__main__":
    main()
