import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Week7.model_deployment.artifact_training import (
    ArtifactGenerationError,
    ArtifactPaths,
    DEFAULT_OUTPUT_DIR,
    load_reference_metrics,
    reset_inherited_artifact_permissions,
    verify_artifact_hashes,
    verify_metric_reproduction,
)


class ArtifactPathTests(unittest.TestCase):
    def test_default_artifact_names_are_stable(self):
        paths = ArtifactPaths.from_directory(Path("artifacts"))

        self.assertEqual(paths.linear.name, "linear_all_vix.joblib")
        self.assertEqual(paths.lstm.name, "lstm_all_vix.keras")
        self.assertEqual(paths.x_scaler.name, "lstm_x_scaler.joblib")
        self.assertEqual(paths.delta_scaler.name, "lstm_delta_scaler.joblib")
        self.assertEqual(paths.manifest.name, "manifest.json")


class MetricReproductionTests(unittest.TestCase):
    def test_linear_metrics_must_match_reference_to_one_e_minus_nine(self):
        reference = {"Validation_RMSE": 1.0715801428452516}
        verify_metric_reproduction(
            "Linear Regression",
            {"Validation_RMSE": reference["Validation_RMSE"] + 5e-10},
            reference,
        )

        with self.assertRaisesRegex(
            ArtifactGenerationError,
            "Linear Regression Validation_RMSE",
        ):
            verify_metric_reproduction(
                "Linear Regression",
                {"Validation_RMSE": reference["Validation_RMSE"] + 2e-9},
                reference,
            )

    def test_lstm_metrics_use_documented_tensorflow_tolerance(self):
        reference = {
            "Validation_MAE": 0.43673560649150206,
            "Validation_RMSE": 0.8658823756560654,
            "Validation_R2": 0.9220875962495196,
        }
        verify_metric_reproduction(
            "LSTM + BSM",
            {
                "Validation_MAE": reference["Validation_MAE"] + 0.04,
                "Validation_RMSE": reference["Validation_RMSE"] + 0.04,
                "Validation_R2": reference["Validation_R2"] - 0.009,
            },
            reference,
        )

        with self.assertRaisesRegex(
            ArtifactGenerationError,
            r"LSTM \+ BSM Validation_RMSE",
        ):
            verify_metric_reproduction(
                "LSTM + BSM",
                {"Validation_RMSE": reference["Validation_RMSE"] + 0.051},
                {"Validation_RMSE": reference["Validation_RMSE"]},
            )

    def test_missing_or_non_finite_metric_is_rejected(self):
        with self.assertRaises(ArtifactGenerationError):
            verify_metric_reproduction(
                "Linear Regression",
                {},
                {"Test_R2": 0.8},
            )
        with self.assertRaises(ArtifactGenerationError):
            verify_metric_reproduction(
                "LSTM + BSM",
                {"Test_R2": float("nan")},
                {"Test_R2": 0.9},
            )

    def test_reference_loader_selects_all_vix_rows_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text(
                "ModelFamily,Variant,Validation_MAE,Validation_RMSE,Validation_R2,"
                "Test_MAE,Test_RMSE,Test_R2\n"
                "Linear Regression,All VIX Features,1,2,0.8,3,4,0.7\n"
                "LSTM + BSM,All VIX Features,0.5,1,0.9,0.6,1.2,0.85\n",
                encoding="utf-8",
            )

            result = load_reference_metrics(path)

        self.assertEqual(set(result), {"Linear Regression", "LSTM + BSM"})
        self.assertEqual(result["Linear Regression"]["Test_RMSE"], 4.0)
        self.assertEqual(result["LSTM + BSM"]["Validation_R2"], 0.9)


class ArtifactHashTests(unittest.TestCase):
    def test_manifest_hashes_verify_exact_binary_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ArtifactPaths.from_directory(root)
            payloads = {
                paths.linear: b"linear",
                paths.lstm: b"lstm",
                paths.x_scaler: b"x-scaler",
                paths.delta_scaler: b"delta-scaler",
            }
            for path, payload in payloads.items():
                path.write_bytes(payload)
            manifest = {
                "hashes": {
                    "linear_all_vix.joblib": "7f2fe580edb35154041fa3d4b41dd6d3adaef0c85d2ff6309f1d4b520eeecda3",
                    "lstm_all_vix.keras": "9247369ab1f2ec3b6ce890f44f1837e9bb1925f977bd3641bf9bbefb3a9bace3",
                    "lstm_x_scaler.joblib": "2b978b3132e4529b56e80e39cbe4230f04e6e773ff0237138d7c8a6ca3d7d138",
                    "lstm_delta_scaler.joblib": "cb314780da5b6565d98395dcfad1980e1a07936536c59664574c95e9d2a1d646",
                }
            }

            verify_artifact_hashes(paths, manifest)
            paths.linear.write_bytes(b"changed")

            with self.assertRaisesRegex(ArtifactGenerationError, "hash mismatch"):
                verify_artifact_hashes(paths, manifest)

    def test_missing_hash_entry_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = ArtifactPaths.from_directory(directory)
            for path in paths.binary_paths:
                path.write_bytes(b"content")

            with self.assertRaises(ArtifactGenerationError):
                verify_artifact_hashes(paths, {"hashes": {}})


class ArtifactPermissionTests(unittest.TestCase):
    def test_windows_publication_resets_acl_on_all_five_files(self):
        paths = ArtifactPaths.from_directory(Path("artifacts"))
        completed = type("Completed", (), {"returncode": 0, "stderr": ""})()

        with (
            patch(
                "Week7.model_deployment.artifact_training.platform.system",
                return_value="Windows",
            ),
            patch(
                "Week7.model_deployment.artifact_training.subprocess.run",
                return_value=completed,
            ) as runner,
        ):
            reset_inherited_artifact_permissions(paths)

        expected_paths = [*paths.binary_paths, paths.manifest]
        self.assertEqual(runner.call_count, 5)
        for call, expected_path in zip(runner.call_args_list, expected_paths):
            self.assertEqual(call.args[0], ["icacls", str(expected_path), "/reset"])
            self.assertTrue(call.kwargs["capture_output"])
            self.assertTrue(call.kwargs["text"])

    def test_failed_windows_acl_reset_is_rejected(self):
        paths = ArtifactPaths.from_directory(Path("artifacts"))
        completed = type("Completed", (), {"returncode": 1, "stderr": "denied"})()

        with (
            patch(
                "Week7.model_deployment.artifact_training.platform.system",
                return_value="Windows",
            ),
            patch(
                "Week7.model_deployment.artifact_training.subprocess.run",
                return_value=completed,
            ),
            self.assertRaisesRegex(ArtifactGenerationError, "permissions"),
        ):
            reset_inherited_artifact_permissions(paths)


class PublishedArtifactIntegrationTests(unittest.TestCase):
    def test_published_artifacts_load_with_expected_shapes(self):
        import joblib
        from tensorflow.keras.models import load_model

        paths = ArtifactPaths.from_directory(DEFAULT_OUTPUT_DIR)
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))

        verify_artifact_hashes(paths, manifest)
        linear = joblib.load(paths.linear)
        lstm = load_model(paths.lstm, compile=False)
        x_scaler = joblib.load(paths.x_scaler)
        delta_scaler = joblib.load(paths.delta_scaler)

        self.assertEqual(linear.n_features_in_, 10)
        self.assertEqual(x_scaler.n_features_in_, 13)
        self.assertEqual(lstm.input_shape, (None, 20, 13))
        self.assertEqual(delta_scaler.n_features_in_, 1)
        self.assertEqual(manifest["variant"], "All VIX Features")
        self.assertEqual(manifest["window_size"], 20)


if __name__ == "__main__":
    unittest.main()
