import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Week7.pricing_tool.config import ConfigurationError, load_api_keys


class ApiKeyConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.missing_env_file = Path(self.temporary_directory.name) / "missing.env"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_load_api_keys_reads_environment_without_exposing_values(self):
        fake_values = {
            "ALPHA_VANTAGE_API_KEY": "fake-alpha-for-test",
            "FRED_API_KEY": "fake-fred-for-test",
        }

        with patch.dict(os.environ, fake_values, clear=True):
            keys = load_api_keys(env_file=self.missing_env_file)

        self.assertEqual(keys.alpha_vantage, fake_values["ALPHA_VANTAGE_API_KEY"])
        self.assertEqual(keys.fred, fake_values["FRED_API_KEY"])

    def test_missing_key_error_names_variable_but_not_present_secret(self):
        fake_alpha_key = "fake-alpha-for-test"

        with patch.dict(
            os.environ,
            {"ALPHA_VANTAGE_API_KEY": fake_alpha_key},
            clear=True,
        ):
            with self.assertRaises(ConfigurationError) as caught:
                load_api_keys(env_file=self.missing_env_file)

        message = str(caught.exception)
        self.assertIn("FRED_API_KEY", message)
        self.assertNotIn(fake_alpha_key, message)


if __name__ == "__main__":
    unittest.main()
