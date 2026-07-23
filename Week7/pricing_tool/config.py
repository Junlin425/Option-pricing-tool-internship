"""Safe API credential loading for the Week 7 pricing tool."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is unavailable."""


@dataclass(frozen=True)
class ApiKeys:
    """API credentials required by the daily market-data providers."""

    alpha_vantage: str
    fred: str


def load_api_keys(env_file: Path | None = None) -> ApiKeys:
    """Load required keys without printing or embedding their values."""

    load_dotenv(env_file or DEFAULT_ENV_FILE, override=False)

    values = {
        "ALPHA_VANTAGE_API_KEY": os.getenv("ALPHA_VANTAGE_API_KEY", "").strip(),
        "FRED_API_KEY": os.getenv("FRED_API_KEY", "").strip(),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ConfigurationError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )

    return ApiKeys(
        alpha_vantage=values["ALPHA_VANTAGE_API_KEY"],
        fred=values["FRED_API_KEY"],
    )
