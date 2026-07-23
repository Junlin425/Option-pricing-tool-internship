"""Test Alpha Vantage daily data without storing credentials in source."""

import os
from pathlib import Path

from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("Missing ALPHA_VANTAGE_API_KEY environment variable.")

ts = TimeSeries(
    key=api_key,
    output_format="pandas",
)

print("Testing Alpha Vantage API...")

data, meta = ts.get_daily(symbol="JPM")

print(data.head())
print(f"Rows downloaded: {len(data)}")
print("Alpha Vantage test succeeded.")
