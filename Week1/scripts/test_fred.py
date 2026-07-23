"""Download DGS10 data without storing the FRED credential in source."""

import os
from pathlib import Path

from dotenv import load_dotenv
from fredapi import Fred


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

api_key = os.getenv("FRED_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("Missing FRED_API_KEY environment variable.")

fred = Fred(api_key=api_key)
output_path = PROJECT_ROOT / "Week1" / "data" / "raw" / "treasury_raw.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

print("Downloading Treasury Yield...")

dgs10 = fred.get_series(
    "DGS10",
    observation_start="2018-01-01",
    observation_end="2024-12-31",
)

dgs10.to_csv(output_path)

print(f"Rows downloaded: {len(dgs10)}")
print("FRED test succeeded.")
