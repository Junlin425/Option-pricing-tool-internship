from fredapi import Fred
from pathlib import Path

fred = Fred(api_key= "fd4031deabe8b74ef1b9cf676819c5f6" )

Path("../data/raw").mkdir(parents=True, exist_ok=True)

print("Downloading Treasury Yield...")

# 10-Year Treasury Yield
dgs10 = fred.get_series(
    "DGS10",
    observation_start="2018-01-01",
    observation_end="2024-12-31"
)

dgs10.to_csv("../Week1/data/raw/treasury_raw.csv")

print(f"Rows downloaded: {len(dgs10)}")

print("FRED test succeeded.")