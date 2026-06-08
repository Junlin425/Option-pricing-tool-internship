import yfinance as yf
from pathlib import Path

# Create output directory
Path("../data/raw").mkdir(parents=True, exist_ok=True)

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# JPM Stock Data

print("Downloading JPM data...")

jpm = yf.download(
    "JPM",
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False
)

jpm.to_csv("../Week1/data/raw/jpm_raw.csv")

print(f"JPM rows: {len(jpm)}")

# VIX Data

print("Downloading VIX data...")

vix = yf.download(
    "^VIX",
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False
)

vix.to_csv("../Week1/data/raw/vix_raw.csv")

print(f"VIX rows: {len(vix)}")

print("Yahoo Finance test succeeded.")