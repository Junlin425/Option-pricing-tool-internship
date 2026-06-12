import pandas as pd
import numpy as np

# Load dataset

df = pd.read_csv(
    "Week2/data/Cleaned_structured_dataset.csv",
    parse_dates=["Date"]
)

# Daily Log Return

df["Return"] = np.log(
    df["Close"] /
    df["Close"].shift(1)
)

# Check Result

print(df[
    ["Date", "Close", "Return"]
].head(10))

# Missing Values

print("\nMissing Values:")

print(
    df["Return"].isnull().sum()
)

# Save

df.to_csv(
    "Week2/data/Cleaned_structured_dataset.csv",
    index=False
)

print("\nReturn feature saved.")

# 20-Day Rolling Volatility

df["RollingVol20"] = (
    df["Return"]
    .rolling(window=20)
    .std()
)

print(
    df[
        ["Date",
         "Return",
         "RollingVol20"]
    ].head(25)
)

print("\nMissing Values:")

print(
    df["RollingVol20"]
    .isnull()
    .sum()
)

# Save

df.to_csv(
    "Week2/data/Cleaned_structured_dataset.csv",
    index=False
)

# Dividend Growth Feature

# Load dividend data
dividend = pd.read_csv(
    "Week2/data/jpm_dividend.csv",
    parse_dates=["Date"]
)

# Calculate dividend growth
dividend["DividendGrowth"] = (
    dividend["Dividend"]
    .pct_change()
)

print("\nDividend Dataset:")

print(dividend.head())

print("\nMissing Values:")

print(dividend["DividendGrowth"].isnull().sum())

df["DividendGrowth"] = (
    df["DividendGrowth"]
    .ffill()
    .fillna(0)
)

print(
    df[
        [
            "Date",
            "Dividend",
            "DividendGrowth"
        ]
    ].head(30)
)

# Save

df.to_csv(
    "Week2/data/Cleaned_structured_dataset.csv",
    index=False
)

# Interest Rate Momentum

print("\nCalculating Interest Rate Momentum...")

df["RateMomentum"] = (
    df["Treasury_Yield"]
    -
    df["Treasury_Yield"].shift(5)
)

print(
    df[
        [
            "Date",
            "Treasury_Yield",
            "RateMomentum"
        ]
    ].head(15)
)

print("\nMissing Values:")

print(
    df["RateMomentum"]
    .isnull()
    .sum()
)

# Save

df.to_csv(
    "Week2/data/Cleaned_structured_dataset.csv",
    index=False
)

print("\nRateMomentum feature saved.")

# ==========================
# VIX Return
# ==========================

df["VIX_Return"] = np.log(
    df["VIX_Close"] /
    df["VIX_Close"].shift(1)
)

# Rolling VIX-JPM Correlation
# VIX Return

df["VIX_Return"] = np.log(
    df["VIX_Close"] /
    df["VIX_Close"].shift(1)
)

df["VIX_JPM_Corr"] = (
    df["Return"]
    .rolling(window=20)
    .corr(df["VIX_Return"])
)

print(
    df[
        [
            "Date",
            "Return",
            "VIX_Return",
            "VIX_JPM_Corr"
        ]
    ].head(30)
)

print("\nMissing Values:")

print(
    df["VIX_JPM_Corr"]
    .isnull()
    .sum()
)

# Save

df.to_csv(
    "Week2/data/Cleaned_structured_dataset.csv",
    index=False
)

print("\nVIX_JPM_Corr feature saved.")

# Moving Average 5-day and 20-days Features

print("\nCalculating Moving Averages...")

# 5-Day Moving Average

df["MA5"] = (
    df["Close"]
    .rolling(window=5)
    .mean()
)

# 20-Day Moving Average

df["MA20"] = (
    df["Close"]
    .rolling(window=20)
    .mean()
)

print(
    df[
        [
            "Date",
            "Close",
            "MA5",
            "MA20"
        ]
    ].head(25)
)

print("\nMissing Values:")

print(
    "MA5:",
    df["MA5"].isnull().sum()
)

print(
    "MA20:",
    df["MA20"].isnull().sum()
)

# Save

df.to_csv(
    "Week2/data/Cleaned_structured_dataset.csv",
    index=False
)

print("\nMA features saved.")

# Volume Change

print("\nCalculating Volume Change...")

df["VolumeChange"] = (
    df["Volume"]
    .pct_change()
)

print(
    df[
        [
            "Date",
            "Volume",
            "VolumeChange"
        ]
    ].head(15)
)

print("\nMissing Values:")

print(
    df["VolumeChange"]
    .isnull()
    .sum()
)

# Save

df.to_csv(
    "Week2/data/Cleaned_structured_dataset.csv",
    index=False
)

print("\nVolumeChange feature saved.")

print("\n=========================")
print("Missing Values Summary")
print("=========================")

print(
    df.isnull().sum()
)

# Final Missing Value Treatment

print("\nRows Before Cleaning:")

print(len(df))

# Remove rows containing NaN values

df = df.dropna()

# Reset index

df = df.reset_index(
    drop=True
)

print("\nRows After Cleaning:")

print(len(df))

print("\nRows Removed:")

print(
    len(df.index) - len(df.index)
)

print("\nRemaining Missing Values:")

print(
    df.isnull().sum().sum()
)

# Save Final Feature Dataset

df.to_csv(
    "Week2/data/feature_dataset.csv",
    index=False
)

print(
    "\nfeature_dataset.csv saved."
)