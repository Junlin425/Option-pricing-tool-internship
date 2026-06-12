import pandas as pd

# Load datasets

jpm = pd.read_csv("Week1/data/raw/jpm_raw.csv")
vix = pd.read_csv("Week1/data/raw/vix_raw.csv")
treasury = pd.read_csv("Week1/data/raw/treasury_raw.csv")

# Dataset Information

print("=" * 50)
print("JPM DATA")
print("=" * 50)

print(jpm.head())
print()
print(jpm.info())
print()
print(jpm.shape)

print("\n")

print("=" * 50)
print("VIX DATA")
print("=" * 50)

print(vix.head())
print()
print(vix.info())
print()
print(vix.shape)

print("\n")

print("=" * 50)
print("TREASURY DATA")
print("=" * 50)

print(treasury.head())
print()
print(treasury.info())
print()
print(treasury.shape)

# Missing Values

print("\n")
print("=" * 50)
print("MISSING VALUES")
print("=" * 50)

print("\nJPM")
print(jpm.isnull().sum())

print("\nVIX")
print(vix.isnull().sum())

print("\nTREASURY")
print(treasury.isnull().sum())

# Date Range Check

print("\n")
print("=" * 50)
print("DATE RANGE")
print("=" * 50)

print("\nJPM")
print(jpm.iloc[0, 0])
print(jpm.iloc[-1, 0])

print("\nVIX")
print(vix.iloc[0, 0])
print(vix.iloc[-1, 0])

print("\nTREASURY")
print(treasury.iloc[0, 0])
print(treasury.iloc[-1, 0])