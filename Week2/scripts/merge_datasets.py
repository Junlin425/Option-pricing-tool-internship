import pandas as pd

# Load clean datasets

jpm = pd.read_csv(
    "Week2/data/jpm_clean.csv",
    parse_dates=["Date"]
)

vix = pd.read_csv(
    "Week2/data/vix_clean.csv",
    parse_dates=["Date"]
)

treasury = pd.read_csv(
    "Week2/data/treasury_clean.csv",
    parse_dates=["Date"]
)

# Merge JPM + VIX

master = pd.merge(
    jpm,
    vix,
    on="Date",
    how="inner"
)

# Merge Treasury

master = pd.merge(
    master,
    treasury,
    on="Date",
    how="left"
)

# Sort by date

master = master.sort_values(
    "Date"
).reset_index(drop=True)

# Check result

print("=" * 50)
print("MASTER DATASET")
print("=" * 50)

print(master.head())

print("\n")

print(master.info())

print("\nShape:")
print(master.shape)

print("\nMissing Values:")
print(master.isnull().sum())

# Save

master.to_csv(
    "Week2/data/merged_dataset.csv",
    index=False
)

print("\nSaved:")
print("Week2/data/merged_dataset.csv")