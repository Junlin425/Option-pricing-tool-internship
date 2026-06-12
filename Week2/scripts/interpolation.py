import pandas as pd

# Load dataset

df = pd.read_csv(
    "Week2/data/merged_dataset.csv",
    parse_dates=["Date"]
)

print("=" * 50)
print("Before Interpolation")
print("=" * 50)

print(df.isnull().sum())

# Linear Interpolation

df["Treasury_Yield"] = df[
    "Treasury_Yield"
].interpolate(
    method="linear"
)

# Check Again

print("\n")
print("=" * 50)
print("After Interpolation")
print("=" * 50)

print(df.isnull().sum())

# Save

df.to_csv(
    "Week2/data/Cleaned_structured_dataset.csv",
    index=False
)

print("\nSaved:")
print("Week2/data/Cleaned_structured_dataset.csv")