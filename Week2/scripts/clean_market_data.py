import pandas as pd

# JPM
jpm = pd.read_csv(
    "Week1/data/raw/jpm_raw.csv",
    skiprows=3,
    header=None
)

jpm.columns = [
    "Date",
    "Adj_Close",
    "Close",
    "High",
    "Low",
    "Open",
    "Volume"
]

# 转换日期时间格式
jpm["Date"] = pd.to_datetime(jpm["Date"])

# 转换数值列为数值类型
for col in jpm.columns[1:]:
    jpm[col] = pd.to_numeric(jpm[col])

print("JPM")
print(jpm.head())
print(jpm.info())

# VIX
vix = pd.read_csv(
    "Week1/data/raw/vix_raw.csv",
    skiprows=3,
    header=None
)

vix.columns = [
    "Date",
    "VIX_Adj_Close",
    "VIX_Close",
    "VIX_High",
    "VIX_Low",
    "VIX_Open",
    "VIX_Volume"
]

vix["Date"] = pd.to_datetime(vix["Date"])

for col in vix.columns[1:]:
    vix[col] = pd.to_numeric(vix[col])

print("\nVIX")
print(vix.head())
print(vix.info())


# Treasury

treasury = pd.read_csv(
    "Week1/data/raw/treasury_raw.csv"
)

treasury.columns = [
    "Date",
    "Treasury_Yield"
]

treasury["Date"] = pd.to_datetime(
    treasury["Date"]
)

print("\nTreasury")
print(treasury.head())
print(treasury.info())

# 保存清洗后的数据
jpm.to_csv(
    "Week2/data/jpm_clean.csv",
    index=False
)

vix.to_csv(
    "Week2/data/vix_clean.csv",
    index=False
)

treasury.to_csv(
    "Week2/data/treasury_clean.csv",
    index=False
)