from alpha_vantage.timeseries import TimeSeries

ts = TimeSeries(
    key="ZYZTLWRSBGLU6UJV",
    output_format="pandas"
)

print("Testing Alpha Vantage API...")

data, meta = ts.get_daily(symbol="JPM")

print(data.head())

print("Alpha Vantage test succeeded.")