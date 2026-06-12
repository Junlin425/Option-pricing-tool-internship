import yfinance as yf
import pandas as pd

ticker = yf.Ticker("JPM")

dividend = ticker.dividends

dividend = dividend.reset_index()

dividend.columns = [
    "Date",
    "Dividend"
]

dividend["Date"] = pd.to_datetime(
    dividend["Date"]
).dt.tz_localize(None)

dividend = dividend[
    (dividend["Date"] >= "2018-01-01")
    &
    (dividend["Date"] <= "2024-12-31")
]

print(dividend.head())

print(dividend.shape)


dividend.to_csv(
    "Week2/data/jpm_dividend.csv",
    index=False
)