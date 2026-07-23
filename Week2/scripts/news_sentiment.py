"""Download recent Alpha Vantage news sentiment using an environment key."""

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("Missing ALPHA_VANTAGE_API_KEY environment variable.")

print("Downloading news sentiment...")

response = requests.get(
    "https://www.alphavantage.co/query",
    params={
        "function": "NEWS_SENTIMENT",
        "tickers": "JPM",
        "limit": 1000,
        "apikey": api_key,
    },
    timeout=30,
)
response.raise_for_status()
payload = response.json()

if "feed" not in payload:
    message = payload.get("Information") or payload.get("Note") or "Unexpected API response."
    raise RuntimeError(message)

records = []
for article in payload["feed"]:
    date = pd.to_datetime(article["time_published"][:8], format="%Y%m%d")
    records.append([date, article["overall_sentiment_score"]])

sentiment = pd.DataFrame(records, columns=["Date", "SentimentScore"])
sentiment["SentimentScore"] = (sentiment["SentimentScore"] + 1) / 2
sentiment = sentiment.groupby("Date")["SentimentScore"].mean().reset_index()

output_path = PROJECT_ROOT / "Week2" / "data" / "news_sentiment.csv"
sentiment.to_csv(output_path, index=False)

print(sentiment.head())
print(f"Rows downloaded: {len(sentiment)}")
print("news_sentiment.csv saved")
