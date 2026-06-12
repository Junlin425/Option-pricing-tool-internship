import requests
import pandas as pd

# Alpha Vantage API

url = (
    "https://www.alphavantage.co/query?"
    "function=NEWS_SENTIMENT"
    "&tickers=JPM"
    "&limit=1000"
    f"&apikey={"6P5GROOG5IH2QLKR"}"
)

print("Downloading news sentiment...")

response = requests.get(url)

data = response.json()

# Extract sentiment

records = []

for article in data["feed"]:

    date = article["time_published"][:8]

    date = pd.to_datetime(
        date,
        format="%Y%m%d"
    )

    score = article["overall_sentiment_score"]

    records.append(
        [date, score]
    )

sentiment = pd.DataFrame(
    records,
    columns=[
        "Date",
        "SentimentScore"
    ]
)

# Convert [-1,1] -> [0,1]

sentiment["SentimentScore"] = (
    sentiment["SentimentScore"] + 1
) / 2

# Daily Average

sentiment = (
    sentiment
    .groupby("Date")
    ["SentimentScore"]
    .mean()
    .reset_index()
)

print(sentiment.head())

print("\nShape:")
print(sentiment.shape)

# Save

sentiment.to_csv(
    "../data/news_sentiment.csv",
    index=False
)

print("\nnews_sentiment.csv saved")