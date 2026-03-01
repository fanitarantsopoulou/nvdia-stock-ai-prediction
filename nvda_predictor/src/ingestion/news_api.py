import logging
import os
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

# ingestion/ is one level below src/ — add src/ so config.py is importable
_src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../nvda_predictor/src/
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)
from config import NEWS_API_KEY, TICKER

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when the News API returns HTTP 429 (Too Many Requests)."""


def fetch_news_data(
    query: str = "NVIDIA OR NVDA OR TSMC OR Blackwell",
    days_back: int = 7,
) -> pd.DataFrame:
    """
    Fetches recent news articles related to NVIDIA and its competitors.
    Returns an empty DataFrame (not an exception) when no articles are found.
    Raises RateLimitError on HTTP 429 so callers can apply a neutral-sentiment
    fallback without treating the condition as a generic failure.
    """
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY is missing from the .env file.")

    logger.info("Fetching news for: %s...", query)

    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "publishedAt",
        "language": "en",
        "apiKey":   NEWS_API_KEY,
        "pageSize": 100,
    }

    response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)

    if response.status_code == 429:
        raise RateLimitError("News API rate limit reached (HTTP 429). Using neutral sentiment fallback.")

    if response.status_code != 200:
        # Avoid echoing the raw API message — it may contain the key in 401 responses.
        raise ConnectionError(
            f"News API call failed with HTTP {response.status_code}. "
            "Check NEWS_API_KEY validity and account limits."
        )

    articles = response.json().get("articles", [])

    if not articles:
        logger.warning("No articles found for query: %s", query)
        return pd.DataFrame()

    df = pd.DataFrame(articles)

    # Ensure all expected columns exist before selecting them
    for col in ("publishedAt", "title", "description", "source"):
        if col not in df.columns:
            df[col] = None

    df = df[["publishedAt", "title", "description", "source"]].copy()

    df["source"] = df["source"].apply(lambda x: x.get("name") if isinstance(x, dict) else x)
    df["text"]   = df["title"].fillna("") + ". " + df["description"].fillna("")

    # Normalize timestamps to UTC — guard against tz-naive values
    parsed = pd.to_datetime(df["publishedAt"], errors="coerce", utc=False)
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize("UTC")
    else:
        parsed = parsed.dt.tz_convert("UTC")
    df["publishedAt"] = parsed

    # Drop rows where the timestamp could not be parsed
    df.dropna(subset=["publishedAt"], inplace=True)

    df.set_index("publishedAt", inplace=True)
    df = df[["text", "source"]]

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        news_df = fetch_news_data()
        logger.info("Success. Total articles: %d", len(news_df))
        print(news_df.head())
    except Exception as e:
        logger.error("Error: %s", e)