import logging
import os
import sys

# merger.py lives at src/processing/ — one level up reaches src/
_src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../nvda_predictor/src/
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

import pandas as pd

from ingestion.market_api import fetch_market_data
from ingestion.news_api import fetch_news_data
from processing.sentiment import compute_sentiment

logger = logging.getLogger(__name__)


def merge_data() -> pd.DataFrame:
    logger.info("Starting data merge process...")

    # 1. Fetch Market Data
    market_df = fetch_market_data()

    # 2. Fetch News Data
    try:
        news_df = fetch_news_data()
    except Exception as exc:
        logger.warning("News fetch failed (%s). Proceeding with neutral sentiment.", exc)
        market_df["sentiment_score"] = 0.0
        return market_df

    if news_df.empty:
        logger.warning("No news data to merge. Returning market data with neutral sentiment.")
        market_df["sentiment_score"] = 0.0
        return market_df

    # 3. Compute Sentiment
    try:
        news_df = compute_sentiment(news_df)
    except Exception as exc:
        logger.warning("Sentiment computation failed (%s). Falling back to neutral sentiment.", exc)
        market_df["sentiment_score"] = 0.0
        return market_df

    # 4. Aggregate Sentiment by Hour
    # floor() aligns article timestamps to the start of each hour to match
    # the hourly candle index of market_df.
    news_df.index = news_df.index.floor("h")
    hourly_sentiment = news_df.groupby(news_df.index)["sentiment_score"].mean().to_frame()

    # 5. Join Datasets
    # Left join keeps all market rows. Hours with no news get NaN.
    # Fallback: use the mean sentiment of the available news window rather than
    # neutral 0.0, so older candles are not all penalised equally.
    merged_df = market_df.join(hourly_sentiment, how="left")
    sentiment_fallback = float(hourly_sentiment["sentiment_score"].mean()) if not hourly_sentiment.empty else 0.0
    merged_df["sentiment_score"] = merged_df["sentiment_score"].fillna(sentiment_fallback)
    logger.info("Sentiment fallback value applied to unmatched candles: %.4f", sentiment_fallback)

    logger.info(
        "Merge complete. Rows: %d  Sentiment coverage: %.1f%%",
        len(merged_df),
        100 * (merged_df["sentiment_score"] != 0.0).mean(),
    )

    return merged_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        final_df = merge_data()
        logger.info("Merge successful. Final dataset preview:")
        print(final_df.tail())
    except Exception as e:
        logger.error("Error during merge: %s", e)