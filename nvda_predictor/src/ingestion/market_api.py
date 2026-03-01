import logging
import os
import sys

import pandas as pd
import yfinance as yf

# ingestion/ is one level below src/ — add src/ so config.py is importable
_src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../nvda_predictor/src/
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)
from config import TICKER, HISTORY_PERIOD, TIME_INTERVAL

logger = logging.getLogger(__name__)


def fetch_market_data(
    ticker: str = TICKER,
    period: str = HISTORY_PERIOD,
    interval: str = TIME_INTERVAL,
) -> pd.DataFrame:
    """
    Fetches historical market data (prices, volume) and calculates basic metrics.
    """
    logger.info("Fetching market data for %s (Period: %s, Interval: %s)...", ticker, period, interval)

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)

    if df.empty:
        raise ValueError("No data found. Check the ticker or the connection.")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    df["Returns"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(
            "DataFrame is empty after feature engineering. "
            "The fetched period may be too short to compute returns."
        )

    # Normalize index to UTC — handle both tz-naive and tz-aware index
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
    else:
        df.index = pd.to_datetime(df.index).tz_convert("UTC")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        market_df = fetch_market_data()
        logger.info("Success. Total records: %d", len(market_df))
        print(market_df.head())
    except Exception as e:
        logger.error("Error: %s", e)