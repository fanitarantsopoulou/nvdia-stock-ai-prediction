import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stock data
# yfinance valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
# Max period for hourly (1h) data is 730 days.
# ---------------------------------------------------------------------------
TICKER         = "NVDA"
HISTORY_PERIOD = "6mo"  # Minimum recommended for meaningful LSTM training
TIME_INTERVAL  = "1h"

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
NEWS_API_KEY: str | None = os.getenv("NEWS_API_KEY")

if not NEWS_API_KEY:
    logger.warning(
        "NEWS_API_KEY is not set. News sentiment will be unavailable. "
        "Add NEWS_API_KEY=<your_key> to your .env file."
    )