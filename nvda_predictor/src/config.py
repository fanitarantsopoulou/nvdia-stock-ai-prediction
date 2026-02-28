import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration for the stock price prediction model
TICKER = "NVDA"
HISTORY_PERIOD = "1mo"  # Fetch data for 1 month for training/testing
TIME_INTERVAL = "1h"    # Granularity per hour (matches the news flow)

# API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")