import yfinance as yf
import pandas as pd
import os
import sys

# Add the root directory to the path for imports to work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import TICKER, HISTORY_PERIOD, TIME_INTERVAL

def fetch_market_data(ticker: str = TICKER, period: str = HISTORY_PERIOD, interval: str = TIME_INTERVAL) -> pd.DataFrame:
    """
    Fetches historical market data (prices, volume) and calculates basic metrics.
    """
    print(f"[*] Fetching market data for {ticker} (Period: {period}, Interval: {interval})...")
    
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    
    if df.empty:
        raise ValueError(f"No data found. Check the ticker or the connection.")
        
    # Keep only the essential features
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Feature Engineering: Calculate percentage change (Returns)
    df['Returns'] = df['Close'].pct_change()
    
    # Remove the first row which will have NaN in Returns
    df.dropna(inplace=True)
    
    # Normalize the index timezone to UTC for proper join with news
    df.index = pd.to_datetime(df.index).tz_convert('UTC')
    
    return df

if __name__ == "__main__":
    # Test execution
    try:
        market_df = fetch_market_data()
        print(f"[*] Success. Total records: {len(market_df)}")
        print(market_df.head())
    except Exception as e:
        print(f"[!] Error: {e}")