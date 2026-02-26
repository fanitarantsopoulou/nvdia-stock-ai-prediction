import pandas as pd
import os
import sys

# Append root path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingestion.market_api import fetch_market_data
from src.ingestion.news_api import fetch_news_data
from src.processing.sentiment import compute_sentiment

def merge_data() -> pd.DataFrame:
    print("[*] Starting data merge process...")
    
    # 1. Fetch Market Data
    market_df = fetch_market_data()
    
    # 2. Fetch News Data
    news_df = fetch_news_data()
    
    if news_df.empty:
        print("[!] No news data to merge. Returning market data with neutral sentiment.")
        market_df['sentiment_score'] = 0.0
        return market_df
        
    # 3. Compute Sentiment
    news_df = compute_sentiment(news_df)
    
    # 4. Aggregate Sentiment by Hour
    # Align the random news timestamps to the start of the hour to match market data
    news_df.index = news_df.index.floor('h')
    
    # Calculate the average sentiment if multiple articles exist in the same hour
    hourly_sentiment = news_df.groupby(news_df.index)['sentiment_score'].mean().to_frame()
    
    # 5. Join Datasets
    # Left join ensures we keep all market data rows. 
    # Hours with no news will get NaN, which we fill with 0.0 (neutral).
    merged_df = market_df.join(hourly_sentiment, how='left')
    merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(0.0)
    
    return merged_df

if __name__ == "__main__":
    try:
        final_df = merge_data()
        print("\n[*] Merge successful. Final Dataset Preview:")
        # Print the last 5 rows to see the most recent data
        print(final_df.tail())
    except Exception as e:
        print(f"[!] Error during merge: {e}")