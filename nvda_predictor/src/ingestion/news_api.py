import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Add the root directory to the path for imports to work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nvda_predictor.src.config import NEWS_API_KEY, TICKER

def fetch_news_data(query: str = "NVIDIA OR NVDA OR TSMC OR Blackwell", days_back: int = 7) -> pd.DataFrame:
    """
    Fetches recent news articles related to NVIDIA and its competitors.
    """
    if not NEWS_API_KEY:
        raise ValueError("[!] The NEWS_API_KEY is missing from the .env file.")

    print(f"[*] Fetching news for: {query}...")

    # Calculate the date from which we want news
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWS_API_KEY,
        "pageSize": 100 # Maximum limit in the free tier
    }

    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise ConnectionError(f"API Call Failed: {response.json().get('message')}")

    articles = response.json().get("articles", [])
    
    if not articles:
        print("[!] No articles found.")
        return pd.DataFrame()

    # Convert to DataFrame and keep only the essential columns
    df = pd.DataFrame(articles)
    df = df[['publishedAt', 'title', 'description', 'source']]
    
    # Extract the source name (e.g., 'Reuters') from the dictionary
    df['source'] = df['source'].apply(lambda x: x.get('name') if isinstance(x, dict) else x)
    
    # Combine title and description to provide the full text to the NLP model
    df['text'] = df['title'] + ". " + df['description'].fillna("")
    
    # Convert the timestamp to a datetime object and set it as the index
    df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.tz_convert('UTC')
    df.set_index('publishedAt', inplace=True)
    
    # Keep only the final text column and the source
    df = df[['text', 'source']]
    
    return df

if __name__ == "__main__":
    # Test execution (Requires API Key in .env)
    try:
        news_df = fetch_news_data()
        print(f"[*] Success. Total articles: {len(news_df)}")
        print(news_df.head())
    except Exception as e:
        print(f"[!] Error: {e}")