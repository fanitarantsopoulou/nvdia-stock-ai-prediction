import pandas as pd
from transformers import pipeline
import os
import sys

# Append root path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) )

# Initialize the pipeline globally so the model loads only once in memory
MODEL_NAME = "ProsusAI/finbert"
print(f"[*] Bootstrapping NLP pipeline with model: {MODEL_NAME}...")

# Use GPU if available, otherwise fallback to CPU
device = 0 if __import__('torch').cuda.is_available() else -1
sentiment_analyzer = pipeline("sentiment-analysis", model=MODEL_NAME, device=device)

def compute_sentiment(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
	"""
	Runs FinBERT inference on a DataFrame containing news articles.
	Maps output to a continuous sentiment score [-1.0 to 1.0].
	"""
	if df.empty or text_col not in df.columns:
		print("[!] DataFrame is empty or missing the text column. Skipping inference.")
		return df
        
	print(f"[*] Running sentiment inference on {len(df)} articles...")
    
	# Extract texts as a list for batch processing
	texts = df[text_col].astype(str).tolist()
    
	# Run inference with truncation (BERT handles max 512 tokens)
	results = sentiment_analyzer(texts, padding=True, truncation=True, max_length=512)
    
	# Extract discrete labels
	df['sentiment_label'] = [res['label'] for res in results]
    
	# Convert probability scores into a signed float for the ML model
	# positive -> +score, negative -> -score, neutral -> 0.0
	def normalize_score(res: dict) -> float:
		if res['label'] == 'positive':
			return float(res['score'])
		elif res['label'] == 'negative':
			return -float(res['score'])
		else:
			return 0.0
            
	df['sentiment_score'] = [normalize_score(res) for res in results]
    
	return df

if __name__ == "__main__":
	# Unit test with mock market headlines
	mock_data = {
		'text': [
			"NVIDIA revenue misses expectations, shares plunge 5% in after-hours trading.",
			"TSMC announces major breakthrough in 2nm chip production, boosting NVDA prospects.",
			"NVIDIA holds annual developer conference today."
		]
	}
	test_df = pd.DataFrame(mock_data)
    
	result_df = compute_sentiment(test_df)
    
	pd.set_option('display.max_columns', None)
	pd.set_option('display.width', 1000)
	print("\n[*] Inference Results:")
	print(result_df[['text', 'sentiment_label', 'sentiment_score']])
