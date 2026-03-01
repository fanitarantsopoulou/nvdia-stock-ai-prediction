import logging
import os
import sys

# sentiment.py lives at src/processing/ — one level up reaches src/
_src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../nvda_predictor/src/
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"

# Lazy-loaded pipeline — populated on first call to compute_sentiment().
# Keeping it at module scope means it is still only loaded once per process,
# but the load is deferred until the function is actually called, so importing
# this module never triggers a model download or crashes the server startup.
_sentiment_analyzer = None


def _get_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        logger.info("Loading FinBERT pipeline (model: %s, device: %s)...", MODEL_NAME, device)
        _sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            device=device,
        )
        logger.info("FinBERT pipeline ready.")
    return _sentiment_analyzer


def _normalize_score(res: dict) -> float:
    """Convert a FinBERT result dict to a signed float in [-1.0, 1.0]."""
    label = res["label"]
    score = float(res["score"])
    if label == "positive":
        return score
    if label == "negative":
        return -score
    return 0.0  # neutral


def compute_sentiment(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Runs FinBERT inference on a DataFrame containing news articles.
    Maps output to a continuous sentiment score [-1.0, 1.0].
    Returns a new DataFrame — the input is never mutated.
    """
    if df.empty or text_col not in df.columns:
        logger.warning("DataFrame is empty or missing column '%s'. Skipping inference.", text_col)
        return df

    df = df.copy()  # never mutate the caller's DataFrame

    logger.info("Running sentiment inference on %d articles...", len(df))

    texts   = df[text_col].astype(str).tolist()
    analyzer = _get_analyzer()

    # batch_size=16 prevents OOM on CPU for large article lists
    results = analyzer(texts, padding=True, truncation=True, max_length=512, batch_size=16)

    df["sentiment_label"] = [res["label"] for res in results]
    df["sentiment_score"]  = [_normalize_score(res) for res in results]

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    mock_data = {
        "text": [
            "NVIDIA revenue misses expectations, shares plunge 5% in after-hours trading.",
            "TSMC announces major breakthrough in 2nm chip production, boosting NVDA prospects.",
            "NVIDIA holds annual developer conference today.",
        ]
    }
    test_df = pd.DataFrame(mock_data)
    result_df = compute_sentiment(test_df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(result_df[["text", "sentiment_label", "sentiment_score"]])