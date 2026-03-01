"""
predict.py — NVDA LSTM inference module.

Responsibilities:
  - Fetch and merge price + sentiment data
  - Scale features and run LSTM inference
  - Compute NASDAQ market status (including holidays)
  - Return a consistent JSON-serializable payload

Design notes:
  - Model and scaler are loaded lazily and cached in module-level state
    (_MODEL_CACHE) so that repeated calls (e.g. from a polling loop or
    FastAPI background task) pay the disk-read cost only once.
  - FastAPI's lifespan handler can pre-load the assets and pass them
    explicitly via the `model` / `scaler` keyword arguments, bypassing
    the lazy loader entirely.
  - Every code path returns the same top-level keys so the frontend
    never has to guard against missing fields.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pytz
import torch
import yfinance as yf

# ---------------------------------------------------------------------------
# Path bootstrap — allows running as a standalone script from any cwd
# ---------------------------------------------------------------------------
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))   # .../nvda_predictor/src/model/
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                  # .../nvda_predictor/src/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processing.merger import merge_data  # noqa: E402  (after sys.path patch)
from model.train_lstm import StockPredictorLSTM, SEQ_LENGTH, HIDDEN_SIZE, NUM_LAYERS  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File paths (resolved once at import time)
# ---------------------------------------------------------------------------
_SRC_ROOT    = PROJECT_ROOT                                                          # .../nvda_predictor/src/
_NVDA_ROOT   = os.path.dirname(PROJECT_ROOT)                                         # .../nvda_predictor/
_SCALER_PATH = os.path.join(_NVDA_ROOT, "data", "processed", "scaler_params.npy")
_MODEL_PATH  = os.path.join(_NVDA_ROOT, "data", "processed", "lstm_model.pth")

# ---------------------------------------------------------------------------
# Module-level lazy cache — populated by _load_model_and_scaler()
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICKER          = "NVDA"
NY_TZ           = pytz.timezone("America/New_York")
FEATURE_ORDER   = ["Close", "sentiment_score"]  # Must match training order
NEUTRAL_SENTIMENT = 0.0


# ===========================================================================
# Asset loading
# ===========================================================================

def _load_model_and_scaler() -> Tuple[StockPredictorLSTM, np.ndarray, np.ndarray]:
    """
    Load the LSTM model and MinMax scaler from disk.

    Results are cached in _MODEL_CACHE so that subsequent calls within the
    same process are free (no file I/O, no model init overhead).

    Returns
    -------
    model      : StockPredictorLSTM — eval-mode PyTorch model
    scaler_min : np.ndarray shape (n_features,)
    scaler_max : np.ndarray shape (n_features,)

    Raises
    ------
    FileNotFoundError  — if model or scaler file is missing
    RuntimeError       — if state-dict loading fails
    """
    if _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["scaler_min"], _MODEL_CACHE["scaler_max"]

    # --- Scaler ---
    if not os.path.exists(_SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {_SCALER_PATH}")
    scaler_min, scaler_max = np.load(_SCALER_PATH, allow_pickle=True)

    # --- Model ---
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {_MODEL_PATH}")
    model = StockPredictorLSTM(
        input_size=len(FEATURE_ORDER),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )
    model.load_state_dict(torch.load(_MODEL_PATH, weights_only=True))
    model.eval()

    _MODEL_CACHE["model"]      = model
    _MODEL_CACHE["scaler_min"] = scaler_min
    _MODEL_CACHE["scaler_max"] = scaler_max

    logger.info("Model and scaler loaded and cached from disk.")
    return model, scaler_min, scaler_max


# ===========================================================================
# Market hours logic
# ===========================================================================

def get_market_info() -> Tuple[str, str]:
    """
    Return the current NASDAQ market status and the label for the next
    forecast target time, fully aware of weekends *and* public holidays.

    Strategy
    --------
    1.  Use pandas_market_calendars (mcal) when available — this is the
        only way to correctly handle US market holidays (e.g. Thanksgiving,
        Christmas, Independence Day).
    2.  Fall back to a weekend-only check if mcal is not installed, and
        log a warning so operators know the limitation.

    Returns
    -------
    status        : str  — one of MARKET_OPEN | PRE_MARKET | AFTER_HOURS |
                           CLOSED_WEEKEND | CLOSED_HOLIDAY | CLOSED
    forecast_time : str  — human-readable label, e.g. "MON 09:30"
    """
    now = datetime.now(NY_TZ)

    try:
        import pandas_market_calendars as mcal  # optional dependency
        return _get_market_info_with_calendar(now, mcal)
    except ImportError:
        logger.warning(
            "pandas_market_calendars not installed. "
            "Holiday detection is DISABLED. Install with: "
            "pip install pandas-market-calendars"
        )
        return _get_market_info_fallback(now)


def _get_market_info_with_calendar(now: datetime, mcal) -> Tuple[str, str]:
    """
    Full implementation using pandas_market_calendars.
    Handles weekends, US holidays, pre-market, regular hours, after-hours.
    """
    nasdaq = mcal.get_calendar("NASDAQ")
    today_str = now.strftime("%Y-%m-%d")

    # Schedule for today — empty means weekend or holiday
    schedule_today = nasdaq.schedule(start_date=today_str, end_date=today_str)

    if schedule_today.empty:
        # Weekend or holiday — find the next trading day
        next_open_label = _next_trading_open_label(now, nasdaq)
        status = "CLOSED_WEEKEND" if now.weekday() >= 5 else "CLOSED_HOLIDAY"
        return status, next_open_label

    # Today is a trading day — extract official open/close in NY time
    market_open  = schedule_today.iloc[0]["market_open"].astimezone(NY_TZ)
    market_close = schedule_today.iloc[0]["market_close"].astimezone(NY_TZ)

    # Pre-market window: 04:00 – 09:30
    pre_market_start = now.replace(hour=4, minute=0, second=0, microsecond=0)

    if now < pre_market_start:
        # Fully closed — before pre-market even starts
        return "CLOSED", market_open.strftime("%a %H:%M").upper()

    if pre_market_start <= now < market_open:
        return "PRE_MARKET", market_open.strftime("%a %H:%M").upper()

    if market_open <= now < market_close:
        # Regular session — next forecast is current time + 1 h, capped at close
        next_pred = min(now + timedelta(hours=1), market_close)
        return "MARKET_OPEN", next_pred.strftime("%a %H:%M").upper()

    # After regular close
    after_hours_end = now.replace(hour=20, minute=0, second=0, microsecond=0)
    status = "AFTER_HOURS" if now < after_hours_end else "CLOSED"

    # Next open is the next scheduled trading day
    next_open_label = _next_trading_open_label(now, nasdaq)
    return status, next_open_label


def _next_trading_open_label(from_dt: datetime, nasdaq) -> str:
    """
    Return a formatted label (e.g. 'MON 09:30') for the next NASDAQ open
    after *from_dt*. Searches up to 14 calendar days ahead.
    """
    import pandas as pd  # only needed here, mcal already imported by caller

    search_end = (from_dt + timedelta(days=14)).strftime("%Y-%m-%d")
    # Start from tomorrow to avoid today matching when market is already closed
    search_start = (from_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    future = nasdaq.schedule(start_date=search_start, end_date=search_end)

    if future.empty:
        # Extremely unlikely; return a safe default
        logger.error("Could not find next trading day within 14 days.")
        return "TBD"

    next_open = future.iloc[0]["market_open"].astimezone(NY_TZ)
    return next_open.strftime("%a %H:%M").upper()


def _get_market_info_fallback(now: datetime) -> Tuple[str, str]:
    """
    Weekend-only fallback used when pandas_market_calendars is unavailable.
    Does NOT detect public holidays — use only as a last resort.
    """
    # DST-safe construction: localize a naive datetime rather than .replace()
    def _localize(h: int, m: int) -> datetime:
        naive = datetime(now.year, now.month, now.day, h, m, 0)
        return NY_TZ.localize(naive)

    market_open       = _localize(9, 30)
    market_close      = _localize(16, 0)
    pre_market_start  = _localize(4, 0)
    after_hours_end   = _localize(20, 0)

    if now.weekday() >= 5:
        days_ahead = 7 - now.weekday()  # Sat→2, Sun→1 (both land on Monday)
        next_day   = now + timedelta(days=days_ahead)
        next_open  = NY_TZ.localize(datetime(next_day.year, next_day.month, next_day.day, 9, 30))
        return "CLOSED_WEEKEND", next_open.strftime("%a %H:%M").upper()

    if now < pre_market_start:
        return "CLOSED", market_open.strftime("%a %H:%M").upper()

    if pre_market_start <= now < market_open:
        return "PRE_MARKET", market_open.strftime("%a %H:%M").upper()

    if market_open <= now < market_close:
        next_pred = min(now + timedelta(hours=1), market_close)
        return "MARKET_OPEN", next_pred.strftime("%a %H:%M").upper()

    # After close
    days_ahead = 3 if now.weekday() == 4 else 1  # Friday → Monday
    next_day  = now + timedelta(days=days_ahead)
    next_open = NY_TZ.localize(datetime(next_day.year, next_day.month, next_day.day, 9, 30))
    status    = "AFTER_HOURS" if now < after_hours_end else "CLOSED"
    return status, next_open.strftime("%a %H:%M").upper()


# ===========================================================================
# Live price fetch (with explicit fallback signalling)
# ===========================================================================

def _fetch_live_price(ticker_sym: str, fallback: float) -> Tuple[float, str]:
    """
    Attempt to fetch the latest trade price from yfinance.

    Returns
    -------
    price  : float — live price, or *fallback* if the fetch fails
    source : str   — 'live' or 'stale' (used in the response payload so the
                     frontend can surface a warning to the user)
    """
    try:
        ticker = yf.Ticker(ticker_sym)
        price  = float(ticker.fast_info["last_price"])
        return price, "live"
    except Exception as exc:
        logger.warning(
            "Live price fetch failed for %s (%s). Using last close as fallback.",
            ticker_sym, exc,
        )
        return fallback, "stale"


# ===========================================================================
# Main prediction entry point
# ===========================================================================

def predict_next_hour(
    model: Optional[StockPredictorLSTM] = None,
    scaler_min: Optional[np.ndarray]    = None,
    scaler_max: Optional[np.ndarray]    = None,
) -> dict:
    """
    Run end-to-end NVDA next-hour price prediction.

    Parameters
    ----------
    model, scaler_min, scaler_max
        Pre-loaded assets.  When provided (e.g. passed in by FastAPI's
        lifespan handler) no disk I/O takes place.  When omitted the module
        lazy-loads and caches them automatically.

    Returns
    -------
    dict with keys:
        status          : 'success' | 'error'
        message         : present only when status == 'error'
        ticker          : str
        current_price   : float
        price_source    : 'live' | 'stale'
        predicted_price : float
        difference      : float
        direction       : 'UP' | 'DOWN'
        sentiment_score : float
        history         : list[float]  — last SEQ_LENGTH close prices
        history_times   : list[str]    — matching human-readable labels
        market_status   : str
        forecast_time   : str
    """

    # ------------------------------------------------------------------
    # 1. Load assets (lazy or pre-loaded)
    # ------------------------------------------------------------------
    try:
        if model is None or scaler_min is None or scaler_max is None:
            model, scaler_min, scaler_max = _load_model_and_scaler()
    except FileNotFoundError as exc:
        logger.error("Asset load failed: %s", exc)
        return _error_payload(str(exc))

    # ------------------------------------------------------------------
    # 2. Fetch and merge price + sentiment data
    # ------------------------------------------------------------------
    logger.info("Fetching and merging market data...")
    try:
        df = merge_data()
    except Exception as exc:
        logger.error("merge_data() raised: %s", exc)
        return _error_payload(f"Data pipeline error: {exc}")

    if df is None or df.empty:
        return _error_payload("merge_data() returned no data.")

    if len(df) < SEQ_LENGTH:
        return _error_payload(
            f"Insufficient data: {len(df)} rows available, {SEQ_LENGTH} required."
        )

    # ------------------------------------------------------------------
    # 3. Validate feature columns and order
    # ------------------------------------------------------------------
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        return _error_payload(f"DataFrame is missing required columns: {missing}")

    # Guard against silent wrong predictions caused by column reordering
    assert df[FEATURE_ORDER].columns.tolist() == FEATURE_ORDER, (
        f"Feature order mismatch: {df[FEATURE_ORDER].columns.tolist()} != {FEATURE_ORDER}"
    )

    # ------------------------------------------------------------------
    # 4. Sort and slice
    # ------------------------------------------------------------------
    df = df.sort_index()
    last_n = df.tail(SEQ_LENGTH).copy()

    # Fill any NaN sentiment values with neutral score rather than crashing
    if last_n["sentiment_score"].isna().any():
        nan_count = last_n["sentiment_score"].isna().sum()
        logger.warning(
            "%d NaN sentiment values in the last %d rows — forward-filling, "
            "then back-filling, then defaulting to %.1f.",
            nan_count, SEQ_LENGTH, NEUTRAL_SENTIMENT,
        )
        last_n["sentiment_score"] = (
            last_n["sentiment_score"]
            .ffill()
            .bfill()
            .fillna(NEUTRAL_SENTIMENT)
        )

    # ------------------------------------------------------------------
    # 5. Live price (with graceful fallback)
    # ------------------------------------------------------------------
    fallback_price        = float(df["Close"].iloc[-1])
    live_price, price_src = _fetch_live_price(TICKER, fallback_price)

    # ------------------------------------------------------------------
    # 6. Build history labels (handles tz-naive and tz-aware timestamps)
    # ------------------------------------------------------------------
    history_prices: list[float] = [round(float(p), 2) for p in last_n["Close"].tolist()]
    history_times:  list[str]   = []

    for ts in last_n.index:
        try:
            if ts.tzinfo is None:
                loc_ts = ts.replace(tzinfo=pytz.UTC).astimezone(NY_TZ)
            else:
                loc_ts = ts.astimezone(NY_TZ)
            history_times.append(loc_ts.strftime("%a %H:%M").upper())
        except Exception as exc:
            logger.warning("Could not parse timestamp %s: %s", ts, exc)
            history_times.append("--:--")

    # ------------------------------------------------------------------
    # 7. Scale features
    # ------------------------------------------------------------------
    features = last_n[FEATURE_ORDER].values.astype(np.float32)  # shape (SEQ_LENGTH, n_features)

    denom = scaler_max - scaler_min
    denom[denom == 0] = 1e-8  # avoid division by zero for constant features
    scaled_features = (features - scaler_min) / denom

    # ------------------------------------------------------------------
    # 8. LSTM inference
    # ------------------------------------------------------------------
    try:
        X = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)  # (1, SEQ_LENGTH, n_features)
        with torch.no_grad():
            pred_scaled = model(X).item()
    except Exception as exc:
        logger.error("Model inference failed: %s", exc)
        return _error_payload(f"Inference error: {exc}")

    # Inverse-transform: output corresponds to feature index 0 (Close)
    close_min = float(scaler_min[0])
    close_max = float(scaler_max[0])
    predicted_price = float(pred_scaled * (close_max - close_min) + close_min)

    # ------------------------------------------------------------------
    # 9. Market status
    # ------------------------------------------------------------------
    market_status, forecast_time = get_market_info()

    # ------------------------------------------------------------------
    # 10. Assemble payload
    # ------------------------------------------------------------------
    diff      = round(predicted_price - live_price, 2)
    direction = "UP" if diff > 0 else "DOWN"
    sentiment = round(float(last_n["sentiment_score"].iloc[-1]), 4)

    return {
        "status":          "success",
        "ticker":          TICKER,
        "current_price":   round(live_price, 2),
        "price_source":    price_src,           # 'live' or 'stale' — show warning in UI if 'stale'
        "predicted_price": round(predicted_price, 2),
        "difference":      diff,
        "direction":       direction,
        "sentiment_score": sentiment,
        "history":         history_prices,
        "history_times":   history_times,
        "market_status":   market_status,
        "forecast_time":   forecast_time,
    }


# ===========================================================================
# Helper — consistent error payload
# ===========================================================================

def _error_payload(message: str) -> dict:
    """
    Return a minimal error dict that preserves the same top-level keys as a
    success response so the frontend never needs to guard against missing fields.
    """
    logger.error("predict_next_hour error: %s", message)
    return {
        "status":          "error",
        "message":         message,
        "ticker":          TICKER,
        "current_price":   None,
        "price_source":    None,
        "predicted_price": None,
        "difference":      None,
        "direction":       None,
        "sentiment_score": None,
        "history":         [],
        "history_times":   [],
        "market_status":   "UNKNOWN",
        "forecast_time":   "N/A",
    }


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = predict_next_hour()
    print(json.dumps(result, indent=2))