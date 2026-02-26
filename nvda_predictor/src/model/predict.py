import torch
import numpy as np
import os
import sys
import yfinance as yf
import pytz

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processing.merger import merge_data
from model.train_lstm import StockPredictorLSTM, SEQ_LENGTH, HIDDEN_SIZE, NUM_LAYERS

def predict_next_hour():
    print("[*] Gathering latest data for prediction...")
    ticker_sym = "NVDA"
    df = merge_data()
    
    if df is None or len(df) < SEQ_LENGTH:
        return {"status": "error", "message": f"Not enough data. Need {SEQ_LENGTH} samples."}

    # --- 1. STRICT CHRONOLOGICAL SORT ---
    df = df.sort_index()

    # --- 2. BULLETPROOF DATA SLICE ---
    # We take EXACTLY the last 10 points. No filtering by date.
    # This guarantees a continuous, unbroken line of 10 points.
    last_10 = df.tail(10).copy()

    # --- 3. LIVE PRICE FETCH ---
    nvda_ticker = yf.Ticker(ticker_sym)
    try:
        live_price = float(nvda_ticker.fast_info['last_price'])
    except Exception:
        live_price = float(df['Close'].iloc[-1])

    # --- 4. FORMAT LABELS AS PLAIN TEXT ---
    ny_tz = pytz.timezone('America/New_York')
    history_prices = [round(p, 2) for p in last_10['Close'].tolist()]
    history_times = []
    
    for t in last_10.index:
        if t.tzinfo is None:
            loc_t = t.replace(tzinfo=pytz.UTC).astimezone(ny_tz)
        else:
            loc_t = t.astimezone(ny_tz)
        
        # ΒΑΛΕ ΑΥΤΟ: %a σημαίνει Ημέρα (π.χ. Mon, Tue). Το .upper() το κάνει κεφαλαία.
        history_times.append(loc_t.strftime('%a %H:%M').upper())

    # --- 5. PREPARE FEATURES FOR LSTM ---
    features = df[['Close', 'sentiment_score']].tail(SEQ_LENGTH).values
    
    scaler_params_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scaler_params.npy')
    if not os.path.exists(scaler_params_path):
        return {"status": "error", "message": "Scaler missing"}
        
    scaler_min, scaler_max = np.load(scaler_params_path, allow_pickle=True)
    
    denom = (scaler_max - scaler_min)
    denom[denom == 0] = 1e-8 
    scaled_features = (features - scaler_min) / denom
    
    # --- 6. MODEL INFERENCE ---
    model = StockPredictorLSTM(input_size=2, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    model_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'lstm_model.pth')
    
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        
        X_input = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction_scaled = model(X_input).item()

        actual_prediction = float(prediction_scaled * (scaler_max[0] - scaler_min[0]) + scaler_min[0])
    except Exception as e:
        return {"status": "error", "message": f"Inference failed: {str(e)}"}

    # --- 7. FINAL PAYLOAD ---
    diff = float(actual_prediction - live_price)
    direction = "UP" if diff > 0 else "DOWN"
    sentiment = float(df['sentiment_score'].iloc[-1])

    return {
        "ticker": ticker_sym,
        "current_price": round(live_price, 2),
        "predicted_price": round(actual_prediction, 2),
        "difference": round(diff, 2),
        "direction": direction,
        "sentiment_score": round(sentiment, 2),
        "history": history_prices,
        "history_times": history_times,
        "status": "success"
    }

if __name__ == "__main__":
    print(predict_next_hour())