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
        return {"status": "error", "message": "Not enough data"}

    # --- ULTIMATE TIMELINE FIX ---
    # 1. Sort by index to guarantee correct time order
    df = df.sort_index()
    
    # 2. Get the last 10 sequential records (ignoring date gaps)
    # Using iloc[-10:] ensures we get a continuous line even across days
    last_10 = df.iloc[-10:].copy() 
    
    # 3. Handle Live Price
    nvda_ticker = yf.Ticker(ticker_sym)
    try:
        live_price = float(nvda_ticker.fast_info['last_price'])
    except:
        live_price = float(df['Close'].iloc[-1])

    # 4. Standardize Times to New York (Market Time)
    ny_tz = pytz.timezone('America/New_York')
    history_times = []
    for t in last_10.index:
        # Convert index to NY time and format as HH:MM string
        if t.tzinfo is None:
            localized_t = t.replace(tzinfo=pytz.UTC).astimezone(ny_tz)
        else:
            localized_t = t.astimezone(ny_tz)
        history_times.append(localized_t.strftime('%H:%M'))

    history_prices = [round(p, 2) for p in last_10['Close'].tolist()]

    # --- PREPARE FEATURES FOR LSTM ---
    # Ensure features match the required sequence length
    features = df[['Close', 'sentiment_score']].tail(SEQ_LENGTH).values
    
    # Load Scaler
    scaler_params_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scaler_params.npy')
    scaler_min, scaler_max = np.load(scaler_params_path, allow_pickle=True)
    
    denom = (scaler_max - scaler_min)
    denom[denom == 0] = 1e-8 
    scaled_features = (features - scaler_min) / denom
    
    # --- MODEL INFERENCE ---
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
        return {"status": "error", "message": str(e)}

    return {
        "ticker": ticker_sym,
        "current_price": round(live_price, 2),
        "predicted_price": round(actual_prediction, 2),
        "direction": "UP" if actual_prediction > live_price else "DOWN",
        "sentiment_score": round(float(df['sentiment_score'].iloc[-1]), 2),
        "history": history_prices,
        "history_times": history_times,
        "status": "success"
    }

if __name__ == "__main__":
    # Test execution
    result = predict_next_hour()
    print(result)