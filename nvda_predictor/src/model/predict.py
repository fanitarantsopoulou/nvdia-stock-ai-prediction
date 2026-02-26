import torch
import numpy as np
import os
import sys

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.processing.merger import merge_data
from src.model.train_lstm import StockPredictorLSTM, SEQ_LENGTH, HIDDEN_SIZE, NUM_LAYERS

def predict_next_hour():
    print("[*] Gathering latest data for prediction...")
    df = merge_data()
    if len(df) < SEQ_LENGTH:
        print(f"[!] Not enough data. Need {SEQ_LENGTH} hours.")
        return

    # Prepare features
    features = df[['Close', 'sentiment_score']].tail(SEQ_LENGTH).values
    
    # Load Scaler params
    scaler_params_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'scaler_params.npy')
    if not os.path.exists(scaler_params_path):
        print("[!] Scaler params not found. Run 'python run.py' first.")
        return
        
    scaler_min, scaler_max = np.load(scaler_params_path, allow_pickle=True)
    
    # FIX: Avoid division by zero
    denom = (scaler_max - scaler_min)
    denom[denom == 0] = 1e-8 
    scaled_features = (features - scaler_min) / denom
    
    # Load Model
    model = StockPredictorLSTM(input_size=2, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    model_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'lstm_model.pth')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Predict
    X_input = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction_scaled = model(X_input).item()

    # Inverse Scale
    actual_prediction = prediction_scaled * (scaler_max[0] - scaler_min[0]) + scaler_min[0]
    current_price = df['Close'].iloc[-1]
    diff = actual_prediction - current_price
    direction = "UP 🚀" if diff > 0 else "DOWN 📉"

    print("\n" + "="*40)
    print(f"CURRENT NVDA PRICE: ${current_price:.2f}")
    print(f"AI FORECAST (1H):  ${actual_prediction:.2f}")
    print(f"SIGNAL:            {direction} ({diff:+.2f})")
    print("="*40 + "\n")

if __name__ == "__main__":
    predict_next_hour()