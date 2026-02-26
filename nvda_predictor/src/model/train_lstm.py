import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import sys

# Ensure 'src' is in path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processing.merger import merge_data

# Hyperparameters
SEQ_LENGTH = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 50
LEARNING_RATE = 0.001

class StockPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StockPredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model():
    print("[*] Starting training pipeline...")
    
    # 1. Get merged data
    df = merge_data()
    if df.empty:
        print("[!] No data available for training.")
        return

    # 2. Feature selection (Close Price and Sentiment)
    features = df[['Close', 'sentiment_score']].values
    
    # 3. Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    
    # 4. Create sequences
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    
    # Convert to Tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # 5. Initialize Model, Loss, Optimizer
    model = StockPredictorLSTM(input_size=2, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"[*] Training on {len(X)} sequences...")
    
    # 6. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")
            
    # 7. Save Model and Scaler
    save_path = os.path.join(PROJECT_ROOT, 'data', 'processed')
    os.makedirs(save_path, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_path, 'lstm_model.pth'))
    np.save(os.path.join(save_path, 'scaler_params.npy'), [scaler.data_min_, scaler.data_max_])
    
    print(f"[*] Model saved successfully in {save_path}")

if __name__ == "__main__":
    train_model()