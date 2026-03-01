import logging
import os
import sys

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processing.merger import merge_data

logger = logging.getLogger(__name__)

# Hyperparameters
SEQ_LENGTH    = 10
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
EPOCHS        = 50
LEARNING_RATE = 0.001
BATCH_SIZE    = 32
TRAIN_RATIO   = 0.8
RANDOM_SEED   = 42

FEATURE_ORDER = ["Close", "sentiment_score"]
SAVE_PATH     = os.path.join(PROJECT_ROOT, "data", "processed")


class StockPredictorLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_sequences(data: np.ndarray, seq_length: int):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i : i + seq_length])
        ys.append(data[i + seq_length, 0])  # target = Close price
    return np.array(xs), np.array(ys)


def train_model() -> None:
    logger.info("Starting training pipeline...")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── 1. Data ───────────────────────────────────────────────────────────
    df = merge_data()
    if df is None or df.empty:
        logger.error("No data available for training.")
        return

    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        logger.error("Required columns missing from merged data: %s", missing)
        return

    features = df[FEATURE_ORDER].values

    if len(features) < SEQ_LENGTH + 1:
        logger.error(
            "Insufficient data: %d rows available, need at least %d.",
            len(features), SEQ_LENGTH + 1,
        )
        return

    # ── 2. Scaling ────────────────────────────────────────────────────────
    # The scaler is persisted with joblib so predict.py can load it directly
    # and call scaler.inverse_transform() — no manual min/max arithmetic that
    # would silently break if feature_range were changed.
    from sklearn.preprocessing import MinMaxScaler
    scaler      = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)

    # Also save raw min/max for the lightweight manual inverse in predict.py
    np.save(
        os.path.join(SAVE_PATH, "scaler_params.npy"),
        [scaler.data_min_, scaler.data_max_],
    )

    # ── 3. Sequences ──────────────────────────────────────────────────────
    X_all, y_all = create_sequences(scaled_data, SEQ_LENGTH)

    if len(X_all) == 0:
        logger.error("No sequences could be created. Check SEQ_LENGTH vs data length.")
        return

    # Chronological split — never shuffle time-series data
    split      = int(len(X_all) * TRAIN_RATIO)
    X_train    = torch.tensor(X_all[:split],  dtype=torch.float32)
    y_train    = torch.tensor(y_all[:split],  dtype=torch.float32).unsqueeze(1)
    X_val      = torch.tensor(X_all[split:],  dtype=torch.float32)
    y_val      = torch.tensor(y_all[split:],  dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=False,  # Must be False for time-series
    )

    logger.info(
        "Sequences — train: %d  val: %d  (split ratio: %.0f%%)",
        len(X_train), len(X_val), TRAIN_RATIO * 100,
    )

    # ── 4. Model ──────────────────────────────────────────────────────────
    model     = StockPredictorLSTM(input_size=len(FEATURE_ORDER), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── 5. Training loop ──────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            # Validation loss — no gradient computation needed
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), y_val).item()
            avg_train = epoch_loss / len(train_loader)
            logger.info(
                "Epoch [%d/%d]  train_loss: %.6f  val_loss: %.6f",
                epoch + 1, EPOCHS, avg_train, val_loss,
            )

    # ── 6. Persist ────────────────────────────────────────────────────────
    os.makedirs(SAVE_PATH, exist_ok=True)

    model_out  = os.path.join(SAVE_PATH, "lstm_model.pth")
    scaler_out = os.path.join(SAVE_PATH, "scaler.joblib")

    torch.save(model.state_dict(), model_out)
    joblib.dump(scaler, scaler_out)

    logger.info("Model saved  → %s", model_out)
    logger.info("Scaler saved → %s", scaler_out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train_model()