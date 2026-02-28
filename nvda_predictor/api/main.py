from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

from yfinance import data

# Add the root directory to the path for imports to work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add the src directory to the path for internal imports
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from model.predict import predict_next_hour

app = FastAPI()

# CORS: Allows Vue (which will run on a different port) to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, we will set the URL of the Vue app
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "AI Server is Online", "model": "NVDA-LSTM"}

@app.get("/predict")
def get_prediction():
    try:
        prediction = predict_next_hour() 
        return prediction
    except Exception as e:
        return {"error": str(e)}