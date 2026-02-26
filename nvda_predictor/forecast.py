# forecast.py (save this in the nvda_predictor root folder)
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.predict import predict_next_hour
predict_next_hour()