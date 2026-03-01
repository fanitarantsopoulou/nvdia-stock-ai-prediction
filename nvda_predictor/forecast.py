import json
import logging
import os
import sys

nvda_root = os.path.dirname(os.path.abspath(__file__))          # .../nvda_predictor/
src_root  = os.path.join(nvda_root, "src")                      # .../nvda_predictor/src/
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from model.predict import predict_next_hour

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = predict_next_hour()
    print(json.dumps(result, indent=2))