import logging
import os
import sys

nvda_root = os.path.dirname(os.path.abspath(__file__))
src_root  = os.path.join(nvda_root, "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from model.train_lstm import train_model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Initializing NVDA AI Pipeline...")
    try:
        train_model()
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)