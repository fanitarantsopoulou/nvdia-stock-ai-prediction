import sys
import os

# Add the root directory to the path for imports to work
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from model.predict import predict_next_hour

if __name__ == "__main__":
    predict_next_hour()