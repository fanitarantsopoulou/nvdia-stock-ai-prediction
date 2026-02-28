import sys
import os

# Get absolute paths
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")

# Add 'src' to the very top of sys.path
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print("[*] Initializing NVDA AI Pipeline...")

try:
    # Now we import 'model' directly because 'src' is in sys.path
    from src.model.train_lstm import train_model
    train_model()
except Exception as e:
    print(f"[!] Error: {e}")
    import traceback
    traceback.print_exc()