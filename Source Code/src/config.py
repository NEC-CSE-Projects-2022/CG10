# config.py
import os
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEVICE = "cuda" if (os.environ.get("USE_CUDA","")=="1") else "cpu"
