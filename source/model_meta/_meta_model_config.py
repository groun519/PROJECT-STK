import torch
import numpy as np

SYMBOL = "TSLA"
TARGET_INTERVAL = "30m"
META_DIR = "meta"; 
# BINS = np.linspace(-1.0, 1.0, 11)
BINS = np.array([-1.0, -0.2, 0.2, 1.0])

# ---------- train ----------
BATCH  = 128; 
EPOCHS = 2400; 
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREDICT_DATE = "2025-06-01"