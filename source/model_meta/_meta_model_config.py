import torch
import numpy as np

SYMBOL = "TSLA"
TARGET_INTERVAL = "30m"
META_DIR = "meta"; 
BINS = np.linspace(-1.0, 1.0, 11)
#BINS = np.array([-1.0, -0.2, 0.2, 1.0])

# ---------- train ----------
BATCH  = 128; 
EPOCHS = 120; 
LR = 1e-3
DEVICE = "cpu"

PREDICT_DATE = "2025-06-01"