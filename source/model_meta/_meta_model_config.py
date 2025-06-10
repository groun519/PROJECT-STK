import torch
import numpy as np

SYMBOL = "TSLA"
TARGET_INTERVAL = "30m"
META_DIR = "meta"; 
BINS = np.linspace(-1.0, 1.0, 101)

# ---------- train ----------
BATCH  = 128; 
EPOCHS = 30; 
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"