import torch

SYMBOL = "TSLA"
TARGET_INTERVAL = "30m"
META_DIR = "meta"; 

# ---------- train ----------
BATCH  = 128; 
EPOCHS = 30; 
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"