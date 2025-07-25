import torch

EPOCHS = 5
BATCH_SIZE = 256

#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

MODEL_SAVE_PATH = "./models/"

LABEL_KEYS = ["trend", "regression", "candle", "highest", "position"]

TARGET = "30m" # 5, 15, [30], 60, d
