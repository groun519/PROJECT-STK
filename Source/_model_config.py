import torch

EPOCHS = 60
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "./models/"

LABEL_KEYS = ["trend", "regression", "candle", "highest", "position"]

TARGET = "30m" # 5, 15, [30], 60, d
