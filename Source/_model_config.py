EPOCHS = 30
BATCH_SIZE = 128
DEVICE = "cuda"
MODEL_SAVE_PATH = "./saved_models/"
LEARNING_RATE = 1e-3

INTERVALS = ["5m"]
# INTERVALS = ["5m", "15m", "30m", "60m", "1d"]

# 모델 구조 관련 설정
TRANSFORMER_DIM = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2

LABEL_KEYS = ["trend", "regression", "candle", "highest", "position"]
