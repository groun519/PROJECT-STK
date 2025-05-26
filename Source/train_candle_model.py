import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from data_utils import build_lstm_dataset
from config import TARGET_SYMBOL, TARGET_INTERVAL

# ✅ LSTM 회귀 모델 정의
class CandleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# ✅ 데이터 구성
def build_regression_labels(symbol, window_size=30, target_shift=1):
    X, _ = build_lstm_dataset(symbol, window_size, target_shift)
    if X is None:
        return None, None

    # 다음 시점 OHLC만 별도로 구성
    mtf_data = build_lstm_dataset.mtf_data_cache  # 기존 전처리 데이터 활용
    df = mtf_data["stock"].get(TARGET_INTERVAL)
    if df is None or len(df) < len(X) + window_size:
        print("❌ OHLC 추출 실패")
        return None, None

    future_ohlc = df[["open", "high", "low", "close"]].values[window_size + target_shift:]
    if len(future_ohlc) > len(X):
        future_ohlc = future_ohlc[:len(X)]
    return X[:len(future_ohlc)], future_ohlc

# ✅ 학습 파라미터
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
MODEL_PATH = f"models/candle_model_{TARGET_INTERVAL}.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 데이터 로딩
X, y = build_regression_labels(TARGET_SYMBOL)
if X is None or y is None:
    print("❌ 학습 데이터 생성 실패")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ✅ 모델 구성
input_dim = X.shape[2]
model = CandleLSTM(input_size=input_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

# ✅ 모델 저장
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ 모델 저장 완료: {MODEL_PATH}")

# ✅ 테스트 평가
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    preds = model(X_test_tensor).cpu().numpy()
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

print("\n📊 [검증 결과]")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
