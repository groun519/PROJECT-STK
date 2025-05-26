import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from data_utils import build_lstm_dataset
from model_transformer import LSTMTransformer
from config import SYMBOL_LIST, MARKET_MODEL_PATH

import numpy as np

def train_market_model(epochs=50, batch_size=128):
    X_all, y_all = [], []
    expected_dim = None

    for symbol in SYMBOL_LIST:
        print(f"📊 [{symbol}] 학습 데이터셋 준비 중...")
        X, y = build_lstm_dataset(symbol)
        if X is not None:
            if expected_dim is None:
                expected_dim = X.shape[2]
            if X.shape[2] != expected_dim:
                print(f"⚠️ {symbol}: feature 차원 불일치 (expected {expected_dim}, got {X.shape[2]}) → 제외")
                continue
            X_all.append(X)
            y_all.append(y)

    if not X_all:
        print("❌ 학습 가능한 종목 데이터 없음")
        return

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    print("📈 전체 라벨 분포:", np.unique(y, return_counts=True))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMTransformer(input_size=X.shape[2]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    print(f"🚀 통합 학습 시작 (Epochs: {epochs})")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # ✅ Validation Accuracy 측정
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_test, val_pred)
            
        print(f"[{epoch+1}/{epochs}] Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

    os.makedirs(os.path.dirname(MARKET_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MARKET_MODEL_PATH)
    print(f"✅ 통합 모델 저장 완료: {MARKET_MODEL_PATH}")

    # 평가
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()
        print(classification_report(y_test, pred, digits=4))

if __name__ == "__main__":
    train_market_model()
