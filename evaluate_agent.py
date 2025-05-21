# evaluate_lstm.py

import torch
from train_model import DirectionalLSTM
from data_utils import build_lstm_dataset
from config import SYMBOL_LIST
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

def evaluate_lstm(symbol):
    print(f"\n📊 [{symbol}] LSTM 모델 평가 시작")

    X, y = build_lstm_dataset(symbol)
    if X is None or y is None:
        print(f"❌ {symbol} 데이터셋 생성 실패")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_size = X.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DirectionalLSTM(input_size)
    model_path = f"lstm_{symbol}.pt"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    model.to(device)
    model.eval()

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(X_test).argmax(dim=1).cpu().numpy()
        labels = y_test.cpu().numpy()
        print(classification_report(labels, outputs, digits=4))

if __name__ == "__main__":
    for symbol in SYMBOL_LIST:
        evaluate_lstm(symbol)
