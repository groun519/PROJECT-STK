import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from data_utils import compute_indicators
from model_transformer import LSTMTransformer
from config import INDEX_SYMBOL, TARGET_INTERVAL, MARKET_MODEL_PATH, TARGET_SYMBOL

WINDOW_SIZE = 20

def load_single_interval_data(symbol: str, interval: str):
    try:
        stock_df = yf.download(symbol, period="60d", interval=interval, progress=False, group_by="column")
        index_df = yf.download(INDEX_SYMBOL, period="60d", interval=interval, progress=False, group_by="column")

        data = {"stock": None, "index": None}

        if not stock_df.empty:
            data["stock"] = compute_indicators(stock_df)

        if not index_df.empty:
            data["index"] = compute_indicators(index_df)

        return data
    except Exception as e:
        print(f"[{interval}] 다운로드 실패: {e}")
        return {"stock": None, "index": None}

def build_input_tensor_single(mtf_data):
    stock_df = mtf_data["stock"]
    index_df = mtf_data["index"]

    if stock_df is None or index_df is None:
        return None

    if len(stock_df) < WINDOW_SIZE or len(index_df) < WINDOW_SIZE:
        return None

    stock_slice = stock_df.iloc[-WINDOW_SIZE:].values
    index_slice = index_df.iloc[-WINDOW_SIZE:].values

    x = np.concatenate([stock_slice, index_slice], axis=1)
    x = np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    return torch.tensor(x.reshape(1, WINDOW_SIZE, -1), dtype=torch.float32)

def predict_direction_single(symbol: str):
    print(f"📡 [{symbol}] {TARGET_INTERVAL} 분봉 예측 데이터 수집 중...")

    mtf_data = load_single_interval_data(symbol, TARGET_INTERVAL)
    if mtf_data["stock"] is None or mtf_data["index"] is None:
        print("❌ 예측 불가: 데이터 부족")
        return

    x_tensor = build_input_tensor_single(mtf_data)
    if x_tensor is None:
        print("❌ 예측 불가: 유효한 시계열 구성 실패")
        return

    input_size = x_tensor.shape[2]
    model = LSTMTransformer(input_size=input_size)
    model.load_state_dict(torch.load(MARKET_MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        out = model(x_tensor)
        probs = torch.softmax(out, dim=1).numpy().flatten()
        pred_class = np.argmax(probs)

    label_map = {0: "하락", 1: "관망", 2: "상승"}
    print(f"\n✅ 예측 결과 for {symbol} ({TARGET_INTERVAL}): {label_map[pred_class]}")
    print(f"📊 확률 분포: 하락 {probs[0]*100:.2f}%, 관망 {probs[1]*100:.2f}%, 상승 {probs[2]*100:.2f}%")

if __name__ == "__main__":
    predict_direction_single(TARGET_SYMBOL)
