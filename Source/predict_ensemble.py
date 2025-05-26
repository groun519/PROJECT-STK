import os
import torch
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_utils import compute_indicators
from model_transformer import LSTMTransformer
from config import INDEX_SYMBOL

from torch.nn.functional import softmax

WINDOW_SIZE = 20
MODEL_DIR = "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_interval_data(symbol: str, interval: str):
    try:
        stock_df = yf.download(symbol, period="60d", interval=interval, progress=False)
        index_df = yf.download(INDEX_SYMBOL, period="60d", interval=interval, progress=False)
        if stock_df.empty or index_df.empty:
            return None
        return {
            "stock": compute_indicators(stock_df),
            "index": compute_indicators(index_df)
        }
    except Exception as e:
        print(f"[{interval}] 데이터 다운로드 실패: {e}")
        return None

def build_input_tensor(mtf_data):
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
    x = MinMaxScaler().fit_transform(x)
    return torch.tensor(x.reshape(1, WINDOW_SIZE, -1), dtype=torch.float32)

def predict_ensemble(symbol: str, intervals: list[str]):
    label_map = {0: "하락", 1: "관망", 2: "상승"}
    all_probs = {}
    input_tensor_cache = {}

    for interval in intervals:
        mtf_data = load_interval_data(symbol, interval)
        if mtf_data is None:
            continue

        x_tensor = build_input_tensor(mtf_data)
        if x_tensor is None:
            continue
        input_tensor_cache[interval] = x_tensor

        model_path = os.path.join(MODEL_DIR, f"direction_model_{interval}.pt")
        if not os.path.exists(model_path):
            continue

        model = LSTMTransformer(input_size=x_tensor.shape[2]).to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"❌ 모델 로딩 실패 [{interval}]: {e}")
            continue

        model.eval()
        with torch.no_grad():
            logits = model(x_tensor.to(device))
            probs = softmax(logits, dim=1).cpu().numpy().flatten()
            all_probs[interval] = probs

    if not all_probs:
        return None, {}

    # soft voting
    probs_array = np.array(list(all_probs.values()))
    avg_probs = probs_array.mean(axis=0)
    pred_class = np.argmax(avg_probs)

    return {
        "label": label_map[pred_class],
        "class_index": pred_class,
        "avg_probs": avg_probs
    }, all_probs
