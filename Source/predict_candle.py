import os
import torch
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

from data_utils import compute_indicators
from config import TARGET_SYMBOL, TARGET_INTERVAL
from train_candle_model import CandleLSTM

# 🔧 설정
WINDOW_SIZE = 20
MODEL_PATH = f"models/candle_model_{TARGET_INTERVAL}.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 입력 데이터 구성
def get_recent_input(symbol, interval):
    try:
        df = yf.download(symbol, period="60d", interval=interval, progress=False)
        df = compute_indicators(df)
        if df is None or len(df) < WINDOW_SIZE + 1:
            return None, None
        latest = df.iloc[-WINDOW_SIZE:].values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(latest)
        x = torch.tensor(scaled.reshape(1, WINDOW_SIZE, -1), dtype=torch.float32)
        return x, df.iloc[-WINDOW_SIZE:].copy()
    except Exception as e:
        print(f"❌ 입력 생성 실패: {e}")
        return None, None

# ✅ 예측 수행
def predict_next_ohlc(x_tensor):
    input_size = x_tensor.shape[2]
    model = CandleLSTM(input_size).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    with torch.no_grad():
        out = model(x_tensor.to(device)).cpu().numpy().flatten()
    return out  # [open, high, low, close]

# ✅ 시각화
def plot_ohlc_with_prediction(df_recent, pred_ohlc):
    df = df_recent.copy()
    pred_time = df.index[-1] + pd.Timedelta("1min")  # 분봉 기준
    df_pred = pd.DataFrame({
        "open": [pred_ohlc[0]],
        "high": [pred_ohlc[1]],
        "low": [pred_ohlc[2]],
        "close": [pred_ohlc[3]]
    }, index=[pred_time])
    df_combined = pd.concat([df[["open", "high", "low", "close"]], df_pred])

    fig = go.Figure(data=[go.Candlestick(
        x=df_combined.index,
        open=df_combined["open"],
        high=df_combined["high"],
        low=df_combined["low"],
        close=df_combined["close"],
        increasing_line_color="green",
        decreasing_line_color="red"
    )])
    fig.update_layout(title="최근 캔들 + 예측 캔들", xaxis_rangeslider_visible=False)
    fig.show()

# ✅ 실행
if __name__ == "__main__":
    print(f"📡 [{TARGET_SYMBOL}] {TARGET_INTERVAL} 캔들 예측 시작")
    x_input, df_recent = get_recent_input(TARGET_SYMBOL, TARGET_INTERVAL)
    if x_input is None:
        print("❌ 입력 데이터 불충분")
        exit()

    pred = predict_next_ohlc(x_input)
    print("\n🔮 예측된 다음 OHLC:")
    print(f"Open:  {pred[0]:.2f}")
    print(f"High:  {pred[1]:.2f}")
    print(f"Low:   {pred[2]:.2f}")
    print(f"Close: {pred[3]:.2f}")

    plot_ohlc_with_prediction(df_recent, pred)
