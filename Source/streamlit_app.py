import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

from predict_ensemble import predict_ensemble
from recommend_position import recommend_position
from train_candle_model import CandleLSTM
from data_utils import compute_indicators
from config import INDEX_SYMBOL

INTERVALS = ["2m", "5m", "15m", "30m", "60m", "1d"]
WINDOW_SIZE = 20

st.set_page_config(page_title="📊 AI 트레이딩 비서", layout="centered")
st.title("📈 AI 기반 트레이딩 포지션 & 캔들 예측 시스템")

# 🎛️ 사용자 입력
symbol = st.text_input("📍 종목 심볼 입력", value="NVDA")
selected_intervals = st.multiselect(
    "⏱ 사용할 분봉 선택", options=INTERVALS, default=["5m", "15m", "1d"]
)
target_interval = st.selectbox("📊 캔들 예측 분봉 선택", options=INTERVALS, index=1)

if st.button("🚀 예측 실행") and selected_intervals:
    with st.spinner("AI 예측 중..."):
        result, indiv = predict_ensemble(symbol, selected_intervals)

    if result is None:
        st.error("예측 실패: 데이터 또는 모델 부족")
    else:
        label_map = {0: "📉 하락", 1: "⏸ 관망", 2: "📈 상승"}

        st.subheader("🔮 방향성 예측 결과")
        st.markdown(f"**최종 예측:** {label_map[result['class_index']]}")

        st.subheader("📊 평균 확률 분포")
        st.bar_chart(pd.DataFrame({
            "하락": [result["avg_probs"][0]],
            "관망": [result["avg_probs"][1]],
            "상승": [result["avg_probs"][2]],
        }).T.rename(columns={0: "확률"}))

        st.subheader("📂 분봉별 예측 결과")
        indiv_df = pd.DataFrame(indiv).T
        indiv_df.columns = ["하락", "관망", "상승"]
        st.dataframe(indiv_df.style.highlight_max(axis=1, color="lightgreen"))

        st.subheader("🧠 포지션 판단")
        recommendation = recommend_position(result["avg_probs"], indiv)
        st.success(recommendation)

    # 📈 캔들 예측 실행
    st.markdown("---")
    st.subheader("🕯️ 다음 캔들 예측 (회귀 기반)")

    @st.cache_data(show_spinner=False)
    def load_recent_data(symbol, interval):
        df = yf.download(symbol, period="60d", interval=interval, progress=False)
        df = compute_indicators(df)
        return df if df is not None and len(df) > WINDOW_SIZE else None

    df_recent = load_recent_data(symbol, target_interval)

    if df_recent is None:
        st.warning(f"{target_interval} 분봉 데이터 부족 또는 지표 계산 실패")
    else:
        # 입력 구성
        recent_slice = df_recent.iloc[-WINDOW_SIZE:].copy()
        scaler = MinMaxScaler()
        x = scaler.fit_transform(recent_slice.values)
        x_tensor = torch.tensor(x.reshape(1, WINDOW_SIZE, -1), dtype=torch.float32)

        # 모델 로딩 & 예측
        input_size = x_tensor.shape[2]
        model_path = f"models/candle_model_{target_interval}.pt"
        model = CandleLSTM(input_size)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            pred = model(x_tensor).numpy().flatten()

        # 캔들 시각화
        pred_time = df_recent.index[-1] + pd.Timedelta("1min")
        df_pred = pd.DataFrame({
            "open": [pred[0]], "high": [pred[1]],
            "low": [pred[2]], "close": [pred[3]]
        }, index=[pred_time])
        df_all = pd.concat([df_recent[["open", "high", "low", "close"]].iloc[-WINDOW_SIZE:], df_pred])

        fig = go.Figure(data=[go.Candlestick(
            x=df_all.index,
            open=df_all["open"],
            high=df_all["high"],
            low=df_all["low"],
            close=df_all["close"],
            increasing_line_color="green",
            decreasing_line_color="red"
        )])
        fig.update_layout(title="최근 캔들 + 예측 캔들", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)

        st.markdown("🔮 **예측된 OHLC**")
        st.write(f"Open: `{pred[0]:.2f}` | High: `{pred[1]:.2f}` | Low: `{pred[2]:.2f}` | Close: `{pred[3]:.2f}`")
