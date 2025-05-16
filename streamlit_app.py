import streamlit as st
import pandas as pd
import time
import os

st.set_page_config(page_title="Trading Agent Dashboard", layout="wide")

LOG_PATH = "live_log.csv"

st.title("📈 실시간 강화학습 에이전트 대시보드")

# 데이터 로딩
def load_data():
    if not os.path.exists(LOG_PATH):
        return None
    try:
        return pd.read_csv(LOG_PATH)
    except:
        return None

# 데이터 불러오기
df = load_data()

if df is None or df.empty:
    st.warning("⚠️ 아직 로그 데이터가 없습니다.")
    st.stop()

# 누적 수익률 계산
initial_asset = df["asset"].iloc[0]
df["profit_rate"] = (df["asset"] - initial_asset) / initial_asset * 100

# ✅ 평가 지표 계산 추가
import numpy as np
returns = df["asset"].pct_change().dropna()
sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
peak = np.maximum.accumulate(df["asset"])
drawdown = (peak - df["asset"]) / peak
mdd = drawdown.max()
total_profit = df["profit_rate"].iloc[-1]

# 레이아웃 구성
col1, col2, col3 = st.columns(3)
col1.metric("💰 총 자산", f"${df['asset'].iloc[-1]:,.2f}")
col2.metric("📦 보유 주식", f"{df['holdings'].iloc[-1]}")
col3.metric("📈 누적 수익률", f"{total_profit:.2f}%")

col4, col5 = st.columns(2)
col4.metric("📉 Sharpe Ratio", f"{sharpe:.4f}")
col5.metric("🚨 Max Drawdown", f"{mdd*100:.2f}%")


# 선 차트 (자산/가격)
st.subheader("📊 계좌 곡선")
st.line_chart(df["asset"])

st.subheader("📉 주가 추이")
st.line_chart(df["price"])

# 행동 로그 표시
st.subheader("🛠️ 최근 행동 로그")
action_chart = df[["action"]].copy()
st.line_chart(action_chart[-100:])  # 최근 100개만

# 테이블 요약
with st.expander("🔍 로그 데이터 보기"):
    st.dataframe(df.tail(10), use_container_width=True)


