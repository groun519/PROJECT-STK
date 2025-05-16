import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Trading Agent Dashboard", layout="wide")

MODELS_ROOT = Path("models")
existing_models = [d for d in MODELS_ROOT.iterdir() if d.is_dir() and (d / "live_log.csv").exists()]
model_names = [d.name for d in existing_models]

st.title("📈 실시간 강화학습 에이전트 대시보드")

if not existing_models:
    st.warning("⚠️ models/ 폴더에 live_log.csv가 있는 모델이 없습니다.")
    st.stop()

# 🔁 자동 새로고침 항상 켜기 (10초)
st_autorefresh(interval=10000, key="refresh")

# 모델 선택
selected = st.selectbox("📁 모델 버전 선택", model_names)
log_path = MODELS_ROOT / selected / "live_log.csv"

# 데이터 로딩 (캐시: 10초 유지)
@st.cache_data(ttl=10)
def load_data(path):
    try:
        return pd.read_csv(path)
    except:
        return None

df = load_data(log_path)

if df is None or df.empty:
    st.warning("⚠️ 로그 데이터가 없습니다.")
    st.stop()

# 누적 수익률 계산
initial_asset = df["asset"].iloc[0]
df["profit_rate"] = (df["asset"] - initial_asset) / initial_asset * 100

# 평가 지표 계산
returns = df["asset"].pct_change().dropna()
sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
peak = np.maximum.accumulate(df["asset"])
drawdown = (peak - df["asset"]) / peak
mdd = drawdown.max()
total_profit = df["profit_rate"].iloc[-1]

# 지표 레이아웃
col1, col2, col3 = st.columns(3)
col1.metric("💰 총 자산", f"${df['asset'].iloc[-1]:,.2f}")
col2.metric("📦 보유 주식", f"{df['holdings'].iloc[-1]}")
col3.metric("📈 누적 수익률", f"{total_profit:.2f}%")

col4, col5 = st.columns(2)
col4.metric("📉 Sharpe Ratio", f"{sharpe:.4f}")
col5.metric("🚨 Max Drawdown", f"{mdd*100:.2f}%")

# 🔄 그래프를 2개씩 나란히 배치 (최근 500개만)
st.subheader("📊 그래프 비교 보기")

gcol1, gcol2 = st.columns(2)
with gcol1:
    st.caption("📊 계좌 곡선")
    st.line_chart(df["asset"].tail(500))
with gcol2:
    st.caption("📉 주가 추이")
    st.line_chart(df["price"].tail(500))

gcol3, gcol4 = st.columns(2)
with gcol3:
    st.caption("🛠️ 최근 행동 로그")
    st.line_chart(df[["action"]].tail(500))
with gcol4:
    st.caption("💸 현금 잔고")
    st.line_chart(df[["balance"]].tail(500))

# 데이터 테이블
with st.expander("🔍 로그 데이터 보기"):
    st.dataframe(df.tail(10), use_container_width=True)
