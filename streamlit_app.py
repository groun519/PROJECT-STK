import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

st.set_page_config(page_title="LSTM 예측 대시보드", layout="wide")

inference_files = list(Path(".").glob("inference_*.csv"))

st.title("📊 예측 기반 전략 시각화 (LSTM 전용)")

if not inference_files:
    st.warning("⚠️ 예측 결과 파일(inference_*.csv)이 존재하지 않습니다.")
    st.stop()

# 종목 선택
symbol_map = {f.name.split("_")[1].split(".")[0]: f for f in inference_files}
symbol = st.selectbox("종목 선택", sorted(symbol_map.keys()))
df = pd.read_csv(symbol_map[symbol])

if df.empty or "prediction" not in df.columns:
    st.warning("❌ 예측 파일 형식이 올바르지 않습니다.")
    st.stop()

# 정답률 계산
df["match"] = df["prediction"] == df["actual"]
accuracy = df["match"].mean()

st.subheader(f"📈 [{symbol}] 예측 vs 실제 방향 비교")
col1, col2 = st.columns(2)
col1.metric("정답률", f"{accuracy*100:.2f}%")
col2.metric("총 샘플 수", f"{len(df)}개")

# 시계열 예측 결과 비교 (최근 100개)
chart_data = df[["step", "prediction", "actual"]].set_index("step").tail(100)
st.line_chart(chart_data, height=300)

# 예측 분포 시각화
st.subheader("📊 예측 결과 분포")
dist = df["prediction"].value_counts().sort_index()
st.bar_chart(dist)

# 최근 예측 로그
with st.expander("🧮 예측 로그 데이터"):
    st.dataframe(df.tail(20), use_container_width=True)
