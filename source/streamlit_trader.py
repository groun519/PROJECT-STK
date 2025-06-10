# source/streamlit_trader.py
import streamlit as st, pandas as pd, plotly.graph_objects as go
from model_meta.predict_meta      import predict
from model_meta.predict_meta_seq  import forecast_100, is_market_time
from data._data_config            import INTERVALS

st.set_page_config(page_title="AI 트레이딩 대시보드", layout="wide")
st.title("📈 AI 30-분봉 예측 & 포지션")

symbol       = st.text_input("심볼", "TSLA")
mode         = st.selectbox("모델 타입", ["단일틱(30m)", "시퀀스(100틱)"])
interval_sel = st.selectbox("예측 분봉", ["30m","60m"])

if st.button("🚀 예측 실행"):
    ts_now = pd.Timestamp.utcnow().floor("30min").tz_localize("UTC")
    hist_start = "2025-04-01"

    if mode.startswith("단일"):
        res = predict(ts_now, history_start=hist_start)
        st.write(res)
    else:
        res = forecast_100(ts_now, history_start=hist_start, symbol=symbol)
        if "error" in res:
            st.error(res["error"])
        else:
            st.write(f"**{len(res['direction_seq'])} 캔들 예측 완료**")
            # ─ 캔들 차트 (예측은 선) ─
            df_pred = pd.DataFrame(res["ohlc_seq"], columns=["open","high","low","close"])
            df_pred["ts"] = [ts_now + pd.Timedelta(minutes=30*(i+1))
                              for i in range(len(df_pred))]
            fig = go.Figure(data=[go.Candlestick(
                x=df_pred["ts"],
                open=df_pred["open"],
                high=df_pred["high"],
                low=df_pred["low"],
                close=df_pred["close"],
                increasing_line_color="green", decreasing_line_color="red"
            )])
            fig.update_layout(height=400, title="예측 100-캔들 OHLC(배율)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 방향 시퀀스")
            st.bar_chart(pd.Series(res["direction_seq"]).value_counts().sort_index())

            st.subheader("⚖️ 비중 시퀀스")
            st.line_chart(res["position_seq"])
