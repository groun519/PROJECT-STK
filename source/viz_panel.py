# source/tools/viz_panel.py
import numpy as np, pandas as pd, matplotlib.pyplot as plt, mplfinance as mpf
from datetime import timedelta
from pathlib import Path

# ── 프로젝트 설정 / 함수 ----------------------------------------------------
from data._data_config           import START_DATE, END_DATE, SYMBOL_LIST
from data.data_fetcher           import load_multitimeframe_data
from model_meta.predict_meta_seq import forecast_100, is_market_time
from model_meta._meta_model_config import SYMBOL   # 기본 심볼 (TSLA)

# ── 자동 계산: 학습·예측 범위 ----------------------------------------------
TRAIN_START = START_DATE                               # 2025-05-01
TRAIN_END   = END_DATE                                 # 2025-06-01
EVAL_START  = END_DATE                                 # 6-01
EVAL_END    = (pd.Timestamp(END_DATE) +                # 6-10  (= END+9d)
               pd.Timedelta(days=9)).strftime("%Y-%m-%d")

PREDICT_TS  = pd.Timestamp(EVAL_START).tz_localize("UTC")
SAVE_PATH   = Path(".") / "viz_panel.png"
REAL_BACK   = 50                                       # 실제 구간 캔들 수

# ── 데이터 로드 -------------------------------------------------------------
df30 = load_multitimeframe_data(
           SYMBOL, start=TRAIN_START, end=EVAL_END)["stock"]["30m"]

if df30.index.tz is None:
    df30.index = df30.index.tz_localize("UTC")
else:
    df30.index = df30.index.tz_convert("UTC")

# 이후
nearest_idx = df30.index.get_indexer([PREDICT_TS], method="nearest")[0]
last_close  = df30["close"].iloc[nearest_idx]

real_eval = df30.loc[EVAL_START:EVAL_END]

# ── 100-캔들 예측 -----------------------------------------------------------
pred = forecast_100(PREDICT_TS, history_start=TRAIN_START, symbol=SYMBOL)
if "error" in pred:
    raise RuntimeError(pred["error"])

# ── 예측 OHLC → 절대가 ------------------------------------------------------
last_close = df30.loc[PREDICT_TS]["close"]
df_pred = (pd.DataFrame(pred["ohlc_seq"],
          columns=["open","high","low","close"])
          .add(1).mul(last_close))
df_pred.index = [PREDICT_TS + timedelta(minutes=30*(i+1))
                 for i in range(len(df_pred))]

# ── 캔들 차트 데이터 --------------------------------------------------------
candle_all = pd.concat([real_eval.iloc[-REAL_BACK:], df_pred])
candle_all.index.name = "datetime"

# ── 실제 방향 / 누적 수익 계산 ---------------------------------------------
pct_true = (real_eval["close"].shift(-1) -
            real_eval["close"]) / real_eval["close"]
dir_true = np.where(pct_true > 0.002, 0,
           np.where(pct_true < -0.002, 2, 1))[:-1]

true_delta = pct_true.iloc[1:len(pred["position_seq"])+1].values
equity = np.cumsum(np.array(pred["position_seq"]) * true_delta)

# ── 시각화 -----------------------------------------------------------------
fig = plt.figure(figsize=(14, 12))
gs  = fig.add_gridspec(4, 1, height_ratios=[3,1,1,1], hspace=0.4)

# ① 캔들
ax0 = fig.add_subplot(gs[0])
mpf.plot(candle_all, type='candle', ax=ax0,
         axtitle=f"{SYMBOL} 30-m  (real {REAL_BACK} + pred {len(df_pred)})",
         style='yahoo', ylabel='Price')

# ② 방향 시퀀스
ax1 = fig.add_subplot(gs[1])
ax1.step(range(len(pred["direction_seq"])), pred["direction_seq"],
         where='post', label="pred")
if len(dir_true) >= len(pred["direction_seq"]):
    ax1.step(range(len(dir_true[:len(pred["direction_seq"])])),
             dir_true[:len(pred["direction_seq"])], where='post',
             label="true", alpha=.5)
acc = (pred["direction_seq"][:len(dir_true)] == dir_true).mean()
ax1.set_title(f"Direction sequence   (ACC={acc:.3f})")
ax1.set_yticks([0,1,2], ["↑","→","↓"]); ax1.legend(); ax1.grid(alpha=.3)

# ③ 비중 & 고점
ax2 = fig.add_subplot(gs[2])
ax2.plot(pred["position_seq"], label="position_frac")
ax2.axhline(0, color='k', lw=.5)
ax2.set_title("Position fraction sequence"); ax2.grid(alpha=.3)
ax22 = ax2.twinx()
ax22.axhline(pred["highest_pred"], color='r', ls='--', label='highest_pred')
ax22.legend(loc="upper right")

# ④ 누적 수익
ax3 = fig.add_subplot(gs[3])
ax3.plot(equity); ax3.set_title("Cumulative return (sim)")
ax3.grid(alpha=.3)

fig.savefig(SAVE_PATH, dpi=120, bbox_inches="tight")
print("✅ 시각화 저장:", SAVE_PATH)
