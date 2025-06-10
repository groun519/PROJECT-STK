import numpy as np, pandas as pd, matplotlib.pyplot as plt, mplfinance as mpf
from datetime import timedelta
from pathlib import Path
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정 (Windows 기준)

# ── config 연동 ------------------------------------------------------------
from data._data_config           import START_DATE, END_DATE
from data.data_fetcher           import load_multitimeframe_data
from model_meta.predict_meta_seq import forecast_100
from model_meta._meta_model_config import SYMBOL, TARGET_INTERVAL, PREDICT_DATE

# ── 설정값 적용 ------------------------------------------------------------
TRAIN_START = START_DATE
TRAIN_END   = END_DATE
PREDICT_TS  = pd.Timestamp(PREDICT_DATE).tz_localize("UTC")
REAL_BACK   = 130
SAVE_PATH   = Path("viz")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# ── 데이터 로딩 ------------------------------------------------------------
df_all = load_multitimeframe_data(SYMBOL, start=TRAIN_START, end=PREDICT_DATE, disable_log=True)
df30 = df_all["stock"][TARGET_INTERVAL]
if df30.index.tz is None:
    df30.index = df30.index.tz_localize("UTC")
else:
    df30.index = df30.index.tz_convert("UTC")

nearest_idx = df30.index.get_indexer([PREDICT_TS], method="nearest")[0]

# ── 예측 실행 --------------------------------------------------------------
pred = forecast_100(PREDICT_TS, history_start=TRAIN_START, symbol=SYMBOL)
if "error" in pred:
    raise RuntimeError(pred["error"])

# ── 예측 복원 --------------------------------------------------------------
last_close = df30["close"].iloc[nearest_idx]
df_pred = pd.DataFrame(pred["ohlc_seq"], columns=["open","high","low","close"])
df_pred = df_pred.mul(last_close)
df_pred.index = [PREDICT_TS + timedelta(minutes=30 * (i+1)) for i in range(len(df_pred))]

# ── 실제 캔들 구간: 예측과 동일 기간 ----------------------------------------
df_true = df30.loc[df_pred.index[0]:df_pred.index[-1]]

# ───────────── 시각화 ① 캔들 (예측 vs 실제) -------------------------------
fig, ax0 = plt.subplots(figsize=(14, 6))
width = 0.3  # 캔들 너비

# 실제 캔들 (검정색)
for t, row in df_true.iterrows():
    color = 'black' if row["close"] >= row["open"] else 'gray'
    ax0.plot([t, t], [row["low"], row["high"]], color=color, linewidth=1)
    ax0.bar(t, row["close"] - row["open"], width=width, bottom=row["open"], color=color, alpha=0.7)

# 예측 캔들 (빨간색)
for t, row in df_pred.iterrows():
    color = 'red' if row["close"] >= row["open"] else 'darkred'
    ax0.plot([t, t], [row["low"], row["high"]], color=color, linewidth=1)
    ax0.bar(t, row["close"] - row["open"], width=width, bottom=row["open"], color=color, alpha=0.4)

ax0.set_title(f"{SYMBOL} 예측 vs 실제 캔들 비교")
ax0.set_ylabel("가격")
ax0.tick_params(axis='x', rotation=45)
ax0.grid(alpha=0.3)
fig.savefig(SAVE_PATH / "candle.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ───────────── 시각화 ② 방향 ------------------------------------------------
pct_true = (df_true["close"].shift(-1) - df_true["close"]) / df_true["close"]
dir_true = np.where(pct_true > 0.002, 0,
           np.where(pct_true < -0.002, 2, 1))[:-1]
N = min(len(pred["position_seq"]), len(pct_true) - 1)
true_delta = pct_true.iloc[1:N+1].values
equity = np.cumsum(np.array(pred["position_seq"][:N]) * true_delta)

fig, ax1 = plt.subplots(figsize=(14, 3))
ax1.step(range(len(pred["direction_seq"])), pred["direction_seq"], where='post', label="예측")
if len(dir_true) >= len(pred["direction_seq"]):
    ax1.step(range(len(dir_true[:N])), dir_true[:N], where='post', label="실제", alpha=.5)
acc = (pred["direction_seq"][:N] == dir_true[:N]).mean()
ax1.set_title(f"예측 방향 시퀀스 (정확도: {acc:.3f})")
ax1.set_yticks([0,1,2])
ax1.set_yticklabels(["상승", "보합", "하락"])
ax1.set_ylim(-0.5, 2.5)
ax1.legend(); ax1.grid(alpha=0.3)
fig.savefig(SAVE_PATH / "direction.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ───────────── 시각화 ③ 포지션 (twin y-axis 추가) --------------------------
fig, ax2 = plt.subplots(figsize=(14, 3))
ax2.plot(pred["position_seq"], label="포지션 비율", color='blue')
ax2.axhline(0, color='gray', lw=0.5)
ax2.set_ylim(-0.1, 0.1)
ax2.set_title("포지션 비중 시퀀스")
ax2.grid(alpha=0.3)

ax22 = ax2.twinx()
ax22.axhline(pred["highest_pred"], color='red', ls='--', label='예측 최고가')
ax22.set_ylim(0, max(1.2, pred["highest_pred"] * 1.1))
ax22.set_ylabel("배율")

ax2.legend(loc="upper left")
ax22.legend(loc="upper right")
fig.savefig(SAVE_PATH / "position.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ───────────── 시각화 ④ 수익 ------------------------------------------------
fig, ax3 = plt.subplots(figsize=(14, 3))
ax3.plot(equity, color='green')
ax3.set_title("누적 수익률 (시뮬레이션)")
ax3.grid(alpha=0.3)
fig.savefig(SAVE_PATH / "return.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ───────────── 시각화: 실제 캔들 단독 -----------------------------
fig, ax_true = plt.subplots(figsize=(14, 6))
width = 0.3

for t, row in df_true.iterrows():
    color = 'black' if row["close"] >= row["open"] else 'gray'
    ax_true.plot([t, t], [row["low"], row["high"]], color=color, linewidth=1)
    ax_true.bar(t, row["close"] - row["open"], width=width, bottom=row["open"], color=color, alpha=1.0)

ax_true.set_title(f"{SYMBOL} 실제 캔들 (예측 구간)")
ax_true.set_ylabel("가격")
ax_true.tick_params(axis='x', rotation=45)
ax_true.grid(alpha=0.3)
fig.savefig(SAVE_PATH / "true_candle.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ───────────── 시각화: 예측 캔들 단독 + 최고가 라인 -----------------------------
fig, ax_pred = plt.subplots(figsize=(14, 6))
for t, row in df_pred.iterrows():
    color = 'red' if row["close"] >= row["open"] else 'darkred'
    ax_pred.plot([t, t], [row["low"], row["high"]], color=color, linewidth=1)
    ax_pred.bar(t, row["close"] - row["open"], width=width, bottom=row["open"],
                edgecolor=color, facecolor='none', linewidth=1.5)

# 최고 고가 수평선
max_high = df_pred["high"].max()
ax_pred.axhline(max_high, color='orange', ls='--', lw=1.5, label=f'예측 고가 최대값: {max_high:.2f}')
ax_pred.set_title(f"{SYMBOL} 예측 캔들 (최고 고가 포함)")
ax_pred.set_ylabel("가격")
ax_pred.tick_params(axis='x', rotation=45)
ax_pred.grid(alpha=0.3)
ax_pred.legend()
fig.savefig(SAVE_PATH / "pred_candle.png", dpi=120, bbox_inches="tight")
plt.close(fig)


print(f"✅ 시각화 6개 저장 완료: {SAVE_PATH.resolve()}")
