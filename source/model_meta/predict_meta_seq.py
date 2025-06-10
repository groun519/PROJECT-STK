# source/model_meta/predict_meta_seq.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import sys, numpy as np, pandas as pd
from datetime import timedelta
from pandas import Timestamp

from data._data_config import (
    START_DATE
)

# 기존 단일-틱 예측 함수·설정 재사용
from model_meta.predict_meta import predict, load_multitimeframe_data, is_market_time
from model_meta._meta_model_config import (
    SYMBOL, PREDICT_DATE
)

# ────────────────────────────────────────────────────────────────
def _append_fake_candle(df, ts, trend_frac, ohlc_ratio):
    """
    df : 30 분봉 DataFrame (columns: open, high, low, close, volume)
    ts : 다음 캔들 Timestamp
    trend_frac : 한 캔들 종가 변화율
    ohlc_ratio : 길이 4 의 [openΔ, highΔ, lowΔ, closeΔ] 비율
    """
    last_close = df["close"].iloc[-1]
    open_  = last_close * (1 + trend_frac)                 # 간단: open = prev_close*(1+Δ)
    high_  = open_ * (1 + max(ohlc_ratio[1], 0))
    low_   = open_ * (1 + min(ohlc_ratio[2], 0))
    close_ = open_ * (1 + ohlc_ratio[3])

    df.loc[ts] = [open_, high_, low_, close_, np.nan]      # volume 은 NaN

# ────────────────────────────────────────────────────────────────
def forecast_100(ts_start: pd.Timestamp,
                 history_start: str | pd.Timestamp = START_DATE,
                 symbol: str = SYMBOL,
                 steps: int = 450):
    """
    ts_start      : 예측 기준 30 분봉 Timestamp (UTC 권장)
    history_start : 과거 캔들 최소 시작일 ('YYYY-MM-DD' 또는 Timestamp)
    symbol        : 종목
    steps         : 예측할 캔들 수 (기본 100)

    반환 dict
      direction_seq : list[int]  (0↑ 1→ 2↓)
      delta_seq     : list[float]
      position_seq  : list[float]
      ohlc_seq      : list[list[float]]  각 원소 길이 4
      highest_pred  : float | None       (100 캔들 내 예측 최고가 비율)
    """
    # ── 시각 보정 (UTC tz-aware) ───────────────────────────────
    ts_start = (ts_start if ts_start.tzinfo else ts_start.tz_localize("UTC")).tz_convert("UTC")
    hist_start = (pd.Timestamp(history_start)
                  if isinstance(history_start, str) else history_start)
    hist_start = (hist_start if hist_start.tzinfo else hist_start.tz_localize("UTC")).tz_convert("UTC")

    # ── 과거 30 분봉 로드 (close 컬럼만 있으면 충분) ───────────
    df_30 = load_multitimeframe_data(
                symbol,
                start=hist_start.strftime("%Y-%m-%d"),
                end  =ts_start.strftime("%Y-%m-%d"),
                disable_log=True
            )["stock"]["30m"].copy()

    if len(df_30) == 0:
        return {"error": "no historical data"}

    direction_seq, delta_seq, pos_seq, ohlc_seq = [], [], [], []
    cur_ts = ts_start

    for _ in range(steps):
        if not is_market_time(cur_ts):
            cur_ts += timedelta(minutes=30)
            continue
        
        res = predict(cur_ts)
        if res is None:
            cur_ts += timedelta(minutes=30)
            continue

        direction_seq.append(res["direction_cls"])
        delta_seq.append(res["trend_frac"])
        pos_seq.append(res["position_frac"])
        ohlc_seq.append(res["candle_100"])

        # 다음 캔들 Timestamp
        next_ts = cur_ts + timedelta(minutes=30)
        _append_fake_candle(df_30, next_ts,
                            res["trend_frac"],
                            res["candle_100"])
        cur_ts = next_ts

    highest_pred = (max(x[1] for x in ohlc_seq)    # highΔ 중 최대
                    if ohlc_seq else None)

    return {
        "direction_seq": direction_seq,
        "delta_seq":     delta_seq,
        "position_seq":  pos_seq,
        "ohlc_seq":      ohlc_seq,
        "highest_pred":  highest_pred
    }

# ─────────────────────────────── CLI 테스트 ────────────────────────────────
if __name__ == "__main__":
    # python predict_meta_seq.py   [predict_ts]   [history_start]
    ts_arg  = sys.argv[1] if len(sys.argv) > 1 else PREDICT_DATE
    hist_arg= sys.argv[2] if len(sys.argv) > 2 else START_DATE

    ts0 = Timestamp(ts_arg)
    result = forecast_100(ts0, history_start=hist_arg)

    from pprint import pprint
    pprint({"steps": len(result["direction_seq"]), **result})
