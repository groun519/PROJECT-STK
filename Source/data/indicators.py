import pandas as pd
import numpy as np
import ta

# ✅ 라벨링을 위한 수익률 임계값 (타임프레임별)
LABEL_THRESHOLDS = {
    "2m": 0.003,
    "5m": 0.005,
    "15m": 0.007,
    "30m": 0.010,
    "60m": 0.012,
    "1d": 0.015,
}

def get_threshold(interval):
    return LABEL_THRESHOLDS.get(interval, 0.01)

def compute_indicators(df):
    try:
        # ✅ 컬럼 통일 처리
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None, map(str, col))).lower() for col in df.columns]
        else:
            df.columns = [str(col).lower() for col in df.columns]

        df = df.apply(pd.to_numeric, errors="coerce")

        def find_col(possible_names):
            for name in df.columns:
                for key in possible_names:
                    if name.startswith(key):
                        return name
            return None

        open_col = find_col(["open"])
        high_col = find_col(["high"])
        low_col = find_col(["low"])
        close_col = find_col(["close"])
        volume_col = find_col(["volume"])

        if not all([open_col, high_col, low_col, close_col, volume_col]):
            raise KeyError("필수 컬럼 누락: open/high/low/close/volume")

        # ✅ 기술 지표 계산
        close = df[close_col]
        high = df[high_col]
        low = df[low_col]
        volume = df[volume_col]

        df["rsi"] = ta.momentum.RSIIndicator(close).rsi()
        macd = ta.trend.MACD(close)
        df["macd"] = macd.macd_diff()
        df["ema20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        bb = ta.volatility.BollingerBands(close)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["volume_change"] = volume.pct_change(fill_method=None)

        # ✅ 컬럼 이름 통일
        df["open"] = df[open_col]
        df["high"] = df[high_col]
        df["low"] = df[low_col]
        df["close"] = df[close_col]
        df["volume"] = df[volume_col]

        expected_cols = [
            "open", "high", "low", "close", "volume",
            "rsi", "macd", "ema20", "bb_upper", "bb_lower", "volume_change"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan

        return df[expected_cols]

    except Exception as e:
        print(f"[지표 계산 실패] {e}")
        return None
