import yfinance as yf
import pandas as pd
import ta
import numpy as np

def compute_indicators(df):
    try:
        # ✅ MultiIndex 컬럼 → 단일 문자열 컬럼으로 평탄화
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in df.columns]

        # ✅ 소문자 통일
        df.columns = [col.lower() for col in df.columns]

        # ✅ TSLA 컬럼 접미사 제거
        rename_map = {
            "close_tsla": "close",
            "open_tsla": "open",
            "high_tsla": "high",
            "low_tsla": "low",
            "volume_tsla": "volume"
        }
        df.rename(columns=rename_map, inplace=True)

        # ✅ 필수 컬럼 확인
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise KeyError(f"'{col}' column not found in df")

        # ✅ 기술 지표 계산
        close = df["close"]
        df["rsi"] = ta.momentum.RSIIndicator(close).rsi()
        macd = ta.trend.MACD(close)
        df["macd"] = macd.macd_diff()
        stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], close)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        df["sma5"] = close.rolling(5).mean()
        df["volume_change"] = df["volume"].pct_change()

        return df.dropna()

    except Exception as e:
        print(f"[지표 계산 실패] {e}")
        return None



def load_multitimeframe_data(symbol="TSLA", start="2024-04-01", end="2024-05-01"):
    intervals = ["2m", "5m", "15m", "30m", "60m", "1d"]
    data = {}
    for interval in intervals:
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                group_by="column",  # ✅ 추가
            )
            print(f"[{interval}] columns:", df.columns.tolist())  # 확인용
            if not df.empty:
                df_ind = compute_indicators(df)
                if df_ind is not None:
                    data[interval] = df_ind
                else:
                    print(f"❌ {interval}: 지표 계산 실패 → 저장하지 않음")
            else:
                print(f"⚠️ {interval}: 데이터 없음")
        except Exception as e:
            print(f"❗ Failed to load {symbol} ({interval}): {e}")
    return data
