import os
import yfinance as yf
import pandas as pd
import ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import INDEX_SYMBOL, TARGET_INTERVAL, START_DATE, END_DATE

# ✅ 캐시 디렉토리 생성
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def compute_indicators(df):
    try:
        # ✅ 컬럼 이름 정리: 다중 인덱스 → 단일, 소문자화
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
        else:
            df.columns = [col.lower() for col in df.columns]
        df.columns = [col.replace("_tsla", "") for col in df.columns]

        # ✅ 열 타입 강제 변환
        df = df.apply(pd.to_numeric, errors="coerce")
        
        # ✅ 필수 컬럼 확인
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise KeyError(f"'{col}' column not found in df")

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

def load_from_cache_or_download(symbol, interval, start, end):
    fname = f"{symbol}_{interval}_{start}_{end}.csv".replace(":", "-")
    path = os.path.join(CACHE_DIR, fname)

    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)

    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, group_by="column")
    if not df.empty:
        df.to_csv(path)
    return df


def load_multitimeframe_data(symbol, index_symbol=INDEX_SYMBOL, start=START_DATE, end=END_DATE):
    intervals = ["2m", "5m", "15m", "30m", "60m", "1d"]
    data = {"stock": {}, "index": {}}
    for interval in intervals:
        try:
            df_stock = load_from_cache_or_download(symbol, interval, start, end)
            df_index = load_from_cache_or_download(index_symbol, interval, start, end)


            if not df_stock.empty:
                df_stock = compute_indicators(df_stock)
                data["stock"][interval] = df_stock
            if not df_index.empty:
                df_index = compute_indicators(df_index)
                data["index"][interval] = df_index
        except Exception as e:
            print(f"❗ {interval} 다운로드 실패: {e}")
    return data

def build_lstm_dataset(symbol, window_size=30, target_shift=1, target_column="close"):
    mtf_data = load_multitimeframe_data(symbol)
    if not mtf_data["stock"] or not mtf_data["index"]:
        print("❌ 데이터 로딩 실패")
        return None, None

    if TARGET_INTERVAL not in mtf_data["stock"]:
        print(f"❌ {TARGET_INTERVAL} 분봉 데이터가 존재하지 않습니다.")
        return None, None

    intervals = ["2m", "5m", "15m", "30m", "60m", "1d"]
    features = []
    target_df = mtf_data["stock"][TARGET_INTERVAL]
    if target_df is None:
        print(f"❌ {symbol}의 {TARGET_INTERVAL} 데이터 없음 또는 지표 계산 실패")
        return None, None

    for i in range(window_size, len(target_df) - target_shift):
        stack = []
        for interval in intervals:
            for key in ["stock", "index"]:
                df = mtf_data[key].get(interval)
                if df is None:
                    continue
                slice = df.iloc[i - window_size:i]
                if len(slice) < window_size:
                    pad = np.zeros((window_size - len(slice), slice.shape[1]))
                    slice = np.vstack([pad, slice.values])
                else:
                    slice = slice.values
                stack.append(slice)
        if stack:
            x = np.concatenate(stack, axis=1)
            features.append(x)

    X = np.stack(features, axis=0)

    # ✅ NaN/inf 방지: 유효하지 않은 값 제거
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)


    # ✅ MinMax 정규화 적용
    num_samples, seq_len, input_dim = X.shape
    scaler = MinMaxScaler()
    X = X.reshape(-1, input_dim)
    X = scaler.fit_transform(X)
    X = X.reshape(num_samples, seq_len, input_dim)

    # ✅ 정답 벡터 생성
    y = []
    close_series = target_df[target_column].values
    for i in range(window_size, len(close_series) - target_shift):
        future = close_series[i + target_shift]
        current = close_series[i]
        change = (future - current) / current
        if change > 0.01:
            y.append(2)  # 상승
        elif change < -0.01:
            y.append(0)  # 하락
        else:
            y.append(1)  # 관망

    y = np.array(y)
    print("정답 라벨 분포:", np.unique(y, return_counts=True))
    return X, y
