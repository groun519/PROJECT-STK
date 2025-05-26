import os
import yfinance as yf
import pandas as pd
import ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import INDEX_SYMBOL, TARGET_INTERVAL, START_DATE, END_DATE

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def compute_indicators(df):
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
        else:
            df.columns = [col.lower() for col in df.columns]
        df.columns = [col.replace("_tsla", "") for col in df.columns]
        df = df.apply(pd.to_numeric, errors="coerce")

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

def load_cached_data(symbol, interval):
    fname = f"{symbol}_{interval}.csv"
    path = os.path.join(CACHE_DIR, fname)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def save_cached_data(symbol, interval, df):
    fname = f"{symbol}_{interval}.csv"
    path = os.path.join(CACHE_DIR, fname)
    df.to_csv(path)

def update_cache(symbol, interval, start, end):
    cached_df = load_cached_data(symbol, interval)
    latest_date = cached_df.index[-1] if not cached_df.empty else pd.to_datetime(start)

    try:
        df_new = yf.download(symbol, start=latest_date, end=end, interval=interval, progress=False, group_by="column")
        if df_new.empty:
            return cached_df
        df_combined = pd.concat([cached_df, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
        save_cached_data(symbol, interval, df_combined)
        return df_combined
    except Exception as e:
        print(f"❗ 다운로드 실패 ({symbol}, {interval}): {e}")
        return cached_df

def load_multitimeframe_data(symbol, index_symbol=INDEX_SYMBOL, start=START_DATE, end=END_DATE):
    intervals = ["2m", "5m", "15m", "30m", "60m", "1d"]
    data = {"stock": {}, "index": {}}
    for interval in intervals:
        stock_df = update_cache(symbol, interval, start, end)
        index_df = update_cache(index_symbol, interval, start, end)

        if not stock_df.empty:
            ind = compute_indicators(stock_df)
            if ind is not None:
                data["stock"][interval] = ind
        if not index_df.empty:
            ind = compute_indicators(index_df)
            if ind is not None:
                data["index"][interval] = ind
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
                if df is None or len(df) < i:
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

    if not features:
        return None, None

    X = np.stack(features, axis=0)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    num_samples, seq_len, input_dim = X.shape
    scaler = MinMaxScaler()
    X = X.reshape(-1, input_dim)
    X = scaler.fit_transform(X)
    X = X.reshape(num_samples, seq_len, input_dim)

    y = []
    close_series = target_df[target_column].values
    for i in range(window_size, len(close_series) - target_shift):
        future = close_series[i + target_shift]
        current = close_series[i]
        change = (future - current) / current
        if change > 0.01:
            y.append(2)
        elif change < -0.01:
            y.append(0)
        else:
            y.append(1)

    y = np.array(y)
    print("정답 라벨 분포:", np.unique(y, return_counts=True))
    return X, y
