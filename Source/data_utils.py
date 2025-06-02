import os
import yfinance as yf
import pandas as pd
import ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import INDEX_SYMBOL, TARGET_INTERVAL, START_DATE, END_DATE

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ✅ 분봉별 라벨 기준 설정
LABEL_THRESHOLDS = {
    "5m": 0.005,
    "15m": 0.007,
    "30m": 0.01,
    "60m": 0.012,
    "1d": 0.015,
}
def get_threshold(interval):
    return LABEL_THRESHOLDS.get(interval, 0.01)

def compute_indicators(df):
    try:
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

        close = df[close_col]
        df["rsi"] = ta.momentum.RSIIndicator(close).rsi()
        macd = ta.trend.MACD(close)
        df["macd"] = macd.macd_diff()
        stoch = ta.momentum.StochasticOscillator(df[high_col], df[low_col], close)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        df["sma5"] = close.rolling(5).mean()
        df["volume_change"] = df[volume_col].pct_change(fill_method=None)

        df["open"] = df[open_col]
        df["high"] = df[high_col]
        df["low"] = df[low_col]
        df["close"] = df[close_col]
        df["volume"] = df[volume_col]

        expected_cols = [
            "open", "high", "low", "close", "volume",
            "rsi", "macd", "stoch_k", "stoch_d", "sma5", "volume_change"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan

        return df[expected_cols]
    except Exception as e:
        print(f"[지표 계산 실패] {e}")
        return None

def load_cached_data(symbol, interval):
    fname = f"{symbol.replace('.', '-')}_{interval}.csv"
    path = os.path.join(CACHE_DIR, fname)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0)
            df.index = pd.to_datetime(df.index, errors="coerce")
            if not df.index.tz:
                df.index = df.index.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
            return df
        except Exception as e:
            print(f"[캐시 로딩 실패] {symbol} {interval}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_cached_data(symbol, interval, df):
    fname = f"{symbol.replace('.', '-')}_{interval}.csv"
    path = os.path.join(CACHE_DIR, fname)
    df.to_csv(path)

# ✅ 안정화된 버전
def update_cache(symbol, interval, start, end):
    symbol = symbol.replace(".", "-")
    cached_df = load_cached_data(symbol, interval)
    latest_date = cached_df.index[-1] if not cached_df.empty else pd.to_datetime(start)

    try:
        print(f"[다운로드 요청] {symbol} ({interval}) from {latest_date} to {end}")
        ticker = yf.Ticker(symbol)
        df_new = ticker.history(interval=interval, start=latest_date, end=end)

        if df_new.empty:
            print(f"[경고] {symbol} {interval}: 다운로드된 데이터가 비어 있음")
            return cached_df

        df_new.index = pd.to_datetime(df_new.index, errors="coerce")
        if not df_new.index.tz:
            df_new.index = df_new.index.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")

        df_combined = pd.concat([cached_df, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
        save_cached_data(symbol, interval, df_combined)
        return df_combined

    except Exception as e:
        print(f"❗ 다운로드 실패 ({symbol}, {interval}): {e}")
        return cached_df

def load_multitimeframe_data(symbol, index_symbol=INDEX_SYMBOL, start=START_DATE, end=END_DATE):
    intervals = ["5m", "15m", "30m", "60m", "1d"]
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

#here
def build_lstm_dataset(symbol, target_column="close"):
    from config import SYMBOL_LIST, TARGET_INTERVAL
    mtf_data = load_multitimeframe_data(symbol)

    if not mtf_data["stock"] or not mtf_data["index"]:
        print("❌ 데이터 로딩 실패")
        return None, None

    if TARGET_INTERVAL not in mtf_data["stock"]:
        print(f"❌ {TARGET_INTERVAL} 분봉 데이터가 존재하지 않습니다.")
        return None, None

    INTERVAL_MINUTES = {"2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1d": 1440}
    WINDOW_MINUTES = 60  # 1시간 기준으로 동기화
    REQUIRED_LENGTH = {k: WINDOW_MINUTES // v for k, v in INTERVAL_MINUTES.items()}

    target_df = mtf_data["stock"][TARGET_INTERVAL]
    threshold = get_threshold(TARGET_INTERVAL)

    features, labels = [], []
    ref_shape = None
    for i in range(WINDOW_MINUTES, len(target_df) - 1):
        anchor_time = target_df.index[i]
        stack = []

        for interval in INTERVAL_MINUTES.keys():
            win_len = REQUIRED_LENGTH[interval]
            for key in ["stock", "index"]:
                df = mtf_data[key].get(interval)
                if df is None or anchor_time not in df.index:
                    continue
                df_slice = df[df.index <= anchor_time].tail(win_len)
                if len(df_slice) < win_len:
                    pad = np.zeros((win_len - len(df_slice), df.shape[1]))
                    slice_arr = np.vstack([pad, df_slice.values])
                else:
                    slice_arr = df_slice.values
                stack.append(slice_arr)

        if len(stack) != len(INTERVAL_MINUTES) * 2:
            continue

        x = np.concatenate(stack, axis=1)
        if ref_shape is None:
            ref_shape = x.shape[1]
        if x.shape[1] != ref_shape:
            continue
        features.append(x)

        # label
        future = target_df[target_column].iloc[i + 1]
        current = target_df[target_column].iloc[i]
        change = (future - current) / current
        if change > threshold:
            labels.append(2)
        elif change < -threshold:
            labels.append(0)
        else:
            labels.append(1)

    if not features:
        return None, None

    X = np.stack(features)
    y = np.array(labels)

    # 정규화
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    scaler = MinMaxScaler()
    X = X.reshape(-1, X.shape[2])
    X = scaler.fit_transform(X)
    X = X.reshape(-1, WINDOW_MINUTES, ref_shape)

    print("정답 라벨 분포:", np.unique(y, return_counts=True))
    return X, y

def build_generic_dataset(interval: str, window_size=30, target_shift=1, target_column="close"):
    from config import SYMBOL_LIST
    global TARGET_INTERVAL
    TARGET_INTERVAL = interval

    X_all, y_all = [], []
    expected_dim = None

    for symbol in SYMBOL_LIST:
        print(f"📡 [{symbol} / {interval}] 데이터셋 생성 중...")
        X, y = build_lstm_dataset(symbol, window_size=window_size, target_shift=target_shift, target_column=target_column)

        if X is None or y is None:
            continue

        if expected_dim is None:
            expected_dim = X.shape[2]
        if X.shape[2] != expected_dim:
            print(f"⚠️ {symbol}: 차원 불일치 → 건너뜀")
            continue

        X_all.append(X)
        y_all.append(y)

    if not X_all:
        print(f"❌ {interval} 기준 학습 가능한 데이터 없음")
        return None, None

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    print(f"✅ {interval} 기준 총 샘플 수: {X.shape[0]}, 라벨 분포: {np.unique(y, return_counts=True)}")
    return X, y
