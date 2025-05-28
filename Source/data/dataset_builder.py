# dataset_builder.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data.data_fetcher import load_multitimeframe_data
from data.indicators import compute_indicators, get_threshold

# 각 타임프레임의 분 단위 변환값
INTERVAL_MINUTES = {
    "2m": 2,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "1d": 1440,
}

# 기준 시간 윈도우 (예: 과거 60분 = 다양한 분봉으로 구성)
WINDOW_MINUTES = 60

# 각 타임프레임에서 몇 개의 캔들이 필요한지 계산
REQUIRED_LENGTH = {k: WINDOW_MINUTES // v for k, v in INTERVAL_MINUTES.items()}

def build_lstm_dataset(symbol, interval, start, end, target_column="close"):
    mtf_data = load_multitimeframe_data(symbol, start=start, end=end)

    if not mtf_data["stock"] or not mtf_data["index"]:
        print("❌ 데이터 로딩 실패")
        return None, None

    if interval not in mtf_data["stock"]:
        print(f"❌ {interval} 분봉 데이터가 존재하지 않습니다.")
        return None, None

    target_df = mtf_data["stock"][interval]
    target_df = compute_indicators(target_df)
    threshold = get_threshold(interval)

    features, labels = [], []
    ref_shape = None

    for i in range(WINDOW_MINUTES, len(target_df) - 1):
        anchor_time = target_df.index[i]
        stack = []

        for tf in INTERVAL_MINUTES.keys():
            win_len = REQUIRED_LENGTH[tf]
            for key in ["stock", "index"]:
                df_raw = mtf_data[key].get(tf)
                if df_raw is None:
                    continue
                df = compute_indicators(df_raw)
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

        # 라벨 계산
        future = target_df[target_column].iloc[i + 1]
        current = target_df[target_column].iloc[i]
        change = (future - current) / current
        if change > threshold:
            labels.append(2)  # 상승
        elif change < -threshold:
            labels.append(0)  # 하락
        else:
            labels.append(1)  # 관망

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

def build_generic_dataset(interval, start, end, target_column="close", symbol_list=None):
    if symbol_list is None:
        from config import SYMBOL_LIST
    else:
        SYMBOL_LIST = symbol_list

    X_all, y_all = [], []
    expected_dim = None

    for symbol in SYMBOL_LIST:
        print(f"📡 [{symbol} / {interval}] 데이터셋 생성 중...")
        X, y = build_lstm_dataset(symbol, interval, start, end, target_column)

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
