import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_fetcher import load_multitimeframe_data
from Source.data._data_config import (
    SYMBOL_LIST, TARGET_INTERVAL, TARGET_COLUMN,
    INTERVAL_MINUTES, WINDOW_MINUTES,
    LABEL_THRESHOLDS, REQUIRED_LENGTH
)

def get_threshold(interval):
    return LABEL_THRESHOLDS.get(interval, 0.01)

def build_lstm_dataset(symbol):
    mtf_data = load_multitimeframe_data(symbol)

    if not mtf_data["stock"] or not mtf_data["index"]:
        print("❌ 데이터 로딩 실패")
        return None, None

    if TARGET_INTERVAL not in mtf_data["stock"]:
        print(f"❌ {TARGET_INTERVAL} 분봉 데이터가 존재하지 않습니다.")
        return None, None

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

        # 라벨링: 이진 분류 (상승 vs 하락/보합)
        future = target_df[TARGET_COLUMN].iloc[i + 1]
        current = target_df[TARGET_COLUMN].iloc[i]
        change = (future - current) / current
        label = 1 if change > threshold else 0
        labels.append(label)

        # 🎯 주석: 향후 수익률 회귀 예측 라벨링도 고려 가능
        # labels.append(change)

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

    return X, y

def build_generic_dataset(interval: str):
    global TARGET_INTERVAL
    TARGET_INTERVAL = interval

    X_all, y_all = [], []
    expected_dim = None

    for symbol in SYMBOL_LIST:
        print(f"📡 [{symbol} / {interval}] 데이터셋 생성 중...")
        X, y = build_lstm_dataset(symbol)

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
    print(f"✅ {interval} 기준 총 샘플 수: {X.shape[0]}")
    return X, y

def show_label_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  🟢 라벨 {u}: {c}개 ({(c / len(y)) * 100:.2f}%)")
