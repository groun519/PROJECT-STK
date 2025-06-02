import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_fetcher import load_multitimeframe_data
from labeling_utils import (
    label_binary, label_three_class,
    label_position_class, label_return_regression
)
from _data_config import (
    SYMBOL_LIST, TARGET_INTERVAL, TARGET_COLUMN,
    INTERVAL_MINUTES, REQUIRED_LENGTH, LABEL_THRESHOLDS
)

def get_threshold(interval):
    return LABEL_THRESHOLDS.get(interval, 0.01)

def get_label_function(mode="binary"):
    if mode == "binary":
        return label_binary
    elif mode == "three":
        return label_three_class
    elif mode == "position":
        return label_position_class
    elif mode == "regression":
        return label_return_regression
    else:
        raise ValueError(f"[라벨링 모드 오류] 지원하지 않는 라벨링 모드: {mode}")

def normalize_features(X):
    scaler = MinMaxScaler()
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    X = X.reshape(-1, X.shape[2])
    X = scaler.fit_transform(X)
    X = X.reshape(-1, X.shape[1], X.shape[2])
    return X, scaler

def build_lstm_dataset(symbol):
    from data_fetcher import load_multitimeframe_data

    # 데이터 가져오기
    mtf_data = load_multitimeframe_data(symbol)
    if not mtf_data["stock"] or not mtf_data["index"]:
        print(f"❌ {symbol}: 데이터 로딩 실패 (stock or index)")
        return None, None

    if TARGET_INTERVAL not in mtf_data["stock"]:
        print(f"❌ {symbol}: {TARGET_INTERVAL} 분봉 데이터가 없습니다.")
        return None, None

    target_df = mtf_data["stock"][TARGET_INTERVAL]
    threshold = get_threshold(TARGET_INTERVAL)
    label_fn = get_label_function(LABELING_MODE)

    features, labels = [], []
    ref_shape = None

    # (불필요하게 복잡한 index 변환 등은 생략)
    for i in range(REQUIRED_LENGTH[TARGET_INTERVAL], len(target_df) - 1):
        anchor_time = target_df.index[i]
        stack, valid = [], True

        for interval in INTERVAL_MINUTES.keys():
            win_len = REQUIRED_LENGTH[interval]
            for key in ["stock", "index"]:
                df = mtf_data[key].get(interval)
                if df is None or len(df) < win_len:
                    valid = False
                    break

                # anchor_time에 근접한 위치에서 win_len 만큼 슬라이스
                pos = df.index.get_indexer([anchor_time], method="nearest")[0]
                if pos == -1 or pos < win_len:
                    valid = False
                    break
                df_slice = df.iloc[pos - win_len + 1: pos + 1]
                if len(df_slice) < win_len:
                    valid = False
                    break
                stack.append(df_slice.values)
            if not valid:
                break

        if not valid or len(stack) != len(INTERVAL_MINUTES) * 2:
            continue

        x = np.concatenate(stack, axis=1)
        if ref_shape is None:
            ref_shape = x.shape[1]
        if x.shape[1] != ref_shape:
            continue

        features.append(x)
        label_df = target_df.iloc[i:i+2]
        try:
            label = label_fn(label_df, threshold=threshold).iloc[0]
            labels.append(label)
        except Exception as e:
            print(f"❌ 라벨 생성 실패: {e}")
            continue

    X = np.stack(features, axis=0) if features else None
    y = np.array(labels) if labels else None
    return X, y

def build_generic_dataset(interval: str, label_mode='binary'):
    global TARGET_INTERVAL
    TARGET_INTERVAL = interval

    X_all, y_all = [], []
    expected_dim = None

    for symbol in SYMBOL_LIST:
        print(f"📡 [{symbol} / {interval}] 데이터셋 생성 중...")
        X, y = build_lstm_dataset(symbol, label_mode=label_mode)

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
