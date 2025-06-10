import numpy as np
import pandas as pd
from data.data_fetcher import load_multitimeframe_data
from data.labeling_utils import get_all_labels
from data._data_config import (
    SYMBOL_LIST, 
    INTERVALS, REQUIRED_LENGTH, LABEL_THRESHOLDS
)

def get_threshold(interval):
    return LABEL_THRESHOLDS.get(interval, 0.01)

def build_lstm_dataset(symbol, interval : str):
    mtf_data = load_multitimeframe_data(symbol)
    if not mtf_data["stock"] or not mtf_data["index"]:
        print(f"❌ {symbol}: 데이터 로딩 실패 (stock or index)")
        return None, None

    if interval not in mtf_data["stock"]:
        print(f"❌ {symbol}: {interval} 분봉 데이터가 없습니다.")
        return None, None

    target_df = mtf_data["stock"][interval]
    threshold = get_threshold(interval)

    features, labels = [], []
    ref_shape = None

    for i in range(REQUIRED_LENGTH[interval], len(target_df) - 1):
        anchor_time = target_df.index[i]
        stack, valid = [], True


        win_len = REQUIRED_LENGTH[interval]
        for key in ["stock", "index"]:
            df = mtf_data[key].get(interval)
            # if df is None or len(df) < win_len:
            #     print(f"[{symbol}][{interval}][{key}] 데이터 없음 or 부족, 건너뜀")
            #     valid = False
            #     break

            # ✅ 기술지표 누락 체크
            # missing_cols = [col for col in TECHNICAL_INDICATORS if col not in df.columns]
            # if missing_cols:
            #     print(f"[{symbol}][{interval}][{key}] 누락된 기술지표 칼럼: {missing_cols}")
            #     valid = False
            #     break
                
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            if anchor_time.tz is None:
                anchor_time = anchor_time.tz_localize("UTC")
            pos = df.index.get_indexer([anchor_time], method="nearest")[0]
                
            if pos == -1 or pos < win_len:
                valid = False
                break
            df_slice = df.iloc[pos - win_len + 1 : pos + 1]
            if len(df_slice) < win_len:
                valid = False
                break
            stack.append(df_slice.values)

        if not valid:
            break
        
        # if not valid or len(stack) != len(INTERVAL_MINUTES) * 2:
        #     continue
        
        x = np.concatenate(stack, axis=1)
        
        if ref_shape is None:
            ref_shape = x.shape[1]
        if x.shape[1] != ref_shape:
            continue
        
        features.append(x)
        label_df = target_df.iloc[i:i+2]
        
        try:
            label_dict = get_all_labels(label_df, threshold)
            labels.append(label_dict)
        except Exception as e:
            print(f"❌ 라벨 생성 실패: {e}")
            continue
    
    X = np.stack(features, axis=0) if features else None
    y = labels if labels else None
    
    return X, y

def build_generic_dataset(interval: str):

    X_all, y_all = [], []
    expected_dim = None

    for symbol in SYMBOL_LIST:
        print(f"📡 [{symbol} / {interval}] 데이터셋 생성 중...")
        X, y = build_lstm_dataset(symbol, interval)

        if X is None or y is None:
            continue
        if expected_dim is None:
            expected_dim = X.shape[2]
        if X.shape[2] != expected_dim:
            print(f"⚠️ {symbol}: 차원 불일치 → 건너뜀")
            continue

        X_all.append(X)
        y_all.extend(y)

    if not X_all:
        print(f"❌ {interval} 기준 학습 가능한 데이터 없음")
        return None, None

    X = np.concatenate(X_all, axis=0)
    y = y_all
    print(f"✅ {interval} 기준 총 샘플 수: {X.shape[0]}")
    return X, y
