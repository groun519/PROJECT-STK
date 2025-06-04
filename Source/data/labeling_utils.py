import numpy as np

def label_binary(df, threshold):
    # 상승/하락 구분
    future = df["close"].values[-1]
    current = df["close"].values[0]
    change = (future - current) / current
    return np.array([1 if abs(change) > threshold else 0])

def label_three_class(df, threshold):
    future = df["close"].values[-1]
    current = df["close"].values[0]
    change = (future - current) / current
    if change > threshold:
        return np.array([2])  # 상승
    elif change < -threshold:
        return np.array([0])  # 하락
    else:
        return np.array([1])  # 관망

def label_position(df, threshold):
    # 단순 예시: 포지션 [매수/관망/매도]
    return label_three_class(df, threshold)

def label_regression(df, threshold=None):
    # 수익률 예측 (정규화)
    future = df["close"].values[-1]
    current = df["close"].values[0]
    change = (future - current) / current
    return np.array([change])

def label_candle_regression(df, threshold=None):
    o = df["open"].values[-1]
    h = df["high"].values[-1]
    l = df["low"].values[-1]
    c = df["close"].values[-1]
    return np.array([o, h, l, c])

def get_all_labels(df, threshold):
    return {
        "binary": label_binary(df, threshold)[0],
        "three_class": label_three_class(df, threshold)[0],
        "position": label_position(df, threshold)[0],
        "regression": label_regression(df)[0],
        "candle": label_candle_regression(df),
    }
