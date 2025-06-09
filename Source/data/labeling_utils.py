import numpy as np
import pandas as pd

def label_trend(df):
    """주가 변화율을 실수로 반환 (종가 기준)"""
    current = df["close"].values[0]
    future = df["close"].values[-1]
    return np.array([(future - current) / current])

def label_regression(df, n=1):
    """N시점 뒤의 변화율"""
    current = df["close"].values[0]
    if len(df["close"].values) <= n:
        return np.array([0.0])
    future = df["close"].values[n]
    return np.array([(future - current) / current])

def label_candle_regression(df, n=1):
    """N시점 뒤의 open/high/low/close"""
    if len(df) <= n:
        o, h, l, c = df.iloc[-1].loc[["open","high","low","close"]]
    else:
        o, h, l, c = df.iloc[n].loc[["open","high","low","close"]]
    return np.array([o, h, l, c])

def label_highest(df, n=1):
    """
    n개 구간 내 high의 최대값 반환
    """
    if len(df) < n + 1:
        return np.max(df["high"].values)
    return np.max(df["high"].values[:n+1])

def label_position(df, threshold=0.01):
    """
    종가 변화율 기준으로 [-1, 1] 실수값 추천.
    -1: 강매도, 0: 관망, 1: 강매수
    """
    current = df["close"].values[0]
    future = df["close"].values[-1]
    change = (future - current) / current
    # 매수/매도 비중은 비선형 scaling도 가능, 우선은 단순 선형
    # threshold 기준 미만은 0(관망), 이상은 비율로 환산
    if abs(change) < threshold:
        return np.array([0.0])  # 관망
    return np.array([np.clip(change, -1.0, 1.0)])


def get_all_labels(df, threshold=0.01, n=100):
    return {
        "trend": label_trend(df)[0],
        "regression": label_regression(df, n=n)[0],
        "candle": label_candle_regression(df, n=n),
        "highest": label_highest(df, n=n),
        "position": label_position(df, threshold=threshold)[0],
    }
