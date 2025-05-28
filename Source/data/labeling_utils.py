import pandas as pd
import numpy as np

def label_binary(df: pd.DataFrame, threshold: float = 0.002) -> pd.Series:
    """
    이진 분류 라벨: 상승(1), 하락 또는 횡보(0)
    기준: 다음 캔들의 종가가 현재 종가 대비 threshold 이상 상승하면 1, 아니면 0
    """
    returns = df['Close'].shift(-1) / df['Close'] - 1
    return (returns > threshold).astype(int)

def label_three_class(df: pd.DataFrame, threshold: float = 0.002) -> pd.Series:
    """
    삼분류 라벨: 하락(0), 횡보(1), 상승(2)
    기준: 수익률이 -threshold, +threshold 기준으로 구간 분류
    """
    returns = df['Close'].shift(-1) / df['Close'] - 1
    labels = pd.Series(np.where(returns > threshold, 2,
                        np.where(returns < -threshold, 0, 1)), index=df.index)
    return labels

def label_position_class(df: pd.DataFrame, threshold: float = 0.002) -> pd.Series:
    """
    다중 클래스 포지션 라벨링 (5등급)
    0: 강매도, 1: 약매도, 2: 보유, 3: 약매수, 4: 강매수
    """
    returns = df['Close'].shift(-1) / df['Close'] - 1
    labels = pd.Series(np.digitize(returns, [-0.01, -0.004, 0.004, 0.01]), index=df.index)
    return labels

# [TODO] 회귀용 수익률 라벨
def label_return_regression(df: pd.DataFrame) -> pd.Series:
    """
    회귀용 수익률 라벨. 다음 종가 대비 수익률
    """
    returns = df['Close'].shift(-1) / df['Close'] - 1
    return returns
