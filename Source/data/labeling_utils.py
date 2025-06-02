import pandas as pd
import numpy as np

### 1. 이진 분류 (binary)
def label_binary(df: pd.DataFrame, threshold: float = 0.002) -> pd.Series:
    """
    상승(1), 하락 또는 횡보(0)
    기준: 다음 캔들의 종가가 현재 종가 대비 threshold 이상 상승하면 1, 아니면 0
    """
    returns = df['Close'].shift(-1) / df['Close'] - 1
    return (returns > threshold).astype(int)

### 2. 삼분류 (three_class)
def label_three_class(df: pd.DataFrame, threshold: float = 0.002) -> pd.Series:
    """
    하락(0), 횡보(1), 상승(2)
    기준: 수익률이 -threshold, +threshold 기준으로 구간 분류
    """
    returns = df['Close'].shift(-1) / df['Close'] - 1
    labels = pd.Series(np.where(returns > threshold, 2,
                        np.where(returns < -threshold, 0, 1)), index=df.index)
    return labels

### 3. 포지션 분류 (position)
def label_position_class(df: pd.DataFrame) -> pd.Series:
    """
    포지션 라벨링 (5등급)
    0: 강매도, 1: 약매도, 2: 보유, 3: 약매수, 4: 강매수
    기준: 수익률 구간에 따라 등급 지정
    """
    returns = df['Close'].shift(-1) / df['Close'] - 1
    bins = [-np.inf, -0.01, -0.004, 0.004, 0.01, np.inf]
    labels = pd.Series(np.digitize(returns, bins) - 1, index=df.index)
    return labels

### 4. 회귀용 라벨 (regression)
def label_return_regression(df: pd.DataFrame) -> pd.Series:
    """
    다음 캔들 기준 수익률 반환 (float)
    """
    returns = df['Close'].shift(-1) / df['Close'] - 1
    return returns

### 5. 라벨링 자동 적용 함수
def apply_labeling(df: pd.DataFrame, mode: str, threshold: float = 0.002) -> pd.Series:
    """
    라벨링 모드에 따라 자동으로 라벨 적용
    - mode: "binary", "three_class", "position", "regression"
    - threshold: binary/three_class용 수익률 기준
    """
    if mode == "binary":
        return label_binary(df, threshold)
    elif mode == "three_class":
        return label_three_class(df, threshold)
    elif mode == "position":
        return label_position_class(df)
    elif mode == "regression":
        return label_return_regression(df)
    else:
        raise ValueError(f"[오류] 지원하지 않는 라벨링 모드: {mode}")
