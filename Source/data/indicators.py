import pandas as pd
import ta
from _data_config import TECHNICAL_INDICATORS, TECHNICAL_PARAMS

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    지정된 기술 지표를 계산하여 DataFrame에 추가합니다.
    NaN이 포함된 행은 제거하여 후속 처리에서 오류를 방지합니다.
    """

    df = df.copy()

    # ✅ 컬럼 이름 정규화 (전부 소문자)
    df.columns = [col.lower() for col in df.columns]

    # ✅ 필수 컬럼 존재 여부 확인
    required_cols = ["close", "volume", "high", "low"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"필수 컬럼 '{col}'이(가) 누락되었습니다.")

    # ✅ 숫자형 변환
    df = df.apply(pd.to_numeric, errors="coerce")

    try:
        # RSI
        if "rsi" in TECHNICAL_INDICATORS:
            window = TECHNICAL_PARAMS.get("rsi", {}).get("window", 14)
            df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=window).rsi()

        # MACD
        if "macd" in TECHNICAL_INDICATORS:
            fast = TECHNICAL_PARAMS.get("macd", {}).get("fast", 12)
            slow = TECHNICAL_PARAMS.get("macd", {}).get("slow", 26)
            signal = TECHNICAL_PARAMS.get("macd", {}).get("signal", 9)
            macd = ta.trend.MACD(close=df["close"], window_slow=slow, window_fast=fast, window_sign=signal)
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()

        # SMA
        if "sma5" in TECHNICAL_INDICATORS:
            window = TECHNICAL_PARAMS.get("sma5", {}).get("window", 5)
            df["sma5"] = df["close"].rolling(window=window).mean()

        # 볼린저 밴드
        if "bollinger" in TECHNICAL_INDICATORS:
            window = TECHNICAL_PARAMS.get("bollinger", {}).get("window", 20)
            stddev = TECHNICAL_PARAMS.get("bollinger", {}).get("std", 2)
            bb = ta.volatility.BollingerBands(close=df["close"], window=window, window_dev=stddev)
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_lower"] = bb.bollinger_lband()

        # 거래량 변화율
        if "volume_change" in TECHNICAL_INDICATORS:
            df["volume_change"] = df["volume"].pct_change().fillna(0)

        # ✅ 필요한 지표 확장 가능
        # if "cci" in TECHNICAL_INDICATORS:
        #     df["cci"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
        # if "adx" in TECHNICAL_INDICATORS:
        #     df["adx"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["close"], window=14).adx()
        # if "stoch_rsi" in TECHNICAL_INDICATORS:
        #     df["stoch_rsi"] = ta.momentum.StochRSIIndicator(df["close"]).stochrsi_k()
    
    except Exception as e:
        print(f"[지표 계산 오류] {e}")
        return pd.DataFrame()  # 오류 발생 시 빈 데이터프레임 반환

    # ✅ NaN 행 제거
    df = df.dropna()

    return df

