import pandas as pd
import ta
from _data_config import TECHNICAL_INDICATORS, TECHNICAL_PARAMS

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    지정된 기술 지표를 계산하여 DataFrame에 추가합니다.
    NaN이 포함된 행은 제거하여 후속 처리에서 오류를 방지합니다.
    """

    df = df.copy()

    # ✅ 컬럼 이름 정규화
    df.columns = [col.lower() for col in df.columns]

    # ✅ 필수 컬럼 확인
    required_cols = ["close", "volume", "high", "low"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"필수 컬럼 '{col}'이(가) 누락되었습니다.")

    # ✅ 숫자형 변환
    df = df.apply(pd.to_numeric, errors="coerce")

    try:
        # ✅ RSI
        if "rsi" in TECHNICAL_INDICATORS:
            window = TECHNICAL_PARAMS.get("rsi", {}).get("window", 14)
            df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=window).rsi()

        # ✅ MACD
        if "macd" in TECHNICAL_INDICATORS:
            fast = TECHNICAL_PARAMS.get("macd", {}).get("fast", 12)
            slow = TECHNICAL_PARAMS.get("macd", {}).get("slow", 26)
            signal = TECHNICAL_PARAMS.get("macd", {}).get("signal", 9)
            macd = ta.trend.MACD(close=df["close"], window_slow=slow, window_fast=fast, window_sign=signal)
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()

        # ✅ Bollinger Bands
        if "boll_upper" in TECHNICAL_INDICATORS or "boll_lower" in TECHNICAL_INDICATORS:
            window = TECHNICAL_PARAMS.get("boll", {}).get("window", 20)
            std = TECHNICAL_PARAMS.get("boll", {}).get("std", 2)
            bb = ta.volatility.BollingerBands(close=df["close"], window=window, window_dev=std)
            if "boll_upper" in TECHNICAL_INDICATORS:
                df["boll_upper"] = bb.bollinger_hband()
            if "boll_lower" in TECHNICAL_INDICATORS:
                df["boll_lower"] = bb.bollinger_lband()

        # ✅ 거래량 변화율
        if "volume_change" in TECHNICAL_INDICATORS:
            df["volume_change"] = df["volume"].pct_change().fillna(0)

        # 💤 확장 가능 지표 (필요시 아래 주석 해제)
        # if "sma5" in TECHNICAL_INDICATORS:
        #     window = TECHNICAL_PARAMS.get("sma5", {}).get("window", 5)
        #     df["sma5"] = df["close"].rolling(window=window).mean()

        # if "ema12" in TECHNICAL_INDICATORS:
        #     span = TECHNICAL_PARAMS.get("ema12", {}).get("span", 12)
        #     df["ema12"] = df["close"].ewm(span=span, adjust=False).mean()

        # if "stoch_k" in TECHNICAL_INDICATORS or "stoch_d" in TECHNICAL_INDICATORS:
        #     k_win = TECHNICAL_PARAMS.get("stoch", {}).get("k_window", 14)
        #     d_win = TECHNICAL_PARAMS.get("stoch", {}).get("d_window", 3)
        #     stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=k_win, smooth_window=d_win)
        #     if "stoch_k" in TECHNICAL_INDICATORS:
        #         df["stoch_k"] = stoch.stoch()
        #     if "stoch_d" in TECHNICAL_INDICATORS:
        #         df["stoch_d"] = stoch.stoch_signal()

    except Exception as e:
        print(f"[지표 계산 오류] {e}")
        return pd.DataFrame()  # 오류 발생 시 빈 DataFrame 반환

    df = df.dropna()
    return df
