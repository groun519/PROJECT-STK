from datetime import datetime

### 1. 심볼 설정 ###
SYMBOL_LIST = ["AAPL", "MSFT"]
INDEX_SYMBOL = "QQQ"

### 2. 날짜 범위 ###
START_DATE = "2025-05-01"
END_DATE = "2025-05-20"

### 3. 타임프레임 설정 ###
TARGET_INTERVAL = "30m"
TARGET_COLUMN = "close"
INTERVAL_MINUTES = {
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "1d": 1440,
}

# 🔧 3-1. 날짜 기반 윈도우 수 자동 계산
def get_dynamic_required_length(index_symbol, interval, start, end, margin_days=5):
    import yfinance as yf
    df = yf.download(index_symbol, start=start, end=end, interval=interval, progress=False)
    n_rows = len(df)
    if interval == "1d":
        margin = margin_days
    else:
        minutes = int(interval.replace("m", ""))
        margin = margin_days * (390 // minutes)
    winlen = max(30, n_rows - margin)
    return winlen

REQUIRED_LENGTH = {
    "5m": get_dynamic_required_length(INDEX_SYMBOL, "5m", START_DATE, END_DATE, margin_days=3),
    "15m": get_dynamic_required_length(INDEX_SYMBOL, "15m", START_DATE, END_DATE, margin_days=3),
    "30m": get_dynamic_required_length(INDEX_SYMBOL, "30m", START_DATE, END_DATE, margin_days=3),
    "60m": get_dynamic_required_length(INDEX_SYMBOL, "60m", START_DATE, END_DATE, margin_days=3),
    "1d": get_dynamic_required_length(INDEX_SYMBOL, "1d", START_DATE, END_DATE, margin_days=3),
}


### 4. 라벨링 설정 ###
LABELING_MODE = "binary"  # binary, three_class, position, regression
LABEL_THRESHOLDS = {
    "5m": 0.005,
    "15m": 0.007,
    "30m": 0.01,
    "60m": 0.012,
    "1d": 0.015,
}

### 5. 기술 지표 목록 (현재 사용 중: 4개) ###
TECHNICAL_INDICATORS = [
    "rsi", "macd",
    "boll_upper", "boll_lower",
    "volume_change",
    # "sma5", "sma20",
    # "ema12", "ema26",
    # "stoch_k", "stoch_d",
    # "cci", "adx", "stoch_rsi"
]

### 6. 기술 지표 파라미터 ###
TECHNICAL_PARAMS = {
    "rsi": {"window": 14},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "boll": {"window": 20, "std": 2},
    "stoch": {"k_window": 14, "d_window": 3},
    # "sma5": {"window": 5},
    # "sma20": {"window": 20},
    # "ema12": {"span": 12},
    # "ema26": {"span": 26},
    # "cci": {"window": 20},
    # "adx": {"window": 14},
    # "stoch_rsi": {"window": 14}
}

### 7. 저장 경로 ###
DATA_PATH = "./cache/"
