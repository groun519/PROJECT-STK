# _data_config.py

from datetime import datetime

### 1. 심볼 설정 ###
SYMBOL_LIST = ["AAPL", "MSFT", "GOOG"]
INDEX_SYMBOL = "^IXIC"

### 2. 날짜 범위 ###
START_DATE = "2025-04-01"
END_DATE = "2025-05-01"

### 3. 타임프레임 설정 ###
TARGET_INTERVAL = "30m"
TARGET_COLUMN = "close"
INTERVAL_MINUTES = {
    "2m": 2,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "1d": 1440,
}

# 🔧 날짜 기반 윈도우 수 자동 계산
def compute_required_length(start_date: str, end_date: str, interval_dict: dict):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_minutes = (end_dt - start_dt).days * 24 * 60
    return {k: total_minutes // v for k, v in interval_dict.items()}

REQUIRED_LENGTH = compute_required_length(START_DATE, END_DATE, INTERVAL_MINUTES)

### 4. 라벨링 설정 ###
LABELING_MODE = "binary"  # binary, three_class, position, regression
LABEL_THRESHOLDS = {
    "2m": 0.003,
    "5m": 0.005,
    "15m": 0.007,
    "30m": 0.01,
    "60m": 0.012,
    "1d": 0.015,
}

### 5. 기술 지표 목록 ###
TECHNICAL_INDICATORS = [
    "rsi", "macd", "sma5", "sma20",
    "ema12", "ema26",
    "boll_upper", "boll_lower",
    "stoch_k", "stoch_d"
    # "cci", "adx", "stoch_rsi" ← 확장 후보
]

### 6. 기술 지표 파라미터 ###
TECHNICAL_PARAMS = {
    "rsi": {"window": 14},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "sma5": {"window": 5},
    "sma20": {"window": 20},
    "ema12": {"span": 12},
    "ema26": {"span": 26},
    "boll": {"window": 20, "std": 2},
    "stoch": {"k_window": 14, "d_window": 3},
    # "cci": {"window": 20},
    # "adx": {"window": 14},
    # "stoch_rsi": {"window": 14}
}

### 7. 저장 경로 ###
DATA_PATH = "./data/cache/"
