# _data_config.py

### 1. 심볼 설정 ###
SYMBOL_LIST = ["AAPL", "MSFT", "GOOG"]
INDEX_SYMBOL = "^IXIC"  # 나스닥 종합지수 (필요 시 사용)

### 2. 데이터 수집 기간 ###
START_DATE = "2024-04-01"
END_DATE = "2024-05-27"

### 3. 대상 분봉 및 컬럼 ###
TARGET_INTERVAL = "30m"  # 주 예측 타임프레임
TARGET_COLUMN = "Close"

### 4. 라벨링 설정 ###
LABELING_MODE = "binary"  # ["binary", "three_class", "position", "regression"]
LABEL_THRESHOLDS = {
    "2m": 0.003,
    "5m": 0.005,
    "15m": 0.007,
    "30m": 0.01,
    "60m": 0.012,
    "1d": 0.015,
}

### 5. 분봉 단위 (분 단위 환산) ###
INTERVAL_MINUTES = {
    "2m": 2,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "1d": 1440,
}

### 6. 학습 윈도우 동기화 기준 ###
WINDOW_MINUTES = 60
REQUIRED_LENGTH = {k: WINDOW_MINUTES // v for k, v in INTERVAL_MINUTES.items()}

### 7. 기술 지표 목록 ###
TECHNICAL_INDICATORS = [
    "rsi",
    "macd",
    "sma5",
    "sma20",
    "ema12",
    "ema26",
    "boll_upper",
    "boll_lower",
    "stoch_k",      # 주석 처리 가능한 확장 옵션
    "stoch_d",
    # "cci",
    # "adx",
    # "stoch_rsi"
]

# 기술 지표 파라미터 별도 딕셔너리 구성 (예정 시)
# TECHNICAL_PARAMS = {
#     "rsi": {"window": 14},
#     "macd": {"fast": 12, "slow": 26, "signal": 9},
#     ...
# }

### 8. 데이터 저장 경로 ###
DATA_PATH = "./data/cache/"  # 추후 구조화 가능
