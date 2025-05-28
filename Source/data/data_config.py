# data_config.py

# 사용할 분봉 목록
INTERVALS = ["2m", "5m", "15m", "30m", "60m", "1d"]

# 분 단위 변환
INTERVAL_MINUTES = {
    "2m": 2,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "1d": 1440,
}

# 기준 시간 윈도우 (ex: 60분)
WINDOW_MINUTES = 60

# 타임프레임별 필요한 캔들 수
REQUIRED_LENGTH = {k: WINDOW_MINUTES // v for k, v in INTERVAL_MINUTES.items()}

# 수익률 라벨링 기준
LABEL_THRESHOLDS = {
    "2m": 0.003,
    "5m": 0.005,
    "15m": 0.007,
    "30m": 0.010,
    "60m": 0.012,
    "1d": 0.015,
}

# 캐시 파일 저장 위치
CACHE_DIR = "cache"

# 기술지표 컬럼 이름 (향후 시각화 등에서 사용 가능)
DEFAULT_FEATURES = [
    "rsi", "macd", "ema20", "bb_upper", "bb_lower", "volume_change"
]
