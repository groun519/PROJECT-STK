# 예측 기준 분봉
TARGET_INTERVAL = "30m"

# 학습 대상 종목 리스트
SYMBOL_LIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "ADBE", "AVGO",
    "COST", "PEP", "CSCO", "INTC", "CMCSA", "NFLX", "AMD", "QCOM", "TXN", "AMGN",
    "INTU", "SBUX", "ISRG", "ADI", "MDLZ", "BKNG", "GILD", "MU", "LRCX", "VRTX",
    "REGN", "KHC", "ADP", "MAR", "LULU", "EA", "BIIB", "MNST",
    "ROST", "CDNS", "CTAS", "DLTR", "EXC", "ILMN", "MCHP", "NXPI", "ORLY", "PAYX",
    "PYPL", "SIRI", "SNPS", "TMUS", "VRSK", "WBA", "XEL", "ZS", "ABNB", "ANSS",
    "ASML", "AZN", "BKR", "CHTR", "CSGP", "DDOG", "DXCM", "FAST", "FTNT", "GEHC",
    "HON", "IDXX", "KLAC", "LIN", "MELI", "MRVL", "MSTR", "ODFL", "ON", "PANW",
    "PDD", "PLTR", "SHOP", "TEAM", "TTWO", "TTD", "WDAY", "WBD", "ARM", "APP",
    "AXON", "CEG", "CPRT", "CSX", "GFS", "KDP", "PCAR", "ROP", "TTD", "VRSK"
]


# 나스닥 시장 지수
INDEX_SYMBOL = "^IXIC"

# 데이터 수집 기간
START_DATE = "2025-04-01"
END_DATE = "2025-05-01"

MARKET_MODEL_PATH = "models/market/lstm_market.pt"
