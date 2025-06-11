from datetime import datetime

### 1. 심볼 설정 ###
SYMBOL_LIST = [
    'NVDA', 'TSLL', 'TSLA', 'PLTR', 'RGTI', 'INTC', 'PLUG', 'TQQQ', 'SMCI', 'SOFI',
    'AAL', 'SOUN', 'IBIT', 'AMZN', 'QQQ', 'MARA', 'AMD', 'WULF', 'WBD', 'GOOGL',
    'GRAB', 'RIVN', 'AVGO', 'RIOT', 'LAES', 'IQ', 'QUBT', 'BITF', 'AGNC', 'CLSK',
    # 'CMCSA', 'JBLU', 'NVDL', 'GOOG', 'RXRX', 'CRWV', 'ERIC', 'CSCO', 'HBAN', 'NVNI',
    # 'WBA', 'AMDL', 'RKLB', 'LYFT', 'MRVL', 'MSTR', 'NITO', 'QSI', 'ONCO', 'BTBT',
    # 'RR', 'CSX', 'CIFR', 'PTON', 'BLMZ', 'VTRS', 'GERN', 'RUN', 'ASST', 'TEM',
    # 'VOD', 'LUNR', 'FFAI', 'PTEN', 'HTZ', 'ETHA', 'MRNA', 'PYPL', 'SBUX', 'MSTX',
    # 'HIVE', 'SHOP', 'DKNG', 'LXRX', 'PACB', 'COIN', 'MDLZ', 'HST', 'DEVS', 'OMEX',
    # 'ONDS', 'FAAS', 'CELH', 'NWL', 'TTD', 'VLY', 'GILD', 'PARA', 'IVVD', 'QCOM',
    # 'SERV', 'APA', 'UAL', 'INVZ', 'SLXN', 'RZLV', 'IBRX', 'LTRY', 'BKR', 'AFRM',
    # 'RCAT', 'CGC', 'GRYP', 'APP', 'VCIT', 'CLRB', 'AMAT', 'RDFN', 'CCCS', 'QYLD',
    # 'BND', 'PONY', 'GTM', 'FAST', 'REKR', 'SHLS', 'NVAX', 'PMAX', 'JEPQ', 'EMB',
    # 'STNE', 'BTDR', 'SPRC', 'CONL', 'MVIS', 'UPST', 'HBIO', 'TIGR', 'TMC', 'MNST',
    # 'HUT', 'ARBE', 'GT', 'ENVX', 'BNAI', 'ARRY', 'KITT', 'ROIV', 'IMNN', 'ARM',
    # 'PRPH', 'CFLT', 'AZN', 'ABP', 'BILI', 'LI', 'TSCO', 'SABR', 'PANW', 'ABNB',
    # 'CZR', 'COMM', 'ESPR', 'PDBC', 'CRDO', 'APPS', 'FLNC', 'HOTH', 'EVGO', 'GCTK',
    # 'SIRI', 'MTCH', 'ARCC', 'DDOG', 'PHIO', 'MBLY', 'PDYN', 'FTNT', 'DASH', 'GEVO',
    # 'FITB', 'SYTA', 'AIFF', 'PSNY', 'SIDU', 'BMGL', 'ALAB', 'HSAI', 'RPRX', 'HUMA',
    # 'SGBX', 'BIDU', 'VCSH', 'BZ', 'SEDG', 'AMPG', 'RUM', 'DLTR', 'HON', 'NFLX',
    # 'ALLO', 'CART', 'ABSI', 'PENN', 'MRSN', 'SHY', 'GORV', 'GEN', 'CEG', 'VKTX',
    # 'FOXA', 'CRNC', 'MAT', 'PFF', 'IPA', 'MLCO', 'GEHC', 'XRX', 'CRWD', 'BNDX',
    # 'ADBE', 'DXCM', 'SNDK', 'WATT', 'KOPN', 'NNE', 'PAA', 'DBX', 'INDI', 'CHX',
    # 'AAOI', 'ECX', 'BLNK', 'BCRX', 'EXE', 'SWTX', 'NAKA', 'SLDP', 'CTSH', 'SWKS',
    # 'NTLA', 'TCOM', 'EA', 'GWAV', 'ABAT', 'FSLR', 'VGSH', 'NXTT', 'AREC', 'SMCX',
    # 'PAYO', 'ULCC', 'IGSB', 'KC', 'FRSH', 'REAL', 'VRME', 'VSAT', 'VVPR', 'MBOT',
    # 'TSLR', 'ENTG', 'CTMX', 'VCLT', 'PLTU', 'CSGP', 'AMGN', 'NWSA', 'MCHI', 'TER',
    # 'TXG', 'GRRR', 'ONB', 'CHSN', 'GOGL', 'CAPT', 'NXPI', 'MBB', 'EDIT', 'BMBL',
    # 'HOLX', 'CISO', 'IUSB', 'ADMA', 'XRAY', 'BYND', 'GTLB', 'UROY', 'HIMX', 'QRVO',
    # 'GNPX', 'BGC', 'VMBS', 'GDS', 'TRIP', 'VERU', 'TSHA', 'QQQM', 'WSC', 'CG',
    # 'SLM', 'PCAR', 'SXTC', 'EXEL', 'SFIX', 'CLNE', 'Z', 'DFDV', 'WDAY', 'SSRM',
]
INDEX_SYMBOL = "^IXIC"

### 2. 날짜 범위 ###
START_DATE = "2025-05-01"
END_DATE = "2025-06-01"

### 3. 타임프레임 설정 ###
TARGET_COLUMN = "close"
INTERVALS = ["5m", "15m", "30m", "60m", "1d"]

REQUIRED_LENGTH = {
    "5m": 128,
    "15m": 64,
    "30m": 32,
    "60m": 16,
    "1d": 8,
}

### 4. 라벨링 설정 ###
LABEL_THRESHOLDS = {
    "5m": 0.005,
    "15m": 0.007,
    "30m": 0.01,
    "60m": 0.012,
    "1d": 0.015,
}

### 5. 기술 지표 목록 (현재 사용 중: 4개) ###
#TECHNICAL_INDICATORS = [  
    # "rsi", "macd",
    # "boll_upper", "boll_lower",
    # "volume_change",
    
    # "sma5", "sma20",
    # "ema12", "ema26",
    # "stoch_k", "stoch_d",
    # "cci", "adx", "stoch_rsi"
#]

### 6. 기술 지표 파라미터 ###
#TECHNICAL_PARAMS = {
    # "rsi": {"window": 14},
    
    # "macd": {"fast": 12, "slow": 26, "signal": 9},
    # "boll": {"window": 20, "std": 2},
    # "stoch": {"k_window": 14, "d_window": 3},
    
    # "sma5": {"window": 5},
    # "sma20": {"window": 20},
    # "ema12": {"span": 12},
    # "ema26": {"span": 26},
    # "cci": {"window": 20},
    # "adx": {"window": 14},
    # "stoch_rsi": {"window": 14}
#}

### 7. 저장 경로 ###
DATA_PATH = "./cache/"
