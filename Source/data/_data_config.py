from datetime import datetime

### 1. 심볼 설정 ###
SYMBOL_LIST = [
    'TSLL'         'PLTR'         'INTC'         'SMCI'         'SOFI'         'IBIT'         'MARA'         'GRAB'         'IQ'           'BITF'    ,
    'CLSK'         'ERIC'         'NVNI'         'LYFT'         'MRVL'         'RR'           'PTON'         'VTRS'         'VOD'          'PTEN'    ,
    'SBUX'         'DKNG'         'LXRX'         'MDLZ'         'OMEX'         'CELH'         'TTD'          'IVVD'         'SLXN'         'RZLV'    ,
    'LTRY'         'BKR'          'AFRM'         'GRYP'         'AMAT'         'CCCS'         'BND'          'GTM'          'FAST'         'STNE'    ,
    'CONL'         'HBIO'         'HUT'          'GT'           'PRPH'         'ESPR'         'FLNC'         'GCTK'         'MTCH'         'PHIO'    ,
    'FTNT'         'GEVO'         'SYTA'         'HSAI'         'BIDU'         'VCSH'         'BZ'           'AMPG'         'GEN'          'CRNC'    ,
    'PFF'          'IPA'          'MLCO'         'GEHC'         'ADBE'         'DXCM'         'DBX'          'INDI'         'BCRX'         'EXE'     ,
    'CTSH'         'SWKS'         'TCOM'         'EA'           'GWAV'         'VGSH'         'AREC'         'ULCC'         'IGSB'         'KC'      ,
    'VVPR'         'TSLR'         'VCLT'         'ONB'          'CHSN'         'GOGL'         'HOLX'         'IUSB'         'ADMA'         'XRAY'    ,
    'UROY'         'GNPX'         'TSHA'         'QQQM'         'WSC'          'SXTC'         'LSCC'         'NTNX'         'SVC'          'TTEK'    ,
    'ASPI'         'SMTC'         'MURA'         'INTR'         'LNZA'         'AKAM'         'BTAI'         'ILMN'         'SAIL'         'SBRA'    ,
    'TRVI'         'IRBT'         'ACAD'         'COST'         'PCSA'         'DWTX'         'URBN'         'FIVE'         'FGEN'         'CCEP'    ,
    'BDTX'         'ARVN'         'EXPE'         'RAPT'         'AMKR'         'LTBR'         'INEO'         'BON'          'EVLV'         'BRKR'    ,
    'SOPA'         'TTWO'         'LZ'           'INOD'         'RETO'         'IGIB'         'HAS'          'ACB'          'ADPT'         'IBKR'    ,
    'CSIQ'         'TROW'         'TECH'         'WKEY'         'GPRE'         'YYAI'         'ODFL'         'BTSG'         'COLB'         'PAGP'    ,
    'MAR'          'ALKS'         'ZION'         'GGLL'         'ASML'         'BIIB'         'CAPR'         'TVGN'         'GLNG'         'PGNY'    ,
    'THTX'         'GOGO'         'OPTX'         'QURE'         'WB'           'NTRA'         'NLSP'         'ENVB'         'VNOM'         'SCNX'    ,
    'KURA'         'PINC'         'OCUL'         'GDRX'         'CCCC'         'BSY'          'NTRS'         'LFMD'         'CDW'          'WFRD'    ,
    'OS'           'PZZA'         'KROS'         'MREO'         'ATAT'         'HKPD'         'UPXI'         'AVDL'         'ADGM'         'SNPS'    ,
    'CHTR'         'GLMD'         'NKTX'         'WVE'          'VERA'         'CAR'          'CLMT'         'MKSI'         'EXLS'         'SHC'     ,
    'HLIT'         'SRRK'         'REG'          'SMCL'         'ARBK'         'SXTP'         'IAS'          'ALKT'         'CGON'         'DLO'     ,
    'NMRK'         'CARG'         'CMPO'         'LILAK'        'SGLY'         'BRY'          'MAPS'         'ADTN'         'JAZZ'         'NVCR'    ,
    'RMBS'         'TRMD'         'SSP'          'VIRT'         'PTCT'         'BABX'         'STAA'         'ASTL'         'LBRDK'        'XNET'    ,
    'SLNO'         'GTEC'         'MRUS'         'RARE'         'ABUS'         'CPIX'         'DH'           'LNW'          'ANAB'         'CRGX'    ,
    'AMLX'         'ALNY'         'CDXS'         'NOTV'         'DRS'          'MGIH'         'IMTX'         'SEIC'         'FHB'          'OCSL'    ,
    'MIRA'         'VRNT'         'VONV'         'CVBF'         'RDNT'         'GCL'          'CNXC'         'ZDAI'         'FTDR'         'LAUR'    ,
    'CHEK'         'IDXX'         'NWS'          'CVAC'         'ATOS'         'PCLA'         'ILAG'         'AGIO'         'RGEN'         'AGEN'    ,
    'DOX'          'XMTR'         'COCO'         'ANGI'         'ACET'         'PI'           'OTLK'         'ALLT'         'HRMY'         'DGRW'    ,
    'EFOI'         'ODP'          'MLKN'         'PROK'         'UMBF'         'CYBR'         'TCPC'         'BFRG'         'SFNC'         'CCLD'    ,
    'IPSC'         'LAMR'         'FOXF'         'ROP'          'STEP'         'IUSG'         'RYTM'         'MKTX'         'VNDA'         'HUBG'    ,
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
