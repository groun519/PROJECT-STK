import os
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
import ta

from data._data_config import (
    DATA_PATH, INTERVAL_MINUTES, START_DATE, END_DATE, INDEX_SYMBOL, TECHNICAL_INDICATORS, TECHNICAL_PARAMS
)

def download_data_with_cache(symbol, interval, start, end):
    
    os.makedirs(DATA_PATH, exist_ok=True)
    
    cache_path = f"{DATA_PATH}/{symbol}_{interval}.csv"
    if os.path.exists(cache_path):
        print(f"[캐시 적중] {symbol} ({interval})")
        return pd.read_csv(cache_path, index_col=0, header=0, parse_dates=True, date_format="%Y-%m-%d %H:%M:%S")

    print(f"[다운로드 요청] {symbol} ({interval}) from {start} to {end}")
    try:
        df = yf.download(symbol, interval=interval, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 10:
            print(f"[데이터 부족] {symbol} ({interval}) → 스킵")
            return None
        df.to_csv(cache_path, date_format="%Y-%m-%d %H:%M:%S")
        return df
    
    except Exception as e:
        print(f"[다운로드 오류] {symbol} ({interval}) → {e}")
        return None

def align_to_nasdaq_trading_days(df, start_date, end_date):
    """
    NASDAQ 영업일 기준으로 DataFrame 인덱스를 정렬하고 결측치는 ffill로 보정.
    - start_date, end_date: 반드시 문자열('YYYY-MM-DD') 형태로 전달 (config 등에서 받아서 사용)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # NASDAQ 캘린더에서 해당 기간 영업일 추출
    cal = mcal.get_calendar('NASDAQ')
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index

    # 영업일 기준으로 리인덱싱 및 결측 ffill
    df = df.reindex(trading_days).ffill()

    return df

# def clean_df_index(df):
#     """
#     1. 인덱스에서 날짜형으로 파싱 불가한 row는 전부 삭제
#     2. DatetimeIndex로 변환 및 tz-localize(UTC)
#     3. index 기준으로 정렬
#     """
#     # (1) 인덱스를 날짜형으로 강제 변환 (파싱 불가한 값은 NaT가 됨)
#     try:
#         df.index = pd.to_datetime(df.index, errors='coerce')
#     except Exception as e:
#         print(f"⚠️ clean_df_index: 날짜 변환 실패: {e}")

#     # (2) 날짜로 변환 안된 (NaT) row 전부 제거
#     df = df[~df.index.isna()]

#     # (3) tz 정보 없으면 UTC로 localize, 있으면 UTC로 통일
#     if isinstance(df.index, pd.DatetimeIndex):
#         if df.index.tz is None:
#             df.index = df.index.tz_localize('UTC')
#         else:
#             df.index = df.index.tz_convert('UTC')

#     # (4) 인덱스 기준으로 정렬
#     df = df.sort_index()
#     return df

def load_multitimeframe_data(symbol, index_symbol=INDEX_SYMBOL, start=START_DATE, end=END_DATE):
    stock_data = {}
    index_data = {}

    for interval in INTERVAL_MINUTES:
        df_symbol = download_data_with_cache(symbol, interval, start, end)
        df_index = download_data_with_cache(index_symbol, interval, start, end)

        if df_symbol is not None and not df_symbol.empty:
            df_symbol = compute_indicators(df_symbol)
            if df_symbol is None or df_symbol.empty:
                print(f"[{symbol}][{interval}] 기술지표 계산 실패/결측치로 데이터 제외됨")
            else:
                stock_data[interval] = df_symbol
        if df_index is not None and not df_index.empty:
            df_index = compute_indicators(df_index)
            if df_index is None or df_index.empty:
                print(f"[{index_symbol}][{interval}] 기술지표 계산 실패/결측치로 데이터 제외됨")
            else:
                index_data[interval] = df_index

    return {"stock": stock_data, "index": index_data}

def compute_indicators(df):
    # 컬럼명 소문자화
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]
    
    # 실수형으로 강제 변환
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # 필수 칼럼 확인
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            print(f"[지표 계산 예외] 필수 컬럼 누락: {col}")
            return None

    try:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # # RSI
        # if "rsi" in TECHNICAL_INDICATORS:
        #     window = TECHNICAL_PARAMS["rsi"].get("window", 14)
        #     df["rsi"] = ta.momentum.RSIIndicator(close, window=window).rsi()
        # # MACD
        # if "macd" in TECHNICAL_INDICATORS:
        #     p = TECHNICAL_PARAMS["macd"]
        #     macd = ta.trend.MACD(
        #         close, 
        #         window_slow=p.get("slow", 26), 
        #         window_fast=p.get("fast", 12), 
        #         window_sign=p.get("signal", 9)
        #     )
        #     df["macd"] = macd.macd_diff()
        # # 볼린저밴드
        # if "boll_upper" in TECHNICAL_INDICATORS or "boll_lower" in TECHNICAL_INDICATORS:
        #     window = TECHNICAL_PARAMS["boll"].get("window", 20)
        #     std = TECHNICAL_PARAMS["boll"].get("std", 2)
        #     boll = ta.volatility.BollingerBands(close, window=window, window_dev=std)
        #     if "boll_upper" in TECHNICAL_INDICATORS:
        #         df["boll_upper"] = boll.bollinger_hband()
        #     if "boll_lower" in TECHNICAL_INDICATORS:
        #         df["boll_lower"] = boll.bollinger_lband()
        # # 거래량 변화율
        # if "volume_change" in TECHNICAL_INDICATORS:
        #     df["volume_change"] = volume.pct_change()

        # NaN/inf 처리 및 칼럼별 예외 로깅
        before_shape = df.shape[0]
        df = df.replace([np.inf, -np.inf], np.nan)
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            print(f"[지표 계산 NaN/inf] 결측치 발생 칼럼: {nan_cols}")
            
        df = df.dropna()
        after_shape = df.shape[0]
        if before_shape != after_shape:
            print(f"[지표 계산 dropna] 결측치로 {before_shape - after_shape}행 삭제")
        return df

    except Exception as e:
        print(f"[지표 계산 실패] {e}")
        return None