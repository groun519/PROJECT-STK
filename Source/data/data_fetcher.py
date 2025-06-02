import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from _data_config import (
    DATA_PATH, INTERVAL_MINUTES, START_DATE, END_DATE, INDEX_SYMBOL
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

def clean_df_index(df):
    """
    1. 인덱스에서 날짜형으로 파싱 불가한 row는 전부 삭제
    2. DatetimeIndex로 변환 및 tz-localize(UTC)
    3. index 기준으로 정렬
    """
    # (1) 인덱스를 날짜형으로 강제 변환 (파싱 불가한 값은 NaT가 됨)
    try:
        df.index = pd.to_datetime(df.index, errors='coerce')
    except Exception as e:
        print(f"⚠️ clean_df_index: 날짜 변환 실패: {e}")

    # (2) 날짜로 변환 안된 (NaT) row 전부 제거
    df = df[~df.index.isna()]

    # (3) tz 정보 없으면 UTC로 localize, 있으면 UTC로 통일
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

    # (4) 인덱스 기준으로 정렬
    df = df.sort_index()
    return df

def load_multitimeframe_data(symbol, index_symbol=INDEX_SYMBOL, start=START_DATE, end=END_DATE):
    stock_data = {}
    index_data = {}

    for interval in INTERVAL_MINUTES:
        df_symbol = download_data_with_cache(symbol, interval, start, end)
        df_index = download_data_with_cache(index_symbol, interval, start, end)

        if df_symbol is not None and not df_symbol.empty:
            df_symbol = clean_df_index(df_symbol)
            stock_data[interval] = df_symbol
        if df_index is not None and not df_index.empty:
            df_symbol = clean_df_index(df_index)
            index_data[interval] = df_index

    return {"stock": stock_data, "index": index_data}
