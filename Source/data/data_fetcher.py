import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from _data_config import DATA_DIR, MULTI_TIMEFRAMES, START_DATE, END_DATE, INDEX_SYMBOL

def download_data_with_cache(symbol, interval, start, end):
    cache_path = f"{DATA_DIR}/{symbol}_{interval}.csv"
    if os.path.exists(cache_path):
        print(f"[캐시 적중] {symbol} ({interval})")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"[다운로드 요청] {symbol} ({interval}) from {start} to {end}")
    try:
        df = yf.download(symbol, interval=interval, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            print(f"[데이터 부족] {symbol} ({interval}) → 스킵")
            return None
        df.to_csv(cache_path)
        return df
    except Exception as e:
        print(f"[다운로드 오류] {symbol} ({interval}) → {e}")
        return None

def merge_multi_timeframe_data(symbol, index_symbol=INDEX_SYMBOL, start=START_DATE, end=END_DATE):
    result = {}

    for interval in MULTI_TIMEFRAMES:
        df_symbol = download_data_with_cache(symbol, interval, start, end)
        df_index = download_data_with_cache(index_symbol, interval, start, end)

        if df_symbol is None or df_index is None:
            result[interval] = None
            continue

        # 공통 인덱스로 병합 (좌우 suffix)
        df_merged = df_symbol.join(df_index, rsuffix='_index', how='inner')
        result[interval] = df_merged

    return result
