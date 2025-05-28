# data_fetcher.py

import os
import pandas as pd
import yfinance as yf
from config import INDEX_SYMBOL, START_DATE, END_DATE

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_cached_data(symbol, interval):
    fname = f"{symbol.replace('.', '-')}_{interval}.csv"
    path = os.path.join(CACHE_DIR, fname)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0)
            df.index = pd.to_datetime(df.index, errors="coerce")
            if not df.index.tz:
                df.index = df.index.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
            return df
        except Exception as e:
            print(f"[캐시 로딩 실패] {symbol} {interval}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_cached_data(symbol, interval, df):
    fname = f"{symbol.replace('.', '-')}_{interval}.csv"
    path = os.path.join(CACHE_DIR, fname)
    df.to_csv(path)

def update_cache(symbol, interval, start, end):
    symbol = symbol.replace(".", "-")
    cached_df = load_cached_data(symbol, interval)
    latest_date = cached_df.index[-1] if not cached_df.empty else pd.to_datetime(start)

    try:
        print(f"[다운로드 요청] {symbol} ({interval}) from {latest_date} to {end}")
        ticker = yf.Ticker(symbol)
        df_new = ticker.history(interval=interval, start=latest_date, end=end)

        if df_new.empty:
            print(f"[경고] {symbol} {interval}: 다운로드된 데이터가 비어 있음")
            return cached_df

        df_new.index = pd.to_datetime(df_new.index, errors="coerce")
        if not df_new.index.tz:
            df_new.index = df_new.index.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")

        df_combined = pd.concat([cached_df, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
        save_cached_data(symbol, interval, df_combined)
        return df_combined

    except Exception as e:
        print(f"❗ 다운로드 실패 ({symbol}, {interval}): {e}")
        return cached_df

def load_multitimeframe_data(symbol, index_symbol=INDEX_SYMBOL, start=START_DATE, end=END_DATE):
    intervals = ["2m", "5m", "15m", "30m", "60m", "1d"]
    data = {"stock": {}, "index": {}}
    for interval in intervals:
        stock_df = update_cache(symbol, interval, start, end)
        index_df = update_cache(index_symbol, interval, start, end)

        if not stock_df.empty:
            data["stock"][interval] = stock_df
        if not index_df.empty:
            data["index"][interval] = index_df
    return data
