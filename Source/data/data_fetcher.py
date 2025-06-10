import os
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal

from data._data_config import (
    DATA_PATH, INTERVALS, START_DATE, END_DATE, INDEX_SYMBOL
)

def download_data_with_cache(symbol, interval, start, end, disable_log = False):
    os.makedirs(DATA_PATH, exist_ok=True)
    cache_path = f"{DATA_PATH}/{symbol}_{interval}.csv"

    if os.path.exists(cache_path):
        if disable_log is False : print(f"📂 [저장된 데이터 사용] {symbol} | {interval} | {start} ~ {end}")
        df = pd.read_csv(
            cache_path,
            header=0,       # 첫 번째 줄을 컬럼명으로
            skiprows=[1],   # 두 번째 줄("Ticker,QQQ,...")을 건너뜀
            index_col=0,    # 첫 컬럼(날짜/Price)을 인덱스로
            parse_dates=True
        )
        return df

    print(f"⬇️ [데이터 다운로드] {symbol} | {interval} | {start} ~ {end}")
    try:
        df = yf.download(symbol, interval=interval, start=start, end=end, progress=False, auto_adjust=True)
        df = df.reset_index()
        # 멀티인덱스 대응: tuple → str
        datetime_col = None
        for col in df.columns:
            col_name = col[0] if isinstance(col, tuple) else col
            if "date" in str(col_name).lower():
                datetime_col = col
                break
        if datetime_col is not None:
            df = df.set_index(datetime_col)
        else:
            raise ValueError(f"{symbol} 다운로드: 날짜 컬럼을 찾을 수 없음")

        df.index.name = None
        df.columns.name = None

        df.to_csv(cache_path, date_format="%Y-%m-%d %H:%M:%S")

        if df.empty or len(df) < 1:
            print(f"⚠️ [데이터 부족] {symbol} | {interval} | 다운로드된 행 개수: {len(df)}")
            return None

        df = align_to_nasdaq_trading_days(df, start, end, interval)
        df.to_csv(cache_path, date_format="%Y-%m-%d %H:%M:%S")
        print(f"💾 [데이터 저장 완료] {symbol} | {interval} | 파일: {cache_path}")
        return df

    except Exception as e:
        print(f"❌ [다운로드 오류] {symbol} | {interval} | {e}")
        return None

def align_to_nasdaq_trading_days(df, start_date, end_date, freq="1d"):
    """
    freq 예시:
        - "1d"      → 일봉
        - "5min"    → 5분봉
        - "15min"   → 15분봉
        - "1h"      → 1시간봉
    """
    if freq == "5m" : freq = "5min"
    elif freq == "15m" : freq = "15min"
    elif freq == "30m" : freq = "30min"
    elif freq == "60m" : freq = "1h"
    
    import pandas_market_calendars as mcal
    import pandas as pd

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    cal = mcal.get_calendar('NASDAQ')
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    market_opens = schedule['market_open']
    market_closes = schedule['market_close']

    # 일봉일 경우
    if freq in ["1d", "D"]:
        trading_index = schedule.index
    else:
        # 분봉: 모든 거래일에 대해 실제 거래 시간 내 인덱스 생성
        trading_index = []
        for open_time, close_time in zip(market_opens, market_closes):
            # 예: 5분봉이라면 freq="5min"
            rng = pd.date_range(start=open_time, end=close_time, freq=freq)
            trading_index.extend(rng)
        trading_index = pd.DatetimeIndex(trading_index)

    # 기준 인덱스로 정렬, 결측은 ffill
    df = df.reindex(trading_index).ffill()
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

def load_multitimeframe_data(symbol, index_symbol=INDEX_SYMBOL, start=START_DATE, end=END_DATE, disable_log=False):
    stock_data = {}
    index_data = {}

    for interval in INTERVALS:
        df_symbol = download_data_with_cache(symbol, interval, start, end, disable_log=disable_log)
        df_index = download_data_with_cache(index_symbol, interval, start, end, disable_log=disable_log)

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
    # ---- 컬럼명 평탄화 ----
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() if isinstance(c, tuple) else str(c).lower()
                      for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    # ---- 필수 OHLCV 남기고 나머지 무시 ----
    essential = ["open", "high", "low", "close", "volume"]
    df = df[essential]

    # ---- numeric & dropna  ----
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    if df.empty:
        return None
    return df



# def compute_indicators(df):
#     # 컬럼명 소문자화
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = [ str(col[0]).lower() if isinstance(col, tuple) else str(col).lower()
#                        for col in df.columns ]
#     else:
#         df.columns = [col.lower() for col in df.columns]
    
#     # 실수형으로 강제 변환
#     df = df.apply(pd.to_numeric, errors="coerce")
    
#     # 필수 칼럼 확인
#     required = {"open","high","low","close","volume"}
#     if not required.issubset(df.columns):
#         print(f"[지표 계산 예외] 필수 컬럼 누락: {required - set(df.columns)}")
#         return None

#     df = df.apply(pd.to_numeric, errors="coerce")
#     df = df.replace([np.inf, -np.inf], np.nan).dropna()

#     try:
#         close = df["close"]
#         high = df["high"]
#         low = df["low"]
#         volume = df["volume"]

#         # # RSI
#         # if "rsi" in TECHNICAL_INDICATORS:
#         #     window = TECHNICAL_PARAMS["rsi"].get("window", 14)
#         #     df["rsi"] = ta.momentum.RSIIndicator(close, window=window).rsi()
#         # # MACD
#         # if "macd" in TECHNICAL_INDICATORS:
#         #     p = TECHNICAL_PARAMS["macd"]
#         #     macd = ta.trend.MACD(
#         #         close, 
#         #         window_slow=p.get("slow", 26), 
#         #         window_fast=p.get("fast", 12), 
#         #         window_sign=p.get("signal", 9)
#         #     )
#         #     df["macd"] = macd.macd_diff()
#         # # 볼린저밴드
#         # if "boll_upper" in TECHNICAL_INDICATORS or "boll_lower" in TECHNICAL_INDICATORS:
#         #     window = TECHNICAL_PARAMS["boll"].get("window", 20)
#         #     std = TECHNICAL_PARAMS["boll"].get("std", 2)
#         #     boll = ta.volatility.BollingerBands(close, window=window, window_dev=std)
#         #     if "boll_upper" in TECHNICAL_INDICATORS:
#         #         df["boll_upper"] = boll.bollinger_hband()
#         #     if "boll_lower" in TECHNICAL_INDICATORS:
#         #         df["boll_lower"] = boll.bollinger_lband()
#         # # 거래량 변화율
#         # if "volume_change" in TECHNICAL_INDICATORS:
#         #     df["volume_change"] = volume.pct_change()

#         # NaN/inf 처리 및 칼럼별 예외 로깅
#         before_shape = df.shape[0]
#         df = df.replace([np.inf, -np.inf], np.nan)
#         nan_cols = df.columns[df.isna().any()].tolist()
#         if nan_cols:
#             print(f"[지표 계산 NaN/inf] 결측치 발생 칼럼: {nan_cols}")
            
#         df = df.dropna()
#         after_shape = df.shape[0]
#         if before_shape != after_shape:
#             print(f"[지표 계산 dropna] 결측치로 {before_shape - after_shape}행 삭제")
#         return df

#     except Exception as e:
#         print(f"[지표 계산 실패] {e}")
#         return None