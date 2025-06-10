import yfinance as yf
import pandas as pd
import urllib.request
import ta
import time

# ----------------------------------------
# 설정
column_width = 5
items_per_line = 10

# ----------------------------------------
# 1. 나스닥 전체 심볼 가져오기
nasdaq_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
urllib.request.urlretrieve(nasdaq_url, "nasdaqlisted.txt")

nasdaq_df = pd.read_csv("nasdaqlisted.txt", sep="|")[:-1]  # 마지막 합계 행 제거
symbols = [str(sym).strip() for sym in nasdaq_df['Symbol'] if pd.notna(sym)]

print(f"총 나스닥 심볼 수: {len(symbols)}")

# ----------------------------------------
# 2. yfinance로 6개월 데이터 다운로드 (100개씩 나누기)
print("데이터 다운로드 중... (시간 소요)")
def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

all_data = {}
for group in chunks(symbols, 100):
    try:
        data = yf.download(
            tickers=group,
            period='6mo',
            interval='1d',
            group_by='ticker',
            threads=True,
            auto_adjust=True,
            progress=False
        )
        if isinstance(data.columns, pd.MultiIndex):
            for sym in group:
                if sym in data.columns.levels[0]:
                    all_data[sym] = data[sym].dropna()
        else:
            sym = group[0]
            all_data[sym] = data.dropna()
    except Exception as e:
        print(f"오류 발생 (건너뜀): {e}")
    time.sleep(1)  # rate limit 방지용 대기

print(f"수집된 종목 수: {len(all_data)}")

# ----------------------------------------
# 3. 기술적 조건 필터링 (신뢰도 기준)
filtered_symbols = []
print("기술 지표 필터링 중...")
for sym, df in all_data.items():
    try:
        if df.shape[0] < 50:
            continue

        df['ma50'] = df['Close'].rolling(window=50).mean()
        df['rsi'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()

        last_close = df['Close'].iloc[-1]
        last_ma50 = df['ma50'].iloc[-1]
        last_rsi = df['rsi'].iloc[-1]

        if last_close > last_ma50 and 30 < last_rsi < 70:
            avg_volume = df['Volume'].mean()
            filtered_symbols.append((sym, avg_volume))
    except Exception:
        continue

# ----------------------------------------
# 4. 평균 거래량 기준 상위 300개 선택
top300 = sorted(filtered_symbols, key=lambda x: x[1], reverse=True)[:300]
SYMBOL_LIST = [sym for sym, _ in top300]

# ----------------------------------------
# 5. 보기 좋게 출력
print("\nSYMBOL_LIST = [")
for i, sym in enumerate(SYMBOL_LIST):
    padded = f"'{sym}'".ljust(column_width + 2)
    end = ",\n" if (i + 1) % items_per_line == 0 else " "
    print(f"    {padded}", end=end)
print("]")
