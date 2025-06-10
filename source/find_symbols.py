import yfinance as yf
import pandas as pd
import urllib.request
import ta

# ----------------------------------------
# 설정
column_width = 8
items_per_line = 10

# ----------------------------------------
# 1. 나스닥 전체 심볼 가져오기
nasdaq_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
urllib.request.urlretrieve(nasdaq_url, "nasdaqlisted.txt")

nasdaq_df = pd.read_csv("nasdaqlisted.txt", sep="|")[:-1]  # 마지막 합계 행 제거
symbols = [str(sym).strip() for sym in nasdaq_df['Symbol'] if pd.notna(sym)]

print(f"총 나스닥 심볼 수: {len(symbols)}")

# ----------------------------------------
# 2. yfinance로 6개월 데이터 다운로드
print("데이터 다운로드 중... (시간 소요)")
data = yf.download(symbols, period='6mo', interval='1d', group_by='ticker', threads=True, auto_adjust=True)

# ----------------------------------------
# 3. 기술적 조건 필터링 (신뢰도 기준)
filtered_symbols = []

print("기술 지표 필터링 중...")
for sym in symbols:
    try:
        df = data[sym].dropna()
        if df.shape[0] < 50:
            continue

        # 50일 이동평균 및 RSI 계산
        df['ma50'] = df['Close'].rolling(window=50).mean()
        df['rsi'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()

        last_close = df['Close'].iloc[-1]
        last_ma50 = df['ma50'].iloc[-1]
        last_rsi = df['rsi'].iloc[-1]

        # 조건: 50일선 위 + RSI 30~70
        if last_close > last_ma50 and 30 < last_rsi < 70:
            avg_volume = df['Volume'].mean()
            filtered_symbols.append((sym, avg_volume))

    except Exception as e:
        continue  # 오류 발생 시 무시

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
