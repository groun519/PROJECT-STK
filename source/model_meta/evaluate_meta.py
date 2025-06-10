# Source/model_meta/evaluate_meta.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# source/model_meta/evaluate_meta_seq.py
import numpy as np, pandas as pd, tqdm, argparse
from datetime import timedelta
from data._data_config import START_DATE, END_DATE
from data.data_fetcher   import load_multitimeframe_data
from model_meta.predict_meta_seq import forecast_100
from model_meta.predict_meta     import is_market_time      # 장중 필터

def evaluate(symbol: str):
    rng = pd.date_range(START_DATE, END_DATE, freq="30min", tz="UTC")
    dir_acc, delta_err, samples = 0, 0, 0

    # 실측 30m DF 한 번만 로드
    df_real = load_multitimeframe_data(
                symbol, start=START_DATE, end=END_DATE)["stock"]["30m"]

    for ts in tqdm.tqdm(rng, desc=symbol):
        if not is_market_time(ts):           # 장중 캔들만
            continue
        pred = forecast_100(ts, history_start=START_DATE, symbol=symbol)
        if "error" in pred:                  # 데이터 부족·장외 skip
            continue
        true_slice = df_real.loc[ts+timedelta(minutes=30):
                                 ts+timedelta(hours=50)]     # 100캔들

        if len(true_slice) < 100:
            continue

        # ① 방향 정확도 (첫 100개 비교)
        pct = (true_slice["close"].values -
               true_slice["close"].iloc[0]) / true_slice["close"].iloc[0]
        true_dir = np.where(pct > 0.002, 0,
                    np.where(pct < -0.002, 2, 1))           # 0↑1→2↓

        dir_acc  += (np.array(pred["direction_seq"][:100]) == true_dir).mean()
        # ② Δ RMSE
        delta_err += np.sqrt(((pred["delta_seq"][:100] - pct)**2).mean())
        samples += 1

    print(f"\n▶ {symbol}  samples={samples}")
    if samples:
        print(" direction acc : {:.3f}".format(dir_acc / samples))
        print(" delta RMSE : {:.4f}".format(delta_err / samples))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL", help="평가용 테스트 심볼")
    args = ap.parse_args()
    evaluate(args.symbol.upper())
