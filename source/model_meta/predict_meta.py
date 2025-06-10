# Source/model_meta/predict_meta.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch, numpy as np, pandas as pd
from pandas import Timestamp
from datetime import datetime, timedelta
from data._data_config import SYMBOL_LIST, INTERVALS, REQUIRED_LENGTH, START_DATE, END_DATE
from data.data_fetcher import load_multitimeframe_data
from model_base._model_config import DEVICE, MODEL_SAVE_PATH
from model_base.model_transformer import MultiHeadTransformer
from model_meta._meta_model_config import BINS, SYMBOL, TARGET_INTERVAL, PREDICT_DATE
from model_meta.train_meta_model import MetaNet   # 메타 모델 클래스

# ── 베이스 모델 로드 한 번만 ----------
models = {}
for ivl in INTERVALS:
    ckpt = torch.load(f"{MODEL_SAVE_PATH}/{ivl}_model.pt",
                      map_location=DEVICE)
    # seq_len 자동 추출
    pe = ckpt["pos_encoder"]                        # 텐서 (L,D) or (D,L)
    seq_len = pe.shape[0] if pe.shape[0] > pe.shape[1] else pe.shape[1]
    in_dim  = ckpt["input_linear.weight"].shape[1]

    net = MultiHeadTransformer(input_dim=in_dim,
                               seq_len=seq_len).to(DEVICE)
    net.load_state_dict(ckpt); net.eval()
    models[ivl] = net

# ── 메타 모델 로드 ---------------------
meta = MetaNet(len(INTERVALS)*3).to(DEVICE)
meta.load_state_dict(torch.load(
        f"meta/meta_mlp_{SYMBOL}_{TARGET_INTERVAL}.pt",
        map_location=DEVICE))
meta.eval()

def is_market_time(ts_utc):
    ts_et = ts_utc.tz_convert("US/Eastern")
    return ts_et.weekday() < 5 and ts_et.time() >= pd.Timestamp("09:30").time() \
                              and ts_et.time() <= pd.Timestamp("16:00").time()

def build_live_sample(mtf, interval, ts):
    df_s = mtf["stock"][interval]
    df_i = mtf["index"][interval]

    # ❶ 두 DataFrame 인덱스를 UTC 로 통일
    if df_s.index.tz is None: df_s.index = df_s.index.tz_localize("UTC")
    else:                     df_s.index = df_s.index.tz_convert("UTC")
    if df_i.index.tz is None: df_i.index = df_i.index.tz_localize("UTC")
    else:                     df_i.index = df_i.index.tz_convert("UTC")

    # ❷ ts 역시 UTC 로 보정
    if ts.tz is None: ts = ts.tz_localize("UTC")
    else:             ts = ts.tz_convert("UTC")

    win  = REQUIRED_LENGTH[interval]
    pos  = df_s.index.get_indexer([ts], method="nearest")[0]
    if pos < win-1: return None

    rows = np.concatenate(
        [df_s.iloc[pos-win+1:pos+1].values,
         df_i.iloc[pos-win+1:pos+1].values], axis=1)
    return torch.tensor(rows, dtype=torch.float32)          # (win, 10)

# ── 메인 추론 함수 ---------------------
def predict(ts: pd.Timestamp):     
    start = Timestamp(START_DATE).tz_localize("UTC").strftime("%Y-%m-%d")
    end   = max(Timestamp(END_DATE).tz_localize("UTC"), ts).strftime("%Y-%m-%d")
    
    mtf   = load_multitimeframe_data(SYMBOL, start=start, end=end, disable_log=True)
    feats = []
    for ivl in INTERVALS:
        x = build_live_sample(mtf, ivl, ts)
        if x is None:
            return None      # 초기 구간 부족 → 예측 skip
        with torch.no_grad():
            logits = models[ivl](x.unsqueeze(0).to(DEVICE))["direction"]
            feats.extend(torch.softmax(logits, -1).cpu().numpy().ravel())
    feats = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = meta(feats)
    # ----- 결과 가공 -----
    trend_val = float(out["trend"].item())
    direction = 0 if trend_val > 0.002 else 2 if trend_val < -0.002 else 1

    pos_cls  = int(out["pos_cls"].argmax(1).item())
    pos_off  = float(out["pos_off"].item())
    position = float(np.clip(BINS[pos_cls] + pos_off, -1, 1))

    return {
        "timestamp":      str(ts),
        "direction_cls":  direction,            # 0↑ 1→ 2↓
        "trend_frac":     trend_val,
        "reg_100":        float(out["reg"].item()),
        "candle_100":     out["candle"].cpu().numpy().ravel().tolist(),
        "highest_100":    float(out["highest"].item()),
        "position_frac":  position              # 최종 비중 [-1,1]
    }

# ── CLI 테스트 -------------------------
if __name__ == "__main__":
    ts = pd.Timestamp(PREDICT_DATE).floor("30min").tz_localize("UTC")
    res = predict(ts)
    if res:
        from pprint import pprint
        pprint(res)
    else:
        print("Not enough data for the requested timestamp.")
