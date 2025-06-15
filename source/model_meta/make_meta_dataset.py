# ───────── import packages ──────────
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch, numpy as np, pandas as pd
from tqdm import tqdm
# ───────── data ──────────
from data._data_config import (INTERVALS, REQUIRED_LENGTH,
                               START_DATE, END_DATE, LABEL_THRESHOLDS)
from data.data_fetcher import load_multitimeframe_data
from data.labeling_utils import get_all_labels
# ───────── base ──────────
from model_base.model_transformer import MultiHeadTransformer
from model_base._model_config import DEVICE, MODEL_SAVE_PATH
# ───────── meta ──────────
from model_meta._meta_model_config import SYMBOL, TARGET_INTERVAL, META_DIR, BINS
# ─────────

os.makedirs(META_DIR, exist_ok=True)

# 1. 베이스 모델 로드 ----------------------------------------------------------
models = {}
for tf in INTERVALS:
    ckpt = torch.load(f"{MODEL_SAVE_PATH}/{tf}_model.pt", map_location=DEVICE)
    seq  = ckpt["pos_encoder"].shape[1]
    dim  = ckpt["input_linear.weight"].shape[1]
    net  = MultiHeadTransformer(input_dim=dim, seq_len=seq).to(DEVICE)
    net.load_state_dict(ckpt); net.eval()
    models[tf] = net

# ── helper: 윈도우 슬라이스 (stock+index) ─────────────────────────
def build_sample(mtf, interval, ts):
    df_s = mtf["stock"][interval]; df_i = mtf["index"][interval]
    for df in (df_s, df_i):
        if df.index.tz is None: df.index = df.index.tz_localize("UTC")
        else:                   df.index = df.index.tz_convert("UTC")
    if ts.tz is None: ts = ts.tz_localize("UTC")
    else:             ts = ts.tz_convert("UTC")

    pos = df_s.index.get_indexer([ts], "nearest")[0]
    if pos == -1:              return None
    win = REQUIRED_LENGTH[interval]
    if pos < win-1:            return None
    
    rows = np.concatenate([
        df_s.iloc[pos-win+1:pos+1].values,
        df_i.iloc[pos-win+1:pos+1].values], axis=1)
    return rows.astype(np.float32)               # (win, feat)


# ── 데이터 수집 루프 ──────────────────────────────────────────────
mtf        = load_multitimeframe_data(SYMBOL, start=START_DATE, end=END_DATE)
target_df  = mtf["stock"][TARGET_INTERVAL]
thr        = LABEL_THRESHOLDS[TARGET_INTERVAL]
ts_range   = target_df.index

X_meta, y_meta = [], []
for ts in tqdm(ts_range, desc="build meta"):
    fv        = []
    
    if ts.tz is None: ts = ts.tz_localize("UTC")
    else:             ts = ts.tz_convert("UTC")
    
    for tf in INTERVALS:                        # 베이스 예측 확률 3×5 =15
        x = build_sample(mtf, tf, ts)
        if x is None: break
        with torch.no_grad():
            prob = models[tf](torch.tensor(x).unsqueeze(0).to(DEVICE))["direction"]
        fv.extend(torch.softmax(prob, -1).cpu().numpy().ravel())
    else:
        pos = target_df.index.get_loc(ts)
        if pos+1 >= len(target_df): break
        lab = get_all_labels(target_df.iloc[pos:pos+2], thr)

        # ---- position 비중(% → [-1,1]) → 11-클래스 + offset ----
        pos_frac = float(lab["position"])               # 연속값
        pos_cls  = np.digitize(pos_frac, BINS) - 1      # 0-10
        pos_off  = pos_frac - BINS[pos_cls]             # [-0.1,0.1]

        label_vec = np.hstack([
            lab["trend"], lab["regression"],
            lab["candle"],                # 4
            lab["highest"],
            pos_cls, pos_off              # 2
        ]).astype(np.float32)             # 총 9
        X_meta.append(fv); y_meta.append(label_vec)


np.savez(f"{META_DIR}/meta_dataset_{SYMBOL}_{TARGET_INTERVAL}.npz",
         X=np.array(X_meta, np.float32),
         y=np.array(y_meta,  np.float32))
print("✔ meta_dataset saved:", len(y_meta), "rows")
