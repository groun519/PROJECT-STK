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
from model_meta._meta_model_config import SYMBOL, TARGET_INTERVAL, META_DIR
# ─────────

os.makedirs(META_DIR, exist_ok=True)

# def build_sample(mtf, interval, ts):
#     df_s = mtf["stock"][interval]; df_i = mtf["index"][interval]
#     win  = REQUIRED_LENGTH[interval] 
#     pos  = df_s.index.get_indexer([ts], method="nearest")[0]
#     if pos < win-1:  return None, "win"
#     rows = np.concatenate(
#         [df_s.iloc[pos-win+1:pos+1].values,
#          df_i.iloc[pos-win+1:pos+1].values], axis=1)
#     return rows.astype(np.float32), None

def build_sample(mtf, interval, ts):
    df_s = mtf["stock"][interval]; df_i = mtf["index"][interval]

    # ── ① 두 DF 인덱스를 UTC 로 localize/convert ───────────────────
    if df_s.index.tz is None: df_s.index = df_s.index.tz_localize("UTC")
    else:                     df_s.index = df_s.index.tz_convert("UTC")
    if df_i.index.tz is None: df_i.index = df_i.index.tz_localize("UTC")
    else:                     df_i.index = df_i.index.tz_convert("UTC")

    # ── ② ts 도 UTC 로 보정 (naive → localize, ET 등 → convert) ───
    if ts.tz is None: ts = ts.tz_localize("UTC")
    else:             ts = ts.tz_convert("UTC")

    # --- 위치 찾기 ---------------------------------------------------
    pos = df_s.index.get_indexer([ts], method="nearest")[0]
    if pos == -1:  return None, f"{interval}:pos"
    win = REQUIRED_LENGTH[interval]
    if pos < win-1: return None, f"{interval}:win"

    rows = np.concatenate(
        [df_s.iloc[pos-win+1:pos+1].values,
         df_i.iloc[pos-win+1:pos+1].values], axis=1)
    return rows.astype(np.float32), None



# 1. 베이스 모델 로드 ----------------------------------------------------------
models = {}
for tf in INTERVALS:
    ckpt = torch.load(f"{MODEL_SAVE_PATH}/{tf}_model.pt", map_location=DEVICE)
    seq  = ckpt["pos_encoder"].shape[1]
    dim  = ckpt["input_linear.weight"].shape[1]
    net  = MultiHeadTransformer(input_dim=dim, seq_len=seq).to(DEVICE)
    net.load_state_dict(ckpt); net.eval()
    models[tf] = net

# ---- 멀티프레임 데이터 메모리 로딩 -------------------------------
mtf = load_multitimeframe_data(SYMBOL, start=START_DATE, end=END_DATE)
target_df = mtf["stock"][TARGET_INTERVAL]

ts_range = target_df.index

X_meta, y_meta = [], []
thr = LABEL_THRESHOLDS[TARGET_INTERVAL]


# for ts in tqdm(target_df.index[:-1], desc="build meta"):
#     vec = []
#     skip = False
#     for tf in INTERVALS:
#         df = mtf["stock"][tf]
#         if ts not in df.index: skip=True; break
#         x  = torch.tensor(df.loc[[ts]].values.astype(np.float32)).to(DEVICE)
#         x = build_sample(mtf, tf, ts)
#         if x is None: skip = True; break
#         with torch.no_grad():
#             p = models[tf](torch.tensor(x).unsqueeze(0).to(DEVICE))["direction"]
#             vec.extend(torch.softmax(p, -1).cpu().numpy().ravel())      # [↑ ↔ ↓]
#     if skip: continue

#     # ---- 30m 라벨 5종 ------------------------------------------------
#     pos = target_df.index.get_loc(ts)
#     if pos+1 >= len(target_df): break
#     lab = get_all_labels(target_df.iloc[pos:pos+2], thr)
#     label_vec = np.hstack([
#         lab["trend"], lab["position"], lab["highest"],
#         lab["regression"],
#         lab["candle"]          # 4-length
#     ])
#     X_meta.append(vec); y_meta.append(label_vec)
    
# ----------------- 수정된 루프 전체 -----------------
skip_cnt = {}                                          # 이유별 건수 집계

for ts in tqdm(target_df.index[:-1], desc="build meta"):
    vec, skip = [], False

    for tf in INTERVALS:
        # ① stock/index 윈도우 → build_sample 로 한방에
        x, reason = build_sample(mtf, tf, ts)          # (win, feat) or None
        if x is None:
            skip_cnt[reason] = skip_cnt.get(reason,0)+1
            skip = True
            break

        # ② (batch=1, seq_len, feat) 로 베이스 모델 추론
        with torch.no_grad():
            logits = models[tf]( torch.tensor(x)
                                 .unsqueeze(0)         # (1,win,feat)
                                 .to(DEVICE) )["direction"]
            vec.extend( torch.softmax(logits,-1)
                              .cpu().numpy().ravel() )  # [↑ ↔ ↓] 3-개

    if skip:
        continue

    # ---- ③ 30 분봉 라벨 5종 ----------------------------------------
    pos = target_df.index.get_loc(ts)
    lab = get_all_labels(target_df.iloc[pos:pos+2], thr)
    print(lab)
    
    label_vec = np.hstack([
        lab["trend"],                   # 0,1,2  (int)
        lab["position"],                # 0~N     (int)
        lab["highest"],                 # 0~N     (int)
        lab["regression"],              # float   ← 스칼라
        lab["candle"]                   # 4-float OHLC
    ]).astype(np.float32)

    X_meta.append(vec)
    y_meta.append(label_vec)

print("\n▶ skip summary:", skip_cnt)
print("▶ collected samples:", len(y_meta))



np.savez(f"{META_DIR}/meta_dataset_{SYMBOL}_{TARGET_INTERVAL}.npz",
         X=np.array(X_meta, np.float32),
         y=np.array(y_meta,  np.float32))
print("✔ meta_dataset saved:", len(y_meta), "rows")
