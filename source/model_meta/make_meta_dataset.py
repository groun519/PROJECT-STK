# Source/meta/make_meta_dataset.py
import os, torch, numpy as np
from tqdm import tqdm

from data._data_config import (INTERVALS,
                               START_DATE, END_DATE, LABEL_THRESHOLDS)
from data.data_fetcher import load_multitimeframe_data
from data.labeling_utils import get_all_labels
from model_transformer import MultiHeadTransformer
from _model_config import DEVICE, MODEL_SAVE_PATH
from model_meta._meta_model_config import SYMBOL, TARGET_INTERVAL, META_DIR

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

# ---- 멀티프레임 데이터 메모리 로딩 -------------------------------
mtf = load_multitimeframe_data(SYMBOL, start=START_DATE, end=END_DATE)
target_df = mtf["stock"][TARGET_INTERVAL]

X_meta, y_meta = [], []
thr = LABEL_THRESHOLDS[TARGET_INTERVAL]

for ts in tqdm(target_df.index[:-1], desc="build meta"):
    vec = []
    skip = False
    for tf in INTERVALS:
        df = mtf["stock"][tf]
        if ts not in df.index: skip=True; break
        x  = df.loc[[ts]].values.astype(np.float32)                     # (1,1,feat)
        with torch.no_grad():
            p = models[tf](torch.tensor(x).to(DEVICE))["direction"]     # logits
            vec.extend(torch.softmax(p, -1).cpu().numpy().ravel())      # [↑ ↔ ↓]
    if skip: continue

    # ---- 30m 라벨 5종 ------------------------------------------------
    pos = target_df.index.get_loc(ts)
    if pos+1 >= len(target_df): break
    lab = get_all_labels(target_df.iloc[pos:pos+2], thr)
    label_vec = np.hstack([
        lab["trend"],          # 1
        lab["position"],       # 1
        lab["highest"],        # 1
        lab["regression"],     # 1 (Δ가격 등)
        lab["candle"]          # 4 (O,H,L,C)
    ])
    X_meta.append(vec); y_meta.append(label_vec)

np.savez(f"{META_DIR}/meta_dataset_{SYMBOL}_{TARGET_INTERVAL}.npz",
         X=np.array(X_meta, np.float32),
         y=np.array(y_meta,  np.float32))
print("✔ meta_dataset saved:", len(y_meta), "rows")
