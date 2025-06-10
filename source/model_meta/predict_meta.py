# ───────── import packages ──────────
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch, numpy as np, pandas as pd, sys
from datetime import datetime
# ───────── data ──────────
from data._data_config import INTERVALS
from data.data_fetcher import load_multitimeframe_data
# ───────── base ──────────
from model_base.model_transformer import MultiHeadTransformer
# ───────── meta ──────────
from model_meta.train_meta_model import MetaNet          # 같은 클래스 재사용
from model_meta._meta_model_config import SYMBOL, TARGET_INTERVAL, DEVICE
# ─────────

# ---- 베이스 모델 로드 ----
from model_base._model_config import MODEL_SAVE_PATH

models={}
for tf in INTERVALS:
    sd=torch.load(f"{MODEL_SAVE_PATH}/{tf}_model.pt",map_location=DEVICE)
    net=MultiHeadTransformer(input_dim=sd["input_linear.weight"].shape[1],
                             seq_len=sd["pos_encoder"].shape[1]).to(DEVICE)
    net.load_state_dict(sd); net.eval(); models[tf]=net

# ---- 메타모델 로드 ----
in_dim = len(INTERVALS)*3
meta = MetaNet(in_dim).to(DEVICE)
meta.load_state_dict(torch.load(f"meta/meta_mlp_{SYMBOL}_{TARGET_INTERVAL}.pt",
                                map_location=DEVICE))
meta.eval()

def predict(ts: pd.Timestamp):
    mtf = load_multitimeframe_data(SYMBOL,
           start=ts-pd.Timedelta("3d"), end=ts+pd.Timedelta("1m"))
    fv=[]
    for tf in INTERVALS:
        df = mtf["stock"][tf];              # DataFrame
        if ts not in df.index: return None
        x  = torch.tensor(df.loc[[ts]].values.astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            p  = models[tf](x)["direction"].softmax(-1).cpu().numpy().ravel()
        fv.extend(p)
    fv = torch.tensor(fv, dtype=torch.float32).to(DEVICE)
    out = meta(fv.unsqueeze(0))
    return {
        "trend":    int(out["trend"].argmax(1).item()),
        "position": int(out["position"].argmax(1).item()),
        "highest":  int(out["highest"].argmax(1).item()),
        "reg":      float(out["reg"].item()),
        "candle":   out["candle"].cpu().numpy().ravel().tolist()
    }

if __name__ == "__main__":
    ts = pd.Timestamp(datetime.utcnow()).floor("30min").tz_localize("UTC")
    print("Meta prediction @", ts, "→", predict(ts))
