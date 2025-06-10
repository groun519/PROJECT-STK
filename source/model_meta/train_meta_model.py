# ───────── import packages ──────────
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import joblib, numpy as np, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
# ───────── meta ──────────
from model_meta._meta_model_config import SYMBOL, BATCH, EPOCHS, LR, DEVICE, TARGET_INTERVAL, BINS
# ─────────

data   = np.load(f"meta/meta_dataset_{SYMBOL}_{TARGET_INTERVAL}.npz")
X, y   = torch.tensor(data["X"]), torch.tensor(data["y"])

# # ── 라벨 배열 확인 ──────────────────────────
# print("unique pos-cls:", np.unique(data["y"][:,7]))
# print("min/max pos-off:", data["y"][:,8].min(), data["y"][:,8].max())
# print("rows:", len(data["y"]))
# exit()          # ← 일단 여기서 종료 후 값만 확인


# ── 라벨 분리 ───────────────────────────────
trend_t   = y[:, 0:1].to(DEVICE)
reg_t     = y[:, 1:2].to(DEVICE)
candle_t  = y[:, 2:6].to(DEVICE)         # 4
highest_t = y[:, 6:7].to(DEVICE)
pos_cls_t = y[:, 7].long().to(DEVICE)    # 0-10
pos_off_t = y[:, 8:9].to(DEVICE)

dataset = TensorDataset(X, trend_t, reg_t, candle_t, highest_t, pos_cls_t, pos_off_t)
train_ds, val_ds = train_test_split(dataset, test_size=0.2, shuffle=False)
train_loader = DataLoader(train_ds, BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   BATCH)

# ── 네트워크 ───────────────────────────────
class MetaNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim,64),nn.ReLU(), nn.Linear(64,32),nn.ReLU())
        self.head_trend   = nn.Linear(32,1)
        self.head_reg     = nn.Linear(32,1)
        self.head_candle  = nn.Linear(32,4)
        self.head_highest = nn.Linear(32,1)
        self.head_pos_cls = nn.Linear(32,len(BINS))
        self.head_pos_off = nn.Linear(32,1)
    def forward(self,x):
        h=self.shared(x)
        return {
            "trend":   self.head_trend(h),
            "reg":     self.head_reg(h),
            "candle":  self.head_candle(h),
            "highest": self.head_highest(h),
            "pos_cls": self.head_pos_cls(h),
            "pos_off": self.head_pos_off(h)
        }

net = MetaNet(X.shape[1]).to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=LR)
ce , mse, sl1 = nn.CrossEntropyLoss(), nn.MSELoss(), nn.SmoothL1Loss()

def epoch(loader, train=True):
    net.train(train)
    tot, acc = 0, 0
    for xb,t,r,c,h,pc,po in loader:
        xb,t,r,c,h,pc,po = [a.to(DEVICE) for a in (xb,t,r,c,h,pc,po)]
        out = net(xb)
        loss = (mse(out["trend"], t) + mse(out["reg"], r) +
                mse(out["candle"], c)*0.5 + mse(out["highest"], h) +
                ce(out["pos_cls"], pc) + sl1(out["pos_off"], po)*0.5)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot += xb.size(0)
        acc += (out["pos_cls"].argmax(1)==pc).sum().item()
    return acc / tot

for ep in range(EPOCHS):
    tr = epoch(train_loader, True)
    vl = epoch(val_loader, False)
    print(f"ep{ep+1:02d} | pos-cls acc  train {tr:.3f}  val {vl:.3f}")

torch.save(net.state_dict(),
           f"meta/meta_mlp_{SYMBOL}_{TARGET_INTERVAL}.pt")
print("✔ meta model saved")