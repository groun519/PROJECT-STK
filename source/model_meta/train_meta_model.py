# ───────── import packages ──────────
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
# ───────── meta ──────────
from model_meta._meta_model_config import SYMBOL, BATCH, EPOCHS, LR, DEVICE, TARGET_INTERVAL
# ─────────

data = np.load(f"meta/meta_dataset_{SYMBOL}_{TARGET_INTERVAL}.npz")
X, y  = torch.tensor(data["X"]), torch.tensor(data["y"])

# y 분리: 분류(3)+분류(3)+분류(3) → trend/position/highest (원-핫), 회귀 1, OHLC-4
trend_t      = y[:,0].long()
position_t   = y[:,1].long()
highest_t    = y[:,2].long()
reg_t        = y[:,3:4].float()
candle_t     = y[:,4:8].float()

dataset = TensorDataset(X, trend_t, position_t, highest_t, reg_t, candle_t)
train_len = int(len(dataset)*0.8)
train_ds, val_ds = random_split(dataset, [train_len, len(dataset)-train_len])
train_loader = DataLoader(train_ds, BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   BATCH)

class MetaNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim,64),nn.ReLU(), nn.Linear(64,32),nn.ReLU())
        self.head_trend    = nn.Linear(32,3)
        self.head_position = nn.Linear(32,3)
        self.head_highest  = nn.Linear(32,3)
        self.head_reg      = nn.Linear(32,1)
        self.head_candle   = nn.Linear(32,4)
    def forward(self,x):
        h=self.shared(x)
        return {
            "trend":    self.head_trend(h),
            "position": self.head_position(h),
            "highest":  self.head_highest(h),
            "reg":      self.head_reg(h),
            "candle":   self.head_candle(h)
        }

net = MetaNet(X.shape[1]).to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=LR)
ce  = nn.CrossEntropyLoss(); mse = nn.MSELoss()

def step(loader, train=True):
    net.train(train)
    tot=0; acc=0
    for xb,t,p,h,r,c in loader:
        xb,t,p,h,r,c=[a.to(DEVICE) for a in (xb,t,p,h,r,c)]
        out = net(xb)
        loss = ce(out["trend"],t)+ce(out["position"],p)+ce(out["highest"],h)\
             + mse(out["reg"],r)+ mse(out["candle"],c)*0.5
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot += xb.size(0); acc += (out["trend"].argmax(1)==t).sum().item()
    return acc/tot

for ep in range(EPOCHS):
    tr  = step(train_loader, True)
    val = step(val_loader,   False)
    print(f"ep {ep+1:02d}  acc  train {tr:.3f}  val {val:.3f}")

torch.save(net.state_dict(),
           f"meta/meta_mlp_{SYMBOL}_{TARGET_INTERVAL}.pt")
print("✔ meta model saved")
