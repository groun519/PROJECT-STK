import torch
import numpy as np
import pandas as pd
from train_model import DirectionalLSTM
from data_utils import build_lstm_dataset
from config import SYMBOL_LIST

def run_inference(symbol):
    print(f"🔍 [{symbol}] 예측 실행 중...")
    X, y = build_lstm_dataset(symbol)
    if X is None or y is None:
        print(f"❌ {symbol} 데이터 로딩 실패")
        return

    input_size = X.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DirectionalLSTM(input_size)
    model_path = f"lstm_{symbol}.pt"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"❌ 모델 파일 없음: {model_path}")
        return

    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_tensor).argmax(dim=1).cpu().numpy()

    df = pd.DataFrame({
        "step": np.arange(len(outputs)),
        "prediction": outputs,
        "actual": y
    })

    out_path = f"inference_{symbol}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ 저장 완료: {out_path}")

if __name__ == "__main__":
    for symbol in SYMBOL_LIST:
        run_inference(symbol)
