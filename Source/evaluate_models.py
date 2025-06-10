import os
import torch
import numpy as np
from data.dataset_builder import build_generic_dataset
from model_transformer import MultiHeadTransformer
from _model_config import DEVICE, MODEL_SAVE_PATH, LABEL_KEYS
from data._data_config import INTERVALS

from sklearn.metrics import accuracy_score, mean_absolute_error

def to_tensor(arr, dtype=torch.float):
    if arr is None:
        return None
    arr = np.array(arr)
    if arr.dtype == np.int32 or arr.dtype == np.int64:
        return torch.tensor(arr, dtype=torch.long)
    return torch.tensor(arr, dtype=dtype)

def evaluate_for_interval(interval):
    print(f"\n==== [{interval}] 평가 시작 ====")
    # 데이터셋 준비
    X, y_dict = build_generic_dataset(interval)
    if X is None or y_dict is None:
        print(f"[{interval}] 데이터셋 없음, 건너뜀")
        return None

    X_tensor = to_tensor(X).to(DEVICE)
    y_tensors = {k: to_tensor(y_dict[k]) for k in LABEL_KEYS if y_dict[k] is not None}

    # 모델 로드
    input_dim = X.shape[2]
    seq_len = X.shape[1]
    model_path = os.path.join(MODEL_SAVE_PATH, f"{interval}_model.pt")
    model = MultiHeadTransformer(input_dim=input_dim, seq_len=seq_len, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 예측
    with torch.no_grad():
        outputs = model(X_tensor)
        # outputs: dict 형태, 키는 LABEL_KEYS

    # 평가
    results = {}
    for key in LABEL_KEYS:
        y_true = y_tensors.get(key)
        y_pred = outputs.get(key)
        if y_true is None or y_pred is None:
            print(f"[WARN][{interval}][{key}] 평가 데이터 누락")
            continue
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        # 디버깅: shape 및 일부 값 확인
        print(f"[DEBUG][{interval}][{key}] y_true[:5]: {y_true[:5]}, y_pred[:5]: {y_pred[:5]}")
        if key in ["trend", "position"]:
            y_pred_label = np.argmax(y_pred, axis=1)
            acc = accuracy_score(y_true, y_pred_label)
            print(f"{key}: acc {acc:.4f}")
            results[key] = {"accuracy": acc}
        else:
            mae = mean_absolute_error(y_true, y_pred)
            print(f"{key}: mae {mae:.4f}")
            results[key] = {"mae": mae}
    print(f"[{interval}] 평가 결과:", results)
    return results

def main():
    all_results = {}
    for interval in INTERVALS:
        results = evaluate_for_interval(interval)
        if results is not None:
            all_results[interval] = results
    print("\n=== 전체 평가 결과 ===")
    for interval, res in all_results.items():
        print(f"{interval}:", res)

if __name__ == "__main__":
    main()
