import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from data.dataset_builder import build_generic_dataset
from model_transformer import MultiHeadTransformer
from _model_config import EPOCHS, BATCH_SIZE, DEVICE, MODEL_SAVE_PATH, LABEL_KEYS
from data._data_config import INTERVALS

def to_tensor(arr, dtype=torch.float):
    if arr is None:
        return None
    arr = np.array(arr)
    if arr.dtype == np.int32 or arr.dtype == np.int64:
        return torch.tensor(arr, dtype=torch.long)
    return torch.tensor(arr, dtype=dtype)

def dict_list_to_tensor_dict(y_dict, dtype=torch.float):
    # y_dict: [{...}, {...}, ...] (샘플 개수만큼의 dict list)
    result = {}
    for key in LABEL_KEYS:
        arr = [sample[key] for sample in y_dict if key in sample]
        arr = np.array(arr)
        # 1D/2D shape 체크 등 추가 가능
        result[key] = torch.tensor(arr, dtype=dtype)
    return result

def train_for_interval(interval, model_save_dir):
    print(f"===== [{interval}] 모델 학습 시작 =====")
    X, y_dict = build_generic_dataset(interval)
    
    if X is None or y_dict is None:
        print(f"[{interval}] 데이터셋 없음, 건너뜀")
        return

    X_tensor = to_tensor(X)
    y_tensors = dict_list_to_tensor_dict(y_dict)
    dataset = TensorDataset(X_tensor, *[y_tensors[k] for k in LABEL_KEYS])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X.shape[2]
    seq_len = X.shape[1]
    model = MultiHeadTransformer(input_dim=input_dim, seq_len=seq_len).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch[0].to(DEVICE)
            y_dict_batch = {k: batch[i+1].to(DEVICE) for i, k in enumerate(LABEL_KEYS)}
            outputs = model(x)
            loss, loss_dict = model.get_loss(outputs, y_dict_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"[{interval}] Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | " +
              " ".join([f"{k}:{loss_dict[k]:.4f}" for k in loss_dict]))

    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"{interval}_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[{interval}] 모델 저장 완료: {model_path}")

def main():
    model_save_dir = MODEL_SAVE_PATH
    for interval in INTERVALS:
        train_for_interval(interval, model_save_dir)

if __name__ == "__main__":
    main()
