import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from data_utils import build_lstm_dataset
from model_transformer import LSTMTransformer
from config import TARGET_SYMBOL

INTERVALS = ["2m", "5m", "15m", "30m", "60m", "1d"]
MODEL_DIR = "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(interval):
    print(f"\n📊 [평가 시작] {interval} 분봉 모델")
    os.environ["TARGET_INTERVAL"] = interval  # build_lstm_dataset 내부에서 사용

    X, y = build_lstm_dataset(symbol=TARGET_SYMBOL)
    if X is None or y is None:
        print(f"❌ {interval} 데이터셋 로딩 실패")
        return None

    # ▶ train/test split (20%)
    try:
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError as e:
        print(f"⚠️ {interval} 분봉 split 실패: {e}")
        return None

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    input_dim = X.shape[2]
    model_path = os.path.join(MODEL_DIR, f"direction_model_{interval}.pt")

    if not os.path.exists(model_path):
        print(f"❌ 모델 파일 없음: {model_path}")
        return None

    model = LSTMTransformer(input_size=input_dim).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return None

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, digits=4))

    return {
        "interval": interval,
        "accuracy": acc
    }

if __name__ == "__main__":
    print("📌 분봉별 모델 성능 평가 시작")
    results = []

    for interval in INTERVALS:
        result = evaluate_model(interval)
        if result:
            results.append(result)

    # 🔽 성능 요약 정렬 출력
    print("\n🏁 [요약 결과]")
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    for r in sorted_results:
        print(f"{r['interval']:>4} | Accuracy: {r['accuracy']*100:.2f}%")
