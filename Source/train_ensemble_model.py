import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from data_utils import build_generic_dataset
from model_transformer import LSTMTransformer

# 🔧 설정값
INTERVALS = ["2m", "5m", "15m", "30m", "60m", "1d"]
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path

for interval in INTERVALS:
    print(f"\n🚀 [{interval}] 분봉 기준 범용 모델 학습 시작")
    run_start = time.time()

    # ✅ 수정된 호출부: 불필요한 인자 제거
    X, y = build_generic_dataset(interval=interval)
    if X is None or y is None:
        print(f"❌ [{interval}] 학습 데이터 생성 실패 → 건너뜀")
        continue

    label_counts = np.unique(y, return_counts=True)
    print(f"📊 라벨 분포: {dict(zip(*label_counts))}")

    writer = SummaryWriter(log_dir=f"runs/{interval}")
    writer.add_text("Hyperparameters", f"Interval={interval}, BatchSize={BATCH_SIZE}, Epochs={EPOCHS}, LR={LEARNING_RATE}")
    for label, count in zip(*label_counts):
        writer.add_scalar("LabelDistribution/Class_" + str(label), count, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X.shape[2]
    model = LSTMTransformer(input_size=input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss, correct = 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(dim=1) == yb).sum().item()

        acc = correct / len(train_ds)
        writer.add_scalar("Train/Loss", total_loss, epoch)
        writer.add_scalar("Train/Accuracy", acc, epoch)
        writer.add_scalar("Epoch/Duration_sec", time.time() - epoch_start, epoch)

        print(f"[{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f} | Train Acc: {acc*100:.2f}%")

    os.makedirs("models", exist_ok=True)
    model_path = f"models/direction_model_{interval}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ 저장 완료: {model_path}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        preds = logits.argmax(dim=1).cpu().numpy()

    val_acc = accuracy_score(y_test, preds)
    writer.add_scalar("Val/Accuracy", val_acc, EPOCHS)
    print("\n📊 [검증 결과]")
    print(classification_report(y_test, preds, digits=4))

    cm_path = f"runs/{interval}/confusion_matrix.png"
    saved_cm_path = plot_confusion_matrix(y_test, preds, class_names=["Down", "Neutral", "Up"], save_path=cm_path)
    writer.add_image("Val/ConfusionMatrix", 
                     torch.tensor(plt.imread(saved_cm_path)).permute(2, 0, 1), 
                     global_step=EPOCHS)

    writer.add_scalar("Training/TotalTime_sec", time.time() - run_start, 0)
    writer.close()
