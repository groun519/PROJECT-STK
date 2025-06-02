from dataset_builder import build_lstm_dataset

def test_dataset_generation():
    symbol = "AAPL"
    X, y = build_lstm_dataset(symbol)
    print("X shape:", X.shape)
    for k in y:
        print(f"y[{k}] shape: {y[k].shape}")

if __name__ == "__main__":
    test_dataset_generation()
