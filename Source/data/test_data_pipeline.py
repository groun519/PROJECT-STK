# test_data_pipeline.py

from _data_config import INTERVALS, SYMBOL_LIST, WINDOW_SIZE, TARGET_SHIFT, LABELING_MODE, LABEL_THRESHOLD
from data_fetcher import fetch_merged_data
from indicators import add_indicators
from dataset_builder import build_dataset
from labeling_utils import (
    label_binary, label_three_class, label_position_class, label_return_regression
)

LABEL_FN_MAP = {
    "binary": label_binary,
    "three_class": label_three_class,
    "position": label_position_class,
    "regression": label_return_regression
}

symbol = SYMBOL_LIST[0]
interval = INTERVALS[0]

print(f"📡 테스트 시작: {symbol} / {interval}")

df = fetch_merged_data(symbol=symbol, interval=interval)
print("✅ 원본 데이터 수집 완료:", df.shape)

df = add_indicators(df)
print("✅ 기술지표 적용 완료:", df.columns.tolist())

X, y = build_dataset(df, window_size=WINDOW_SIZE)

label_fn = LABEL_FN_MAP[LABELING_MODE]
labels = label_fn(df, threshold=LABEL_THRESHOLD)

print("✅ 최종 라벨 예시:", labels.dropna().value_counts())
print("✅ 입력 데이터 shape:", X.shape, "| 라벨 shape:", y.shape)
