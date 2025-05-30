import numpy as np
from dataset_builder import build_generic_dataset, SYMBOL_LIST
from _data_config import TARGET_INTERVAL

def test_pipeline():
    print(f"📦 전체 파이프라인 테스트 시작 - 타겟 분봉: {TARGET_INTERVAL}")
    X, y = build_generic_dataset(TARGET_INTERVAL)

    if X is None or y is None:
        print("❌ 데이터셋 생성 실패")
        return

    print(f"\n✅ 총 샘플 수: {X.shape[0]}개")
    print(f"📐 입력 시퀀스 차원: {X.shape[1:]}")
    print(f"🔍 첫 샘플 확인:")
    print(f"  X[0] shape: {X[0].shape}")
    print(f"  y[0] 라벨: {y[0]}")

    # 라벨 분포
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n🎯 라벨 분포:")
    for u, c in zip(unique, counts):
        print(f"  - 라벨 {u}: {c}개 ({(c/len(y))*100:.2f}%)")

if __name__ == "__main__":
    test_pipeline()
