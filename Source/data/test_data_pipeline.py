# test_dataset_pipeline.py

import pandas as pd
import numpy as np
import traceback
from dataset_builder import build_generic_dataset
from _data_config import TARGET_INTERVAL, SYMBOL_LIST

def test_dataset_generation():
    print(f"\n🧪 [테스트 시작] {TARGET_INTERVAL} 기준 데이터셋 생성 테스트")

    try:
        # 1. 데이터셋 생성 호출
        X, y = build_generic_dataset(interval=TARGET_INTERVAL)

        # 2. None 또는 빈 배열 여부 확인
        if X is None or y is None:
            print("❌ 데이터셋 생성 실패: X 또는 y가 None입니다.")
            return
        if len(X) == 0 or len(y) == 0:
            print("❌ 데이터셋 생성 실패: X 또는 y 길이가 0입니다.")
            return

        # 3. 형상 및 타입 출력
        print(f"\n✅ 데이터셋 생성 완료")
        print(f"📐 X type: {type(X)}, shape: {X.shape}")
        print(f"📐 y type: {type(y)}, shape: {y.shape}")

        # 4. 라벨 분포 확인 (분류형인 경우)
        unique_y = np.unique(y)
        if len(unique_y) < 20:
            print(f"\n🔍 라벨 분포:")
            y_series = pd.Series(y)
            print(y_series.value_counts().sort_index())
        else:
            print(f"\n🔍 라벨 값 예시 (상위 10개): {y[:10]}")

        # 5. 첫 번째 샘플 미리보기
        print(f"\n🔎 첫 번째 샘플 X[0] (shape: {getattr(X[0], 'shape', 'unknown')}):")
        print(X[0])

    except Exception as e:
        print(f"\n❌ 예외 발생: {type(e).__name__}")
        print(f"🧵 에러 메시지: {e}")
        print("🔧 전체 traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_generation()
