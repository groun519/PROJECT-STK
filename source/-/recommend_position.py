import numpy as np

def recommend_position(avg_probs: np.ndarray, individual_probs: dict[str, np.ndarray]) -> str:
    """
    예측 확률 기반으로 포지션 추천을 판단합니다.

    Args:
        avg_probs: soft voting 평균 확률 [하락, 보합, 상승]
        individual_probs: 분봉별 확률 예측 값 {interval: [하락, 보합, 상승]}

    Returns:
        추천 포지션: "매수", "매도", "관망"
    """
    down, neutral, up = avg_probs
    total_intervals = len(individual_probs)
    up_votes = sum(1 for p in individual_probs.values() if np.argmax(p) == 2)
    down_votes = sum(1 for p in individual_probs.values() if np.argmax(p) == 0)

    # 강한 매수 조건
    if up >= 0.65 and down <= 0.2 and up_votes >= total_intervals * 0.6:
        return "📈 매수 추천"

    # 강한 매도 조건
    if down >= 0.65 and up <= 0.2 and down_votes >= total_intervals * 0.6:
        return "📉 매도 추천"

    # 그 외는 관망
    return "⏸ 관망 추천"
