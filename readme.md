# 📈 PROJECT-STK: 주식 예측 멀티모델 시스템

**PROJECT-STK**는 시계열 데이터를 기반으로 다양한 머신러닝 모델을 활용해  
주가의 방향성, 캔들 형태, 그리고 최종 포지션까지 예측하는 통합 트레이딩 모델링 프로젝트입니다.

> 여러 분봉(2m, 5m, 15m, 30m, 1h, 1d)의 데이터를 기반으로  
> 개별 모델 → 앙상블 → 메타모델 → 포지션 추천까지  
> "진짜 쓸 수 있는" 다단계 예측 시스템을 구축했습니다.

---

## 🚀 시작하기

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

(※ `torch`, `yfinance`, `scikit-learn`, `streamlit` 등 포함)

---

### 2. 데이터 다운로드

```bash
python source/find_symbols.py
```

- `nasdaqlisted.txt` 기반 주요 종목(예: TSLA, AAPL 등)의 데이터를 다운로드합니다.
- 결과는 `cache/` 폴더에 저장됩니다.

---

### 3. 분봉별 모델 학습

```bash
python source/model_base/train_ensemble_model.py
```

- 각 타임프레임(예: 15분봉, 1시간봉)에 대해 Transformer 기반 예측 모델을 학습합니다.
- 예측 대상은 **상승 / 하락 / 관망 (3분류)** 입니다.

---

### 4. 앙상블 예측 실행

```bash
python source/-/predict_ensemble.py
```

- 위에서 학습한 분봉별 모델을 **통합(soft voting)** 하여 최종 방향을 예측합니다.

---

### 5. 메타모델 학습 (선택)

```bash
python source/model_meta/make_meta_dataset.py
python source/model_meta/train_meta_model.py
```

- 이전 예측 결과 + 기술 지표를 바탕으로 보정하는 **2차 메타모델**을 학습합니다.

---

### 6. 캔들 회귀 예측 (OHLC)

```bash
python source/-/predict_candle.py
```

- 다음 시점의 캔들(시가, 고가, 저가, 종가)을 수치로 예측합니다.

---

### 7. 포지션 추천

```bash
python source/-/recommend_position.py
```

- 최종 예측 결과를 바탕으로  
  **Market / Long / Short / Wait** 중 하나의 트레이딩 포지션을 추천합니다.

---

## 📊 시각화 대시보드

```bash
streamlit run source/-/streamlit_app.py
```

- 실시간 예측 결과, 수익률, 포지션 추이 등을 웹 UI로 확인할 수 있습니다.

---

## 🧠 모델 구성 요약

이 시스템은 3단계로 예측을 수행합니다:

```
1. 분봉별 모델 (LSTM, Transformer 등) → 예측
2. 앙상블 (soft voting) → 통합 방향 예측
3. 메타모델 (MLP 등) → 과거 성과 기반 보정
4. 포지션 추천기 → 실제 트레이딩 판단 지원
```

기술지표(RSI, MACD 등)는 각 단계에서 활용됩니다.

---

## 🔮 앞으로 할 일

- [ ] 강화학습 기반 전략(PPO 등)과 통합
- [ ] 실시간 매매 시뮬레이터 구현
- [ ] 종목 추천 시스템 확장
- [ ] 리스크 기반 포트폴리오 관리

---

## 📝 참고

- 종목 목록: `nasdaqlisted.txt`
- 데이터 캐시: `cache/`
- 메타 데이터셋: `meta/`
- 전체 설정: `source/-/config.py` 등

---

## 💡 만든 이유

- “진짜 실전에 쓸 수 있는 주식 예측 시스템이 가능한가?”  
- “단순 분류보다, 예측 결과를 실제 트레이딩 판단에 연결할 수 있을까?”

이 두 가지 질문에 답하기 위해 시작된 프로젝트입니다.
