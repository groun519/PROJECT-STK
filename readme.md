# PROJECT-STK

PROJECT-STK는 시계열 기반 주가 예측 모델을 다중 분봉 구조로 학습하고,  
예측 결과를 종합하여 실제 매매 판단에 활용할 수 있는 트레이딩 AI 시스템입니다.

본 프로젝트는 다음과 같은 단계를 포함합니다:

- 분봉별 개별 예측 모델 학습 (LSTM/Transformer 기반)
- 예측 결과 앙상블 (soft voting 방식)
- 예측 성능 기반 메타모델 보정
- OHLC 회귀 예측 및 포지션 추천
- Streamlit 기반 대시보드 제공

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 사용법

### 1. 데이터 수집

```bash
python source/find_symbols.py
```

- `nasdaqlisted.txt`에 명시된 종목에 대해 야후파이낸스 데이터를 수집합니다.
- 결과는 `cache/` 디렉토리에 저장됩니다.

### 2. 모델 학습

#### 분봉별 예측 모델 학습

```bash
python source/model_base/train_ensemble_model.py
```

- 각 분봉(2m, 5m, 15m, 30m, 60m, 1d)에 대해 독립적인 예측 모델을 학습합니다.

#### 메타모델 학습

```bash
python source/model_meta/make_meta_dataset.py
python source/model_meta/train_meta_model.py
```

- 개별 모델의 예측 결과를 입력으로 하여 성능을 보정하는 메타모델을 학습합니다.

#### 캔들 회귀 예측 모델 학습 (선택)

```bash
python source/-/train_candle_model.py
```

- 다음 시점의 시가(Open), 고가(High), 저가(Low), 종가(Close)를 회귀 방식으로 예측합니다.

---

### 3. 예측

#### 방향성 앙상블 예측

```bash
python source/-/predict_ensemble.py
```

- 분봉별 모델의 예측을 종합하여 최종 방향성을 예측합니다.

#### 메타모델 예측

```bash
python source/model_meta/predict_meta.py
```

- 메타모델을 통해 보정된 예측 결과를 생성합니다.

#### 캔들 예측

```bash
python source/-/predict_candle.py
```

- OHLC 형태의 회귀 결과를 출력합니다.

#### 포지션 추천

```bash
python source/-/recommend_position.py
```

- 예측 결과와 기술지표를 조합하여 포지션(Market, Long, Short, Wait 등)을 결정합니다.

---

### 4. 시각화

```bash
streamlit run source/-/streamlit_app.py
```

- 예측 결과와 수익률, 캔들 패턴 등을 웹 기반으로 시각화합니다.

---

## 구성 개요

- `source/model_base/`: 개별 분봉 예측 모델 (Transformer 등)
- `source/model_meta/`: 메타모델 학습 및 예측
- `source/data/`: 데이터 수집 및 전처리, 기술지표 계산
- `source/-/`: 실행 진입점 스크립트
- `cache/`: 수집된 분봉별 CSV 데이터
- `meta/`: 메타모델용 데이터셋 저장소
- `streamlit_app.py`: 대시보드 실행 엔트리

---

## 모델 구조 요약

```
[개별 분봉 모델] → [앙상블] → [메타모델] → [포지션 추천]
                               ↘ [캔들 회귀 예측]
```

---

## 주요 기술

- Python 3.11
- PyTorch
- Scikit-learn
- yfinance
- Streamlit

---

## 개발 계획

- 강화학습(PPO 등) 기반 포지션 제어 통합
- 실시간 시뮬레이터 및 백테스트 도입
- 리스크 기반 수익률 평가 지표 추가
- 멀티 심볼 최적화 모델 확장

---

## 라이선스

MIT License
