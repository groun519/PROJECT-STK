from stable_baselines3 import PPO
from env_trading_mtf import TradingEnvMTF
from data_utils import load_multitimeframe_data
from config import SYMBOL, START_DATE, END_DATE
import time
from pathlib import Path
from datetime import datetime
import sys
import re

def normalize_model_name(name: str) -> str:
    return re.sub(r'[^A-Z0-9_]', '_', name.strip().upper())

def list_existing_models(models_root: Path):
    return [d for d in models_root.iterdir() if d.is_dir() and (d / "model.zip").exists()]

def select_existing_model(existing_models):
    print("\n========== 기존 모델 목록 ==========")
    for idx, model_dir in enumerate(existing_models, 1):
        print(f"{idx}. {model_dir.name}")
    selected = input("불러올 모델 번호 선택: ").strip()
    if not selected.isdigit() or int(selected) < 1 or int(selected) > len(existing_models):
        print("잘못된 입력입니다.")
        sys.exit()
    return existing_models[int(selected) - 1]

def create_new_model_dir(models_root: Path) -> Path:
    raw_name = input("모델명을 입력하세요 (예: macd_rsi_v1): ")
    base_name = normalize_model_name(raw_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"model_{base_name}_{timestamp}"
    model_dir = models_root / version
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def main():
    MODELS_ROOT = Path("models")
    existing_models = list_existing_models(MODELS_ROOT)

    print("========== 실행 모드 선택 ==========")
    if existing_models:
        print("1. 기존 모델 선택")
        print("2. 새 모델 생성")
        print("3. 종료")
        choice = input("선택 (1/2/3): ").strip()
        if choice == "3":
            print("종료합니다.")
            sys.exit()
        elif choice == "1":
            model_dir = select_existing_model(existing_models)
        elif choice == "2":
            model_dir = create_new_model_dir(MODELS_ROOT)
        else:
            print("잘못된 입력입니다.")
            sys.exit()
    else:
        print("※ 기존 모델이 없습니다.")
        print("1. 새 모델 생성")
        print("2. 종료")
        choice = input("선택 (1/2): ").strip()
        if choice == "2":
            print("종료합니다.")
            sys.exit()
        elif choice == "1":
            model_dir = create_new_model_dir(MODELS_ROOT)
        else:
            print("잘못된 입력입니다.")
            sys.exit()

    model_path = model_dir / "model.zip"
    log_path = model_dir / "live_log.csv"

    # 데이터 및 환경 생성
    data = load_multitimeframe_data(SYMBOL, start=START_DATE, end=END_DATE)
    env = TradingEnvMTF(data, log_path=str(log_path))

    # 모델 로드 or 생성
    if model_path.exists():
        print(f"[Model found] Loading model from {model_path}")
        model = PPO.load(str(model_path), env=env)
    else:
        print("[No model found] Creating new model...")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(model_dir / "logs"))

    # 반복 학습
    while True:
        print("[Training step started]")
        model.learn(total_timesteps=10000, reset_num_timesteps=False)
        model.save(str(model_path))
        print("[Model saved to]", model_path)
        time.sleep(3)

if __name__ == "__main__":
    main()
