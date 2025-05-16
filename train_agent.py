from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env_trading_mtf import TradingEnvMTF
from data_utils import load_multitimeframe_data
from config import SYMBOL, START_DATE, END_DATE

# Load data
data = load_multitimeframe_data(SYMBOL, start=START_DATE, end=END_DATE)

# Debug print for 60m close prices
df = data["60m"]
print("✅ 60m 분봉 종가 고유값 수:", df["close"].nunique())
print(df["close"].value_counts().head())
print(df.head())

# Initialize environment and check
env = TradingEnvMTF(data)
check_env(env, warn=True)

# Train agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs/")
model.learn(total_timesteps=100_000)
model.save("ppo_trading_mtf")
