from stable_baselines3 import PPO
from env_trading_mtf import TradingEnvMTF
from data_utils import load_multitimeframe_data
from config import SYMBOL, START_DATE, END_DATE

data = load_multitimeframe_data(SYMBOL, start=START_DATE, end=END_DATE)
env = TradingEnvMTF(data)

model = PPO.load("ppo_trading_mtf", env=env)
model.learn(total_timesteps=100_000)
model.save("ppo_trading_mtf")
