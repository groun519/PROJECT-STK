from data_utils import load_multitimeframe_data
from env_trading_mtf import TradingEnvMTF

data = load_multitimeframe_data("TSLA", start="2024-04-01", end="2024-05-01")
env = TradingEnvMTF(data)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # 랜덤 행동
    obs, reward, done, info = env.step(action)
    print(f"step: {env.current_step}, reward: {reward}")
