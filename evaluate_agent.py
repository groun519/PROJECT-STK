from stable_baselines3 import PPO
from env_trading_mtf import TradingEnvMTF
from data_utils import load_multitimeframe_data
from config import SYMBOL, START_DATE, END_DATE
from visualization import plot_agent_performance

data = load_multitimeframe_data(SYMBOL, start=START_DATE, end=END_DATE)
env = TradingEnvMTF(data)
model = PPO.load("ppo_trading_mtf")

obs = env.reset()
done = False

history = {
    "step": [],
    "price": [],
    "asset": [],
    "action": [],
    "balance": [],
    "holdings": []
}

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    price = data["60m"].iloc[env.current_step]["Close"]
    history["step"].append(env.current_step)
    history["price"].append(price)
    history["asset"].append(info["asset"])
    history["action"].append(float(action[0]))
    history["balance"].append(info["balance"])
    history["holdings"].append(info["holdings"])

plot_agent_performance(history)
