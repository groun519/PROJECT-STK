import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os

class TradingEnvMTF(gym.Env):
    def __init__(self, mtf_data, window_size=30, initial_balance=10000, asset_reset_threshold=20000):
        self.mtf_data = mtf_data
        self.window_size = window_size
        self.intervals = ["2m", "5m", "15m", "30m", "60m", "1d"]
        self.initial_balance = initial_balance
        self.asset_reset_threshold = asset_reset_threshold

        self.reset()  # current_step 등 초기화 먼저 수행

        self.max_steps = min(len(df) for df in mtf_data.values())
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        example_state = self._get_state()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=example_state.shape, dtype=np.float32)

    def _get_state(self):
        slices = []
        for interval in self.intervals:
            df = self.mtf_data[interval]
            slice = df.iloc[self.current_step - self.window_size : self.current_step]

            # ✅ slice 길이 부족하면 0으로 패딩
            if len(slice) < self.window_size:
                pad = np.zeros((self.window_size - len(slice), slice.shape[1]))
                slice = np.vstack([pad, slice.values])
            else:
                slice = slice.values

            slices.append(slice)

        tech_stack = np.concatenate(slices, axis=0).astype(np.float32)
        current_price = self.mtf_data["60m"].iloc[self.current_step]["close"]
        position_value = self.holdings * current_price
        total_asset = self.balance + position_value
        profit_rate = (total_asset - self.initial_balance) / self.initial_balance

        agent_state = np.array([
            self.balance / 10000,
            self.holdings / 100,
            total_asset / 10000,
            profit_rate
        ], dtype=np.float32)

        return np.concatenate([tech_stack.flatten(), agent_state])

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.holdings = 0
        self.total_asset = self.balance
        self.total_reward = 0.0

        # ✅ 덮어쓰지 않도록 조건부 생성
        if not os.path.exists("live_log.csv"):
            with open("live_log.csv", "w") as f:
                f.write("step,price,action,balance,holdings,asset\n")

        return self._get_state(), {}

    def step(self, action):
        action = float(np.array(action).flatten()[0])
        action = np.clip(action, -1.0, 1.0)

        price = self.mtf_data["60m"].iloc[self.current_step]["close"]
        prev_asset = self.balance + self.holdings * price

        if action < 0:
            sell_amount = int(self.holdings * abs(action))
            self.holdings -= sell_amount
            self.balance += sell_amount * price
        elif action > 0:
            max_buy = int(self.balance // price)
            buy_amount = int(max_buy * action)
            self.holdings += buy_amount
            self.balance -= buy_amount * price

        self.total_asset = self.balance + self.holdings * price
        reward = (self.total_asset - prev_asset) / prev_asset
        self.total_reward += reward

        if self.total_asset >= 30000 or self.total_asset <= 3300:
            self.balance = self.initial_balance
            self.holdings = 0
            self.total_asset = self.initial_balance

        self.current_step += 1
        done = self.current_step >= self.max_steps - 1

        with open("live_log.csv", "a") as f:
            f.write(f"{self.current_step},{price},{action},{self.balance},{self.holdings},{self.total_asset}\n")

        obs = self._get_state()
        info = {
            "step": self.current_step,
            "balance": self.balance,
            "holdings": self.holdings,
            "asset": self.total_asset,
            "total_reward": self.total_reward
        }
        return obs, reward, done, False, info
