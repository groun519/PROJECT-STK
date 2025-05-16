import matplotlib.pyplot as plt

def plot_agent_performance(history):
    steps = history["step"]
    
    # 1. 총 자산 추이
    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["asset"], label="Total Asset ($)", color="black")
    plt.title("Agent Asset Value Over Time")
    plt.xlabel("Step")
    plt.ylabel("Asset ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. 가격 + 매수/매도 표시
    buy_signals = [i for i, a in enumerate(history["action"]) if a > 0.05]
    sell_signals = [i for i, a in enumerate(history["action"]) if a < -0.05]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, history["price"], label="Price", alpha=0.7)
    plt.scatter([steps[i] for i in buy_signals],
                [history["price"][i] for i in buy_signals],
                color='green', label='Buy', marker='^')
    plt.scatter([steps[i] for i in sell_signals],
                [history["price"][i] for i in sell_signals],
                color='red', label='Sell', marker='v')
    plt.title("Price with Buy/Sell Actions")
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
