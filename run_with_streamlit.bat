@echo off
echo [Launching Streamlit dashboard...]
start cmd /k "streamlit run streamlit_app.py"

echo [Checking for existing model...]
if exist ppo_trading_mtf.zip (
    echo Model found. Continuing training...
    python train_continue.py
) else (
    echo No model found. Starting new training...
    python train_agent.py
)
