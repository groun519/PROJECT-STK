@echo off
echo Cleaning up model files...

:: Delete trained model
if exist ppo_trading_mtf.zip (
    del ppo_trading_mtf.zip
    echo Model file deleted.
) else (
    echo Model file not found.
)

:: Delete training log
if exist live_log.csv (
    del live_log.csv
    echo Log file deleted.
) else (
    echo Log file not found.
)

:: Delete TensorBoard logs
if exist ppo_logs (
    rmdir /s /q ppo_logs
    echo TensorBoard log folder deleted.
) else (
    echo TensorBoard folder not found.
)

:: Delete Python cache
if exist __pycache__ (
    rmdir /s /q __pycache__
    echo Python __pycache__ folder deleted.
) else (
    echo __pycache__ folder not found.
)

echo Reset complete. Closing...
timeout /t 2 >nul
exit
