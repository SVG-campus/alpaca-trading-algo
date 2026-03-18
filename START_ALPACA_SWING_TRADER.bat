@echo off
echo Starting Alpaca Swing Trading Automation (Paper and Paper Max)...
echo Pulling latest code and discovery JSON from GitHub...
git pull origin main

echo The terminal will stay open but you can minimize it.
echo All logs are saved to data/swing/alpaca_swing_trade.log as well.

:: Ensure data folder exists
mkdir data\swing 2>nul

title Alpaca Swing Paper Trader
python run_alpaca_swing_trade.py

pause