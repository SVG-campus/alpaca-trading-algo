@echo off
echo Starting Alpaca Live Trading Automation...
echo The terminal will stay open but you can minimize it.
echo All logs are saved to data/alpaca_trade.log as well.

:: Ensure data folder exists
mkdir data 2>nul

title Alpaca Live Intraday Trader
python run_alpaca_live_trade.py

pause