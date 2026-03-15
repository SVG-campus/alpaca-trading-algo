@echo off
echo Starting Daily Kaggle Automation Scheduler...
echo This will run every morning at 3:00 AM PST.
echo You can minimize this window.
echo Logs are saved to data/kaggle_scheduler.log and data/kaggle_pull.log

:: Ensure data folder exists
mkdir data 2>nul

title Daily Kaggle Discovery Automation
python scripts/setup_kaggle_creds.py
python scripts/schedule_kaggle.py

pause