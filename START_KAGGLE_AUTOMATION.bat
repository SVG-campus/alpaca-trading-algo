@echo off
echo Starting Daily Kaggle Automation Scheduler...
echo This will run every morning at 4:30 AM PST (Weekdays).
echo You can minimize this window.
echo Logs are saved to data/kaggle_scheduler.log and data/kaggle_pull.log

:: Ensure data folder exists
mkdir data 2>nul

title Daily Local Discovery Automation
python scripts/schedule_kaggle.py

pause