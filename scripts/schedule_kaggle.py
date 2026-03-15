import time
import schedule
import subprocess
import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler("data/kaggle_scheduler.log"),
    logging.StreamHandler()
])

def run_job():
    logging.info("It's 3:00 AM PST! Running Kaggle automated pull...")
    try:
        subprocess.run([sys.executable, "scripts/run_automated_kaggle_pull.py"])
        logging.info("Kaggle automated pull finished.")
    except Exception as e:
        logging.error(f"Error running Kaggle job: {e}")

# Note: The server/laptop time needs to be aligned with the scheduled time. 
# Assuming the laptop is on PST, we schedule for "03:00".
schedule.every().day.at("03:00").do(run_job)

logging.info("Kaggle Scheduler Started. Waiting for 3:00 AM PST...")

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(60)