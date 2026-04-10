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
    logging.info("It's 4:30 AM PST! Running Local Discovery via laptop CPU...")
    try:
        # Run Intraday Discovery
        logging.info("Starting Intraday Discovery...")
        subprocess.run([sys.executable, "kaggle_jobs/intraday/run_discovery.py"], check=True)
        
        # Run Swing Discovery
        logging.info("Starting Swing Discovery...")
        subprocess.run([sys.executable, "kaggle_jobs/swing/run_discovery.py"], check=True)
        
        logging.info("Discovery Generation Finished! Committing to GitHub...")
        subprocess.run(["git", "add", "data/"], check=True)
        subprocess.run(["git", "commit", "-m", f"Auto Local Discovery Update: {datetime.now().strftime('%Y-%m-%d')}"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        
        logging.info("Successfully pushed today's cache to GitHub.")
        
    except Exception as e:
        logging.error(f"Error running automated discovery: {e}")

# Note: The server/laptop time needs to be aligned with the scheduled time. 
# Assuming the laptop is on PST, we schedule for "04:30".
schedule.every().monday.at("04:30").do(run_job)
schedule.every().tuesday.at("04:30").do(run_job)
schedule.every().wednesday.at("04:30").do(run_job)
schedule.every().thursday.at("04:30").do(run_job)
schedule.every().friday.at("04:30").do(run_job)

logging.info("Kaggle Scheduler Started. Waiting for 4:30 AM PST (Weekdays)...")

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(60)