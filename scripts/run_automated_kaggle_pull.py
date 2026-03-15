import os
import time
import subprocess
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler("data/kaggle_pull.log"), logging.StreamHandler()])

def run_kaggle_job(job_name, folder, output_dir):
    import sys
    import os
    kaggle_exe = os.path.join(os.environ.get('APPDATA'), 'Python', 'Python314', 'Scripts', 'kaggle.exe')
    if not os.path.exists(kaggle_exe):
        kaggle_exe = "kaggle" # Fallback to PATH
    
    logging.info(f"Pushing Kaggle kernel for {job_name} from {folder}...")
    try:
        subprocess.run([kaggle_exe, "kernels", "push", "-p", folder], check=True)
    except Exception as e:
        logging.error(f"Failed to push kernel {job_name}: {e}")
        return False
        
    # Read metadata to get kernel ID
    import json
    with open(os.path.join(folder, 'kernel-metadata.json'), 'r') as f:
        meta = json.load(f)
    kernel_id = meta['id']
    
    logging.info(f"Kernel {kernel_id} pushed. Waiting for it to complete...")
    
    # Wait for completion
    timeout = 3600 # 1 hour
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run([kaggle_exe, "kernels", "status", kernel_id], capture_output=True, text=True)
            output = result.stdout.lower()
            
            if "complete" in output:
                logging.info(f"Kernel {kernel_id} completed successfully!")
                break
            elif "error" in output or "failed" in output or "cancelled" in output:
                logging.error(f"Kernel {kernel_id} failed with status: {output}")
                return False
            
            logging.info(f"Kernel status: {output.strip()}. Waiting...")
        except Exception as e:
            logging.warning(f"Error checking status: {e}")
            
        time.sleep(60)
        
    if time.time() - start_time >= timeout:
        logging.error("Kernel execution timed out.")
        return False
        
    # Download output
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Downloading output to {output_dir}...")
    try:
        subprocess.run([kaggle_exe, "kernels", "output", kernel_id, "-p", output_dir], check=True)
        logging.info("Download complete.")
        return True
    except Exception as e:
        logging.error(f"Failed to download output: {e}")
        return False

def push_to_github():
    logging.info("Committing changes to GitHub...")
    try:
        subprocess.run(["git", "add", "data/"], check=True)
        subprocess.run(["git", "commit", "-m", f"Auto Kaggle Discovery Update: {datetime.now().strftime('%Y-%m-%d')}"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        logging.info("Successfully pushed to GitHub.")
    except Exception as e:
        logging.error(f"GitHub push failed: {e}")

def main():
    logging.info("Starting Daily Kaggle Automation...")
    
    # 1. Run Intraday Job
    success_intraday = run_kaggle_job("Intraday V8", "kaggle_jobs/intraday", "data/intraday")
    
    # 2. Run Swing Job (If intraday finished or failed, we still try swing)
    success_swing = run_kaggle_job("Swing V7", "kaggle_jobs/swing", "data/swing")
    
    if success_intraday or success_swing:
        push_to_github()
    else:
        logging.error("Both jobs failed. Skipping GitHub push.")
        
if __name__ == '__main__':
    main()