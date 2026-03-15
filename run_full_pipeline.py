import os
import subprocess
import json
import time

def run_pipeline():
    # 0. Validate Research (Double checking intraday signal logic)
    print("Validating research signals...")
    subprocess.run(["python", "scripts/verify_and_update_research.py"], check=True)

    # 1. Run Kaggle Discovery
    print("Running Kaggle Discovery...")
    try:
        # Assuming run_discovery_kaggle_gpu.py handles the logic
        subprocess.run(["python", "run_discovery_kaggle_gpu.py"], check=True)
    except Exception as e:
        print(f"Discovery failed: {e}")
        return

    # 2. Execute Trades
    print("Executing trades from cache...")
    try:
        subprocess.run(["python", "run_trade_from_cache.py"], check=True)
    except Exception as e:
        print(f"Trading execution failed: {e}")

    # 3. Provision/Sync GCP VM (Conceptual)
    # Using the project defined in env
    gcp_project = os.environ.get('GCP_API_PROJECT')
    print(f"Syncing with GCP project: {gcp_project}")

if __name__ == '__main__':
    run_pipeline()