import os
import subprocess
from kaggle.api.kaggle_api_extended import KaggleApi

# This script can be run on the GCP VM to trigger Kaggle GPU discovery
# It utilizes the official Kaggle CLI API which is free.
# To use: pip install kaggle
# Ensure ~/.kaggle/kaggle.json exists with your credentials.

def trigger_kaggle_discovery():
    api = KaggleApi()
    api.authenticate()
    
    # Push updated code/data to your discovery kernel
    # Replace 'your-username/your-kernel-slug' with your actual kernel
    print("Pushing discovery kernel to Kaggle...")
    api.kernel_push(kernel_path='.', kernel_slug='alpaca-causal-discovery')
    
    print("Kaggle Kernel triggered successfully.")

if __name__ == "__main__":
    trigger_kaggle_discovery()