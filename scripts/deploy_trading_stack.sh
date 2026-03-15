#!/bin/bash
# GCP VM Deployment Script
# Usage: ./deploy_trading_stack.sh <gcp_project_id>

PROJECT_ID=$1
ZONE="us-central1-a"
INSTANCE_NAME="alpaca-trading-engine"

# 1. Provision VM (Requires gcloud CLI authenticated)
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=e2-standard-2 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud

# 2. Setup Remote Environment
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command '
    sudo apt update && sudo apt install -y python3-pip git
    git clone https://github.com/SVG-campus/alpaca-trading-algo.git
    cd alpaca-trading-algo
    pip3 install alpaca-py yfinance pandas numpy scikit-learn google-cloud-compute
    # .env needs to be transferred or populated via secret manager
'

# 3. Schedule Jobs
# This adds the cron jobs to the remote machine
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command '
    (crontab -l 2>/dev/null; echo "0 8 * * 1-5 cd ~/alpaca-trading-algo && /usr/bin/python3 run_discovery_kaggle_gpu.py >> discovery.log 2>&1") | crontab -
    (crontab -l 2>/dev/null; echo "0 10 * * 1-5 cd ~/alpaca-trading-algo && /usr/bin/python3 run_alpaca_live_trade.py >> trading.log 2>&1") | crontab -
'