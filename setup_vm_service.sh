#!/bin/bash
# Setup script for running the trading pipeline as a background service on the GCP VM.

# 1. Install dependencies
sudo apt-get update && sudo apt-get install -y python3-pip git
pip3 install alpaca-trade-api pandas numpy

# 2. Setup the project
mkdir -p ~/alpaca-trading-algo
cd ~/alpaca-trading-algo
# You would clone your repo here: git clone <your-repo-url> .

# Sync logic: Ensure pipeline pulls latest changes before running
echo "0 * * * * cd ~/alpaca-trading-algo && git pull origin main" | crontab -

# 3. Create a systemd service file to keep the scheduler/monitor running
# This ensures it restarts automatically even if the process dies or VM reboots.
cat <<EOF | sudo tee /etc/systemd/system/alpaca-trader.service
[Unit]
Description=Alpaca Trading Pipeline
After=network.target

[Service]
User=$USER
WorkingDirectory=$HOME/alpaca-trading-algo
ExecStart=/usr/bin/python3 run_alpaca_live_trade.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF

# 4. Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable alpaca-trader
sudo systemctl start alpaca-trader

echo "Service deployed. Use 'sudo systemctl status alpaca-trader' to check status."