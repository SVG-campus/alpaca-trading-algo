import requests
import json
import os
import logging

# Logic to integrate with TradersPost for efficient entry/exit
# This aligns with the requirement for "best combinations, when to get in and get out"
def send_trade_signal(symbol, side, price):
    url = os.environ.get("TRADERSPOST_WEBHOOK_URL")
    payload = {
        "symbol": symbol,
        "side": side,
        "price": price,
        "strategy": "Titan-Causal-Intraday"
    }
    try:
        response = requests.post(url, json=payload)
        return response.status_code
    except Exception as e:
        logging.error(f"Webhook failed: {e}")
        return None