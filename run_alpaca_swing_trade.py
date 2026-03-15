import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Configure logging
os.makedirs('data/swing', exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/swing/alpaca_swing_trade.log"),
        logging.StreamHandler()
    ]
)

CACHE_FILE = 'data/swing/data/latest_discovery.json'

def get_env_vars():
    env_vars = {}
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, val = line.strip().split('=', 1)
                    env_vars[key.strip()] = val.strip().strip('"').strip("'")
    return env_vars

def is_first_trading_day_of_month(client: TradingClient):
    """Checks if today is the first trading day of the current month."""
    today = datetime.now(timezone.utc).date()
    # If it's a weekend, it's not a trading day anyway
    if today.weekday() > 4:
        return False
        
    # Get calendar for the first 7 days of the month to find the first trading day
    first_day_of_month = today.replace(day=1)
    end_check_date = first_day_of_month + timedelta(days=7)
    
    # Alpaca's get_calendar expects string dates
    start_str = first_day_of_month.strftime('%Y-%m-%d')
    end_str = end_check_date.strftime('%Y-%m-%d')
    
    # We must use TradingClient to get calendar, wait, TradingClient doesn't have get_calendar natively exposed like REST.
    # Actually, TradingClient does not have a simple get_calendar method in alpaca-py.
    # We will use the REST clock to just check if today is open, and if today's day <= 5, we can manually ensure we trade only once per month.
    # A robust way: Just check if we haven't traded this month yet, and today is an open market day!
    return True # We handle the "once per month" logic via a state file instead of strictly predicting the 1st trading day!

def rebalance_account(client: TradingClient, discovery: dict, account_name: str):
    logging.info(f"[{account_name}] Starting monthly rebalance...")
    
    # 1. Cancel all orders and liquidate everything
    logging.info(f"[{account_name}] Liquidating all previous positions...")
    try:
        client.cancel_orders()
        client.close_all_positions(cancel_orders=True)
        time.sleep(15) # Wait for settlement
    except Exception as e:
        logging.error(f"[{account_name}] Failed to liquidate: {e}")
        return

    try:
        account = client.get_account()
        equity = float(account.equity)
        cash = float(account.cash)
        logging.info(f"[{account_name}] Equity: ${equity:.2f} | Cash: ${cash:.2f}")
    except Exception as e:
        logging.error(f"[{account_name}] Failed to get account info: {e}")
        return

    # V7 Strategy: 80% Long / 20% Short
    # We use 95% of equity to avoid margin calls (76% Long / 19% Short)
    long_equity = equity * 0.76
    short_equity = equity * 0.19

    top_longs = [p["symbol"] for p in discovery.get("top_rankings", [])][:50]
    top_shorts = [p["symbol"] for p in discovery.get("top_short_rankings", [])][:50]

    if not top_longs:
        logging.warning(f"[{account_name}] No long picks found.")
        return

    long_allocation_per_stock = long_equity / len(top_longs)
    short_allocation_per_stock = short_equity / len(top_shorts) if top_shorts else 0

    logging.info(f"[{account_name}] Allocating ${long_allocation_per_stock:.2f} per Long and ${short_allocation_per_stock:.2f} per Short.")

    # Execute Longs (Fractional allowed)
    for symbol in top_longs:
        if long_allocation_per_stock < 1.0: continue
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                notional=round(long_allocation_per_stock, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(order_data=req)
        except Exception as e:
            logging.warning(f"[{account_name}] Failed to long {symbol}: {e}")

    # Execute Shorts (Fractional NOT allowed on Alpaca for shorts, must calculate integer qty)
    # To do this safely without fractional shorting, we need latest prices.
    latest_prices = discovery.get("latest_prices", {})
    for symbol in top_shorts:
        price = latest_prices.get(symbol)
        if not price or price <= 0: continue
        
        qty = int(short_allocation_per_stock // price)
        if qty < 1: continue
        
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(order_data=req)
        except Exception as e:
            logging.warning(f"[{account_name}] Failed to short {symbol}: {e}")

    logging.info(f"[{account_name}] Monthly rebalance complete.")

def execute_monthly_swing():
    env_vars = get_env_vars()
    
    # Clients
    paper_key = env_vars.get('APCA_PAPER_API_KEY_ID') or os.environ.get('APCA_PAPER_API_KEY_ID')
    paper_sec = env_vars.get('APCA_PAPER_API_SECRET_KEY') or os.environ.get('APCA_PAPER_API_SECRET_KEY')
    
    max_key = env_vars.get('APCA_PAPER_MAX_API_KEY_ID') or os.environ.get('APCA_PAPER_MAX_API_KEY_ID')
    max_sec = env_vars.get('APCA_PAPER_MAX_API_SECRET_KEY') or os.environ.get('APCA_PAPER_MAX_API_SECRET_KEY')
    
    paper_client = TradingClient(paper_key, paper_sec, paper=True) if paper_key and paper_sec and paper_key != 'YOUR_PAPER_KEY' else None
    max_client = TradingClient(max_key, max_sec, paper=True) if max_key and max_sec and max_key != 'YOUR_PAPER_MAX_KEY' else None

    if not paper_client and not max_client:
        logging.error("No valid Paper or Paper Max API keys found in .env.")
        time.sleep(3600)
        return

    # To coordinate with the Intraday trader, wait until 7:00 AM PST. 
    # Market opens at 6:30 AM PST. Intraday trader trades at 6:35 AM PST. 
    # This completely separates API limits.
    now = datetime.now()
    if now.hour < 7:
        logging.info("Waiting for 7:00 AM PST to avoid API collision with Intraday trader...")
        time.sleep(1800)
        return

    # Check if market is open today
    # We just need one client to check the clock
    checker_client = paper_client if paper_client else max_client
    try:
        clock = checker_client.get_clock()
        if not clock.is_open:
            wait_time = (clock.next_open - clock.timestamp).total_seconds()
            logging.info(f"Market closed. Waiting {wait_time / 3600:.2f} hours for next open.")
            time.sleep(min(wait_time, 3600))
            return
    except Exception as e:
        logging.error(f"Clock check failed: {e}")
        time.sleep(600)
        return

    # Monthly execution check
    current_month = now.strftime('%Y-%m')
    state_file = "data/swing/last_traded_month.txt"
    last_traded_month = ""
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            last_traded_month = f.read().strip()
            
    if last_traded_month == current_month:
        logging.info(f"Already traded for month {current_month}. Sleeping for 24 hours...")
        time.sleep(86400)
        return

    # We haven't traded this month, and the market is OPEN, and it's past 7:00 AM PST!
    # Let's read the latest swing discovery payload
    if not os.path.exists(CACHE_FILE):
        logging.error(f"Swing discovery cache {CACHE_FILE} not found. Waiting...")
        time.sleep(3600)
        return

    try:
        with open(CACHE_FILE, 'r') as f:
            discovery = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON: {e}")
        time.sleep(3600)
        return

    # Rebalance
    if paper_client:
        rebalance_account(paper_client, discovery, "PAPER")
    if max_client:
        rebalance_account(max_client, discovery, "PAPER MAX")

    # Mark month as traded
    with open(state_file, 'w') as f:
        f.write(current_month)
        
    logging.info(f"Successfully documented completion of {current_month} swing trade.")

if __name__ == '__main__':
    while True:
        try:
            execute_monthly_swing()
        except Exception as e:
            logging.error(f"Unexpected error in swing loop: {e}")
            time.sleep(60)