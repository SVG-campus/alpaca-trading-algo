import os
import os
import glob
import json
import time
import logging
import csv
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Configure logging
os.makedirs('data', exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/alpaca_trade.log"),
        logging.StreamHandler()
    ]
)

# Parse .env file manually so we don't need python-dotenv
env_vars = {}
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.strip().split('=', 1)
                env_vars[key.strip()] = val.strip().strip('"').strip("'")

# We now load BOTH Paper and Live keys for simultaneous execution.
PAPER_API_KEY = env_vars.get('APCA_INTRA_PAPER_API_KEY_ID') or os.environ.get('APCA_INTRA_PAPER_API_KEY_ID')
PAPER_API_SECRET = env_vars.get('APCA_INTRA_PAPER_API_SECRET_KEY') or os.environ.get('APCA_INTRA_PAPER_API_SECRET_KEY')

LIVE_API_KEY = env_vars.get('APCA_LIVE_API_KEY_ID') or os.environ.get('APCA_LIVE_API_KEY_ID')
LIVE_API_SECRET = env_vars.get('APCA_LIVE_API_SECRET_KEY') or os.environ.get('APCA_LIVE_API_SECRET_KEY')

CACHE_FILE = 'data/intraday/data/latest_discovery.json'

def get_trading_clients():
    clients = {}
    if PAPER_API_KEY and PAPER_API_SECRET:
        clients['PAPER'] = TradingClient(PAPER_API_KEY, PAPER_API_SECRET, paper=True)
    else:
        logging.warning("APCA_INTRA_PAPER_API_KEY_ID and SECRET not found in .env. Skipping Paper execution.")
        
    if LIVE_API_KEY and LIVE_API_SECRET:
        clients['LIVE'] = TradingClient(LIVE_API_KEY, LIVE_API_SECRET, paper=False)
    else:
        logging.warning("APCA_LIVE_API_KEY_ID and SECRET not found in .env. Skipping Live execution.")
        
    if not clients:
        logging.error("No valid API keys found in .env. Cannot trade.")
        
    return clients

def wait_for_market_open(client: TradingClient):
    """Wait until the market opens, and then wait 5 minutes to avoid volatility noise."""
    while True:
        clock = client.get_clock()
        if clock.is_open:
            # Check how long it's been open to see if we need to wait for noise to clear
            # The market opens at 9:30 AM EST. We want to wait ~5 minutes after open.
            time_since_open = (clock.timestamp - clock.next_open + timedelta(days=1)).total_seconds() % 86400
            # A more robust way: just sleep 5 minutes the moment it opens if we just woke up
            logging.info("Market is open! Waiting 5 minutes for opening volatility noise to clear before trading...")
            time.sleep(300)
            break
        
        wait_time = (clock.next_open - clock.timestamp).total_seconds()
        logging.info(f"Market closed. Waiting {wait_time / 60:.2f} minutes for next open at {clock.next_open}.")
        
        if wait_time > 3600:
            time.sleep(3600)  # Sleep in 1 hour chunks if far away
        else:
            time.sleep(max(wait_time, 1))

def time_to_market_close(client: TradingClient) -> float:
    """Returns seconds until market close. If closed, returns 0."""
    clock = client.get_clock()
    if not clock.is_open:
        return 0.0
    return (clock.next_close - clock.timestamp).total_seconds()

def log_trade_to_ledger(account_name, symbol, action, reason, pnl_pct=0.0):
    ledger_file = "data/intraday/data/intraday_trade_ledger.csv"
    file_exists = os.path.isfile(ledger_file)
    with open(ledger_file, 'a') as f:
        if not file_exists:
            f.write("Timestamp_UTC,Account,Symbol,Action,Reason,Unrealized_PNL_Pct\n")
        ts = datetime.now(timezone.utc).isoformat()
        f.write(f"{ts},{account_name},{symbol},{action},{reason},{pnl_pct:.4f}\n")

def close_all_positions(clients: dict, reason="End of Day"):
    """Closes all open positions before market close across all active accounts."""
    logging.info(f"{reason} reached. Closing all open positions across {list(clients.keys())}...")
    for account_name, client in clients.items():
        try:
            positions = client.get_all_positions()
            for pos in positions:
                log_trade_to_ledger(account_name, pos.symbol, "SELL", reason, float(pos.unrealized_plpc))
                
            client.cancel_orders()
            client.close_all_positions(cancel_orders=True)
            logging.info(f"[{account_name}] Successfully liquidated all positions.")
        except Exception as e:
            logging.error(f"[{account_name}] Failed to close positions: {e}")

def execute_intraday_trade():
    clients = get_trading_clients()
    if not clients:
        time.sleep(60)
        return

    # We use the first available client to check the clock
    base_client = list(clients.values())[0]

    # 1. Wait for market to be open
    wait_for_market_open(base_client)

    # 2. Check time until close. If less than 15 mins, wait for next day.
    secs_to_close = time_to_market_close(base_client)
    if secs_to_close < 900:
        logging.info("Less than 15 minutes to market close. Not taking new trades today.")
        time.sleep(secs_to_close + 60) # Sleep until market closes
        return

    # 3. Read discovery cache
    if not os.path.exists(CACHE_FILE):
        logging.error(f"Discovery cache {CACHE_FILE} not found. Waiting...")
        time.sleep(300)
        return

    try:
        with open(CACHE_FILE, 'r') as f:
            discovery = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON: {e}")
        time.sleep(300)
        return

    # V9 INTRADAY TRADING LOGIC
    # We should trade ONCE per trading day.
    # Check if we have already traded today based on Eastern Time!
    # Because UTC could roll over mid-afternoon causing double trades.
    # V10 FIX: Trade precisely once per trading day based on Alpaca's Clock
    clock = base_client.get_clock()
    today_str = clock.timestamp.strftime('%Y-%m-%d')
    state_file = "data/intraday/data/last_traded_date.txt"
    last_traded_date = ""
    
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            last_traded_date = f.read().strip()
            
    if last_traded_date == today_str:
        logging.info("Already executed trades for today. Sleeping until tomorrow...")
        # Check if the market is open. If it is, sleep until market close to prevent log spam.
        secs_to_close = time_to_market_close(base_client)
        if secs_to_close > 0:
            time.sleep(secs_to_close + 60)
        else:
            time.sleep(3600)
        return

    # 4. We are cleared to trade today!
    long_pick = discovery.get("long_pick")
    
    if not long_pick:
        logging.warning("No long_pick found in discovery. Skipping.")
        with open(state_file, 'w') as f: f.write(today_str)
        return

    # 5. Liquidate pre-existing positions across all active clients
    logging.info("Liquidating pre-existing positions before today's execution...")
    for account_name, client in clients.items():
        try:
            client.cancel_orders()
            client.close_all_positions(cancel_orders=True)
            logging.info(f"[{account_name}] Liquidated successfully.")
        except Exception as e:
            logging.error(f"[{account_name}] Failed to liquidate: {e}")
            
    time.sleep(15) # Wait for settlement

    # Read Slippage Limits once for the asset
    slippage_cap = float('inf')
    try:
        slippage_files = glob.glob('data/intraday/data/slippage_1pct_adv_*.csv')
        if slippage_files:
            latest_slippage = max(slippage_files, key=os.path.getmtime)
            with open(latest_slippage, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['symbol'] == long_pick:
                        if row['cash_value_1pct_adv']:
                            slippage_cap = float(row['cash_value_1pct_adv'])
                        break
            logging.info(f"Slippage Cap (1% ADV) for {long_pick}: ${slippage_cap:,.2f}")
        else:
            logging.warning("No slippage file found. Cannot verify liquidity ceiling. Proceeding with caution.")
    except Exception as e:
        logging.error(f"Error reading slippage limit: {e}")

    # 6. Execute Simultaneous Buys
    for account_name, client in clients.items():
        try:
            account = client.get_account()
            cash = float(account.cash)
            logging.info(f"[{account_name}] Cash Available: ${cash:.2f}")
            
            # Keep 5% safety buffer
            trade_amount = cash * 0.95
            
            # Enforce Slippage Cap
            if trade_amount > slippage_cap:
                logging.warning(f"[{account_name}] Trade amount (${trade_amount:.2f}) EXCEEDS Slippage Cap (${slippage_cap:.2f}). Limiting trade!")
                trade_amount = slippage_cap

            if trade_amount < 1.0:
                logging.warning(f"[{account_name}] Not enough cash. Using all available cash minus 5c.")
                trade_amount = cash - 0.05
                if trade_amount <= 0:
                    logging.error(f"[{account_name}] Account empty. Skipping trade.")
                    continue
            
            logging.info(f"[{account_name}] Targeting LONG on {long_pick} with ${trade_amount:.2f} (Slippage Safe)")
            
            req = MarketOrderRequest(
                symbol=long_pick,
                notional=round(trade_amount, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            order = client.submit_order(order_data=req)
            logging.info(f"[{account_name}] Submitted BUY order for {long_pick}: ID {order.id}")
            log_trade_to_ledger(account_name, long_pick, "BUY", "V10 Discovery Entry", 0.0)
            
        except Exception as e:
            logging.error(f"[{account_name}] Failed to place order: {e}")
            
    # Mark as traded FOR TODAY so we don't double-buy
    with open(state_file, 'w') as f: 
        f.write(today_str)

    # V10 INTRADAY OPTIMIZED DYNAMIC BRACKETS:
    # Based on exhaustive 1-minute tick simulations, static targets get chopped out.
    # We now implement a robust mathematical trailing logic:
    # 1. Base Target: +14%, Base Stop: -9%
    # 2. Dynamic Trailing: If the stock rallies significantly (+5%), we raise the stop-loss to breakeven!
    target_profit_pct = 0.14
    initial_stop_loss_pct = -0.09
    current_stop_loss_pct = initial_stop_loss_pct
    
    # Wait for the order to fill
    time.sleep(15)
    
    # Track highest PNL per account for the trailing stop loss logic
    highest_pnl_tracker = {name: 0.0 for name in clients.keys()}
    current_stop_loss_tracker = {name: initial_stop_loss_pct for name in clients.keys()}
    
    # Monitor loop
    while True:
        secs_to_close = time_to_market_close(base_client)
        if secs_to_close < 300: # 5 minutes
            logging.info("Approaching market close. Liquidating all active intraday positions.")
            close_all_positions(clients, reason="TIME_STOP_5MIN")
            time.sleep(secs_to_close + 60)
            break
            
        active_accounts = 0
        for account_name, client in list(clients.items()):
            try:
                positions = client.get_all_positions()
                position_found = False
                
                for pos in positions:
                    if pos.symbol == long_pick:
                        position_found = True
                        active_accounts += 1
                        unrealized_pct = float(pos.unrealized_plpc)
                        
                        if unrealized_pct > highest_pnl_tracker[account_name]:
                            highest_pnl_tracker[account_name] = unrealized_pct
                            
                        # DYNAMIC TRAILING LOGIC
                        if highest_pnl_tracker[account_name] >= 0.05 and current_stop_loss_tracker[account_name] < 0.00:
                            logging.info(f"[{account_name}] Rally detected! Moving Stop Loss to Breakeven (0.00%).")
                            current_stop_loss_tracker[account_name] = 0.00
                            
                        if highest_pnl_tracker[account_name] >= 0.10 and current_stop_loss_tracker[account_name] < 0.05:
                            logging.info(f"[{account_name}] Massive Rally detected! Trailing Stop Loss to +5.00%.")
                            current_stop_loss_tracker[account_name] = 0.05

                        logging.info(f"[{account_name}] Monitoring {long_pick} | P/L: {unrealized_pct*100:.2f}% | Max: {highest_pnl_tracker[account_name]*100:.2f}% | SL: {current_stop_loss_tracker[account_name]*100:.2f}%")
                        
                        if unrealized_pct >= target_profit_pct:
                            logging.info(f"[{account_name}] TAKE PROFIT hit! (+{unrealized_pct*100:.2f}%). Liquidating {long_pick}.")
                            client.close_position(long_pick)
                            log_trade_to_ledger(account_name, long_pick, "SELL", "TAKE_PROFIT", unrealized_pct)
                        elif unrealized_pct <= current_stop_loss_tracker[account_name]:
                            logging.info(f"[{account_name}] STOP LOSS hit! ({unrealized_pct*100:.2f}%). Liquidating {long_pick}.")
                            client.close_position(long_pick)
                            log_trade_to_ledger(account_name, long_pick, "SELL", "DYNAMIC_STOP_LOSS", unrealized_pct)
                            
                if not position_found:
                    # If it's not found, the position is closed. We remove the client from active monitoring.
                    pass
                    
            except Exception as e:
                logging.error(f"[{account_name}] Error checking positions: {e}")
                active_accounts += 1 # Keep alive in case of temporary API hiccup
                
        if active_accounts == 0:
            logging.info("All positions across all accounts are closed. Exiting monitor loop.")
            break
            
        time.sleep(60) # Poll every 60 seconds

if __name__ == '__main__':
    while True:
        try:
            execute_intraday_trade()
        except Exception as e:
            logging.error(f"Unexpected error in trading loop: {e}")
            time.sleep(60)
