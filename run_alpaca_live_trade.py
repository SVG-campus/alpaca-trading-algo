import os
import json
import time
import logging
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

# We use APCA_LIVE_API_KEY_ID since this is real money!
API_KEY = os.environ.get('APCA_LIVE_API_KEY_ID')
API_SECRET = os.environ.get('APCA_LIVE_API_SECRET_KEY')
CACHE_FILE = 'data/intraday/latest_discovery.json'

def get_trading_client():
    if not API_KEY or not API_SECRET:
        logging.error("APCA_LIVE_API_KEY_ID and APCA_LIVE_API_SECRET_KEY must be set in .env")
        return None
    return TradingClient(API_KEY, API_SECRET, paper=False)

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

def close_all_positions(client: TradingClient):
    """Closes all open positions before market close."""
    logging.info("End of day reached. Closing all open positions for intraday strategy.")
    try:
        # Cancel all open orders first
        client.cancel_orders()
        # Liquidate all positions
        client.close_all_positions(cancel_orders=True)
        logging.info("Successfully requested closure of all positions.")
    except Exception as e:
        logging.error(f"Failed to close positions: {e}")

def execute_intraday_trade():
    client = get_trading_client()
    if not client:
        return

    # 1. Wait for market to be open
    wait_for_market_open(client)

    # 2. Check time until close. If less than 15 mins, wait for next day.
    secs_to_close = time_to_market_close(client)
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

    # Check if we already traded this JSON payload today
    # We can track the generation time of the JSON
    gen_time_str = discovery.get("generated_at_utc")
    if not gen_time_str:
        logging.warning("No generated_at_utc found in JSON, using modified time.")
        last_mtime = os.path.getmtime(CACHE_FILE)
    else:
        try:
            gen_time = datetime.fromisoformat(gen_time_str.replace('Z', '+00:00'))
            last_mtime = gen_time.timestamp()
        except:
            last_mtime = os.path.getmtime(CACHE_FILE)

    state_file = "data/last_traded_state.txt"
    last_traded_mtime = 0.0
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            try:
                last_traded_mtime = float(f.read().strip())
            except:
                pass

    if last_mtime <= last_traded_mtime:
        logging.info("Already executed trades for this discovery generation. Waiting for next update...")
        time.sleep(300)
        return

    # 4. We have a new payload, execute the trade!
    long_pick = discovery.get("long_pick")
    
    if not long_pick:
        logging.warning("No long_pick found in discovery. Skipping.")
        with open(state_file, 'w') as f: f.write(str(last_mtime))
        return

    # 5. Liquidate any existing positions to ensure we have maximum cash ready!
    logging.info("Checking for any pre-existing positions to liquidate before today's trade...")
    try:
        client.cancel_orders()
        client.close_all_positions(cancel_orders=True)
        time.sleep(10) # Wait for settlement
    except Exception as e:
        logging.error(f"Failed to liquidate pre-existing positions: {e}")

    # Check account balance
    try:
        account = client.get_account()
        cash = float(account.cash)
        logging.info(f"Account Cash Available after liquidation: ${cash:.2f}")
    except Exception as e:
        logging.error(f"Failed to fetch account info: {e}")
        time.sleep(60)
        return

    # Safety buffer: keep 5% or minimum $0.05 cash to prevent margin calls
    trade_amount = cash * 0.95
    if trade_amount < 1.0:
        logging.warning(f"Not enough cash to trade securely. Have ${cash:.2f}, need at least $1.05. Using all available cash for testing: ${cash:.2f}")
        trade_amount = cash - 0.05 # Leave 5 cents just in case
        if trade_amount <= 0:
            logging.error("Account basically empty. Can't trade.")
            with open(state_file, 'w') as f: f.write(str(last_mtime))
            return

    logging.info(f"Targeting LONG on {long_pick} with ${trade_amount:.2f}")

    # Alpaca allows fractional shares. We can use `notional` to specify the dollar amount.
    # To maximize profit and minimize risk, we will add a bracket order.
    # We don't know the exact current price here, but we can set Take Profit / Stop loss via percentage offsets 
    # Or rely on Alpaca's simple market order + an end-of-day exit. 
    # Alpaca's bracket orders require limit_price or stop_price, which means we need the current price to set them accurately.
    # For a purely automated intraday system, buying at market and selling at end of day is the simplest robust start.
    
    try:
        # Submit the buy order
        req = MarketOrderRequest(
            symbol=long_pick,
            notional=round(trade_amount, 2), # Fractional amount in dollars
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        order = client.submit_order(order_data=req)
        logging.info(f"Submitted BUY order for {long_pick}: ID {order.id}")
        
        # Mark as traded
        with open(state_file, 'w') as f: 
            f.write(str(last_mtime))
            
    except Exception as e:
        logging.error(f"Failed to place order: {e}")
        time.sleep(60)
        return

    # V8 INTRADAY OPTIMIZED BRACKET TARGETS:
    # Because our new V8 Kaggle Engine targets highly volatile short-term momentum stocks,
    # we expand the profit target to allow runners to run, while enforcing a tight 2:1 Risk/Reward!
    # Let's set a target of +4.5% profit and -2.0% stop loss for intraday.
    target_profit_pct = 0.045
    stop_loss_pct = -0.02
    
    # Wait for the order to fill
    time.sleep(10)
    
    # Monitor loop
    while True:
        secs_to_close = time_to_market_close(client)
        if secs_to_close < 300: # 5 minutes
            logging.info("Approaching market close. Liquidating intraday position.")
            close_all_positions(client)
            time.sleep(secs_to_close + 60)
            break
            
        try:
            positions = client.get_all_positions()
            if not positions:
                logging.info("No open positions found. Exiting monitor loop.")
                break
                
            for pos in positions:
                if pos.symbol == long_pick:
                    unrealized_pct = float(pos.unrealized_plpc)
                    logging.info(f"Monitoring {long_pick} | Unrealized P/L: {unrealized_pct*100:.2f}% | {secs_to_close / 60:.1f} mins to close")
                    
                    if unrealized_pct >= target_profit_pct:
                        logging.info(f"TAKE PROFIT hit! (+{unrealized_pct*100:.2f}%). Liquidating {long_pick}.")
                        client.close_position(long_pick)
                        break
                    elif unrealized_pct <= stop_loss_pct:
                        logging.info(f"STOP LOSS hit! ({unrealized_pct*100:.2f}%). Liquidating {long_pick}.")
                        client.close_position(long_pick)
                        break
        except Exception as e:
            logging.error(f"Error checking positions: {e}")
            
        time.sleep(60) # Poll every 60 seconds

if __name__ == '__main__':
    while True:
        try:
            execute_intraday_trade()
        except Exception as e:
            logging.error(f"Unexpected error in trading loop: {e}")
            time.sleep(60)
