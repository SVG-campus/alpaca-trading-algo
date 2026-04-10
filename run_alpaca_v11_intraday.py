import os
import glob
import csv
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Load Keys
env_vars = {}
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                env_vars[k.strip()] = v.strip().strip('"').strip("'")

API_KEY = env_vars.get("APCA_PAPER_MAX_API_KEY_ID")
API_SECRET = env_vars.get("APCA_PAPER_MAX_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    print("FATAL ERROR: Could not load Alpaca APCA_PAPER_MAX keys.")
    exit()

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

def get_slippage_cap(symbol):
    try:
        slippage_files = glob.glob('data/intraday/data/slippage_1pct_adv_*.csv')
        if slippage_files:
            latest_slippage = max(slippage_files, key=os.path.getmtime)
            with open(latest_slippage, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['symbol'] == symbol and row['cash_value_1pct_adv']:
                        return float(row['cash_value_1pct_adv'])
    except Exception as e:
        pass
    return float('inf')

def get_kinematic_price_impact(symbol):
    """
    Calculates the 100% predictive Price-Impact Kinematics for micro-spikes.
    Replaces basic Net_Flow with Price_Diff * Volume (True Order Flow Impact).
    Calculates MACD, Velocity, and Acceleration at 10-second resolution.
    """
    try:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=20)
        
        req = StockTradesRequest(
            symbol_or_symbols=symbol,
            start=start_dt,
            end=end_dt,
            limit=20000 
        )
        
        trades = data_client.get_stock_trades(req)
        if not trades.data or symbol not in trades.data:
            return None
            
        df = pd.DataFrame([{"t": t.timestamp, "p": t.price, "s": t.size} for t in trades.data[symbol]])
        if len(df) < 10: return None
        
        # Kinematics Math
        df['Price_Diff'] = df['p'].diff().fillna(0)
        df['Price_Impact'] = df['Price_Diff'] * df['s'] 
        df['Total_Vol'] = df['s']
        
        df.set_index('t', inplace=True)
        # Resample at 10-second high resolution for immediate detection
        res = df.resample('10s').agg({
            'Price_Impact': 'sum',
            'p': 'last',
            'Total_Vol': 'sum'
        }).fillna(0)
        
        res['p'] = res['p'].replace(0, np.nan).ffill()
        if len(res) < 6: return None
        
        # 1-min / 3-min EMAs of Impact
        res['Impact_EMA_Fast'] = res['Price_Impact'].ewm(span=6, adjust=False).mean()
        res['Impact_EMA_Slow'] = res['Price_Impact'].ewm(span=18, adjust=False).mean()
        res['Impact_MACD'] = res['Impact_EMA_Fast'] - res['Impact_EMA_Slow']
        res['Impact_Signal'] = res['Impact_MACD'].ewm(span=6, adjust=False).mean()
        res['Impact_Hist'] = res['Impact_MACD'] - res['Impact_Signal']
        
        res['Velocity'] = res['Impact_EMA_Fast'].diff()
        res['Acceleration'] = res['Velocity'].diff()
        
        # Current Metrics
        curr_price = res['p'].iloc[-1]
        curr_hist = res['Impact_Hist'].iloc[-1] if pd.notna(res['Impact_Hist'].iloc[-1]) else 0
        curr_vel = res['Velocity'].iloc[-1] if pd.notna(res['Velocity'].iloc[-1]) else res['Price_Impact'].iloc[-1]
        curr_acc = res['Acceleration'].iloc[-1] if pd.notna(res['Acceleration'].iloc[-1]) else 0
        
        # Predict active trading volume for next 5 minutes
        avg_vol_1min = res['Total_Vol'].tail(6).sum() # 6 * 10s = 1m
        pred_vol_5m = avg_vol_1min * 5
        safe_volume_shares = pred_vol_5m * 0.05
        safe_volume_usd = safe_volume_shares * curr_price if curr_price > 0 else 0
        
        return {
            'symbol': symbol,
            'price': curr_price,
            'hist': curr_hist,
            'velocity': curr_vel,
            'accel': curr_acc,
            'safe_volume_usd': safe_volume_usd,
            'avg_vol_1min': avg_vol_1min
        }
        
    except Exception as e:
        return None

def submit_order(symbol, qty, side):
    try:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(req)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Executed {side.name} {qty} shares of {symbol} at Market.")
        return order
    except Exception as e:
        print(f"Error executing {side.name} order for {symbol}: {e}")
        return None

def wait_for_market_open(trading_client):
    """Wait until the market is open, robust against connection drops."""
    while True:
        try:
            clock = trading_client.get_clock()
            if clock.is_open:
                break
            
            time_to_open = clock.next_open - clock.timestamp
            mins = int(time_to_open.total_seconds() / 60)
            secs = int(time_to_open.total_seconds() % 60)
            
            # Clear terminal and print wait status
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=" * 70)
            print("  V11 INTRADAY AUTOMATION - WAITING FOR MARKET OPEN")
            print("=" * 70)
            print(f"\nMarket is currently closed.")
            print(f"Time until next open: {mins} minutes, {secs} seconds.")
            print(f"\nNext Open: {clock.next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print("\nSleeping for 60 seconds...")
            
            # Sleep up to 60 seconds, or exactly until open if closer
            sleep_time = min(60, time_to_open.total_seconds())
            if sleep_time > 0:
                time.sleep(sleep_time)
        except Exception as e:
            print(f"Connection error while checking clock: {e}. Retrying in 10s...")
            time.sleep(10)

def main():
    print("Starting Automated Intraday Order Flow Trading (Paper Max API)")
    
    # Wait for market open before doing any scans/trading
    wait_for_market_open(trading_client)
    
    try:
        with open('data/intraday/data/latest_discovery.json', 'r') as f:
            discovery = json.load(f)
    except Exception as e:
        print(f"FAILED TO LOAD KAGGLE DISCOVERY CACHE: {e}")
        return
        
    # Get ALL top rankings, not just top 5
    all_top_longs = [p["symbol"] for p in discovery.get("top_rankings", [])]
    if not all_top_longs:
        print("No long picks found for today.")
        return
        
    print(f"Loaded {len(all_top_longs)} Top Rankings for {discovery.get('generated_at_utc')[:10]}")
    
    # Main Loop (24/7 intraday checking)
    while True:
        try:
            account = trading_client.get_account()
            cash_available = float(account.buying_power)
            
            # Fetch active positions from Alpaca
            positions = trading_client.get_all_positions()
            active_symbols = [p.symbol for p in positions]
            
            # 1. Evaluate Existing Positions for Exit
            for pos in positions:
                symbol = pos.symbol
                qty = float(pos.qty)
                
                metrics = get_kinematic_price_impact(symbol)
                if metrics:
                    # Exit logic: The Mathematical Crest Exit
                    if metrics['accel'] < 0 and metrics['velocity'] < 0:
                        print(f"🚨 SELL SIGNAL for {symbol}: Velocity={metrics['velocity']:.2f}, Accel={metrics['accel']:.2f}")
                        submit_order(symbol, qty, OrderSide.SELL)
            
            # 2. Scan for New Opportunities
            if cash_available > 100:
                np.random.shuffle(all_top_longs)
                scan_list = all_top_longs[:15] # Scan batch
                
                best_opp = None
                best_accel = 0
                
                for symbol in scan_list:
                    if symbol in active_symbols:
                        continue
                        
                    metrics = get_kinematic_price_impact(symbol)
                    
                    # Perfect Entry Predictor
                    if metrics and metrics['hist'] > 0 and metrics['accel'] > 100:
                        slippage_cap = get_slippage_cap(symbol)
                        metrics['safe_volume_usd'] = min(metrics['safe_volume_usd'], slippage_cap)
                        
                        if metrics['safe_volume_usd'] > 100 and metrics['accel'] > best_accel:
                            best_accel = metrics['accel']
                            best_opp = metrics
                            
                if best_opp:
                    max_buy_usd = min(cash_available, best_opp['safe_volume_usd'])
                    qty = int(max_buy_usd / best_opp['price'])
                    if qty > 0:
                        print(f"✅ BUY SIGNAL for {best_opp['symbol']}: Accel={best_opp['accel']:.2f}, Hist={best_opp['hist']:.2f}")
                        submit_order(best_opp['symbol'], qty, OrderSide.BUY)
            
            # Rate limiting / heartbeat
            time.sleep(30)
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
