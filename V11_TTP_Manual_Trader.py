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

# Load Keys
env_vars = {}
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                env_vars[k.strip()] = v.strip().strip('"').strip("'")

API_KEY = env_vars.get("APCA_LIVE_API_KEY_ID")
API_SECRET = env_vars.get("APCA_LIVE_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    print("FATAL ERROR: Could not load Alpaca APCA_LIVE keys.")
    exit()

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

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
        print(f"Error reading slippage limit for {symbol}: {e}")
    return float('inf')

def get_ofi_velocity(symbol):
    """Calculates the current Order Flow Imbalance Velocity (Buying Pressure Acceleration)"""
    try:
        # Get the last 15 minutes of trades
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=15)
        
        req = StockTradesRequest(
            symbol_or_symbols=symbol,
            start=start_dt,
            end=end_dt,
            limit=10000 # Enough to cover 15 mins of heavy trading
        )
        
        trades = data_client.get_stock_trades(req)
        if not trades.data or symbol not in trades.data:
            return 0.0, 0.0
            
        df = pd.DataFrame([{"t": t.timestamp, "p": t.price, "s": t.size} for t in trades.data[symbol]])
        if len(df) < 10: return 0.0, 0.0
        
        df['Price_Diff'] = df['p'].diff()
        df['Tick_Direction'] = np.sign(df['Price_Diff'])
        df['Tick_Direction'] = df['Tick_Direction'].replace(0, np.nan).ffill().fillna(0)
        
        df['Buy_Vol'] = np.where(df['Tick_Direction'] > 0, df['s'], 0)
        df['Sell_Vol'] = np.where(df['Tick_Direction'] < 0, df['s'], 0)
        
        df.set_index('t', inplace=True)
        min_df = df.resample('1min').agg({'Buy_Vol': 'sum', 'Sell_Vol': 'sum'}).fillna(0)
        
        if len(min_df) < 5: return 0.0, 0.0
        
        min_df['Net_Flow'] = min_df['Buy_Vol'] - min_df['Sell_Vol']
        
        # 5-minute rolling Net Flow (Momentum)
        min_df['OFI_5m'] = min_df['Net_Flow'].rolling(5).sum()
        
        # Velocity (Acceleration of Flow)
        min_df['OFI_Velocity'] = min_df['OFI_5m'].diff()
        
        current_momentum = min_df['OFI_5m'].iloc[-1]
        current_velocity = min_df['OFI_Velocity'].iloc[-1]
        
        return current_momentum, current_velocity
        
    except Exception as e:
        # print(f"API Error fetching OFI for {symbol}: {e}")
        return 0.0, 0.0

def start_interactive_app():
    print("==================================================")
    print("  V11 TTP MANUAL TRADER - ORDER FLOW INTERFACE    ")
    print("==================================================")
    
    # 1. Load Discovery
    try:
        with open('data/intraday/data/latest_discovery.json', 'r') as f:
            discovery = json.load(f)
    except Exception as e:
        print(f"FAILED TO LOAD KAGGLE DISCOVERY CACHE: {e}")
        return
        
    top_longs = [p["symbol"] for p in discovery.get("top_rankings", [])]
    if not top_longs:
        print("No long picks found for today.")
        return
        
    print(f"Loaded {len(top_longs)} V9 Top Rankings for {discovery.get('generated_at_utc')[:10]}")
    
    # 2. Get Starting Cash
    while True:
        try:
            cash_input = input("\n[1] Enter your total available TTP Cash Balance (e.g. 6000): $")
            available_cash = float(cash_input.replace(',', ''))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    # Keep track of active positions
    current_position = None
    
    while True:
        if current_position is None:
            # We are looking to buy. We need the stock with the HIGHEST OFI Velocity.
            print("\n[2] Scanning Top 5 candidates for maximum Buying Velocity (OFI)...")
            best_symbol = None
            best_velocity = -999
            
            for symbol in top_longs[:5]: # Scan the top 5 for the absolute best momentum right now
                mom, vel = get_ofi_velocity(symbol)
                print(f"  > {symbol:<5} | 5m Flow: {mom:8.0f} shares | Velocity: {vel:8.0f}")
                if vel > best_velocity and mom > 0: # Must have positive momentum and highest velocity
                    best_velocity = vel
                    best_symbol = symbol
                    
            if not best_symbol:
                print("\n[!] No stocks are currently showing positive accelerating buying pressure.")
                print("Waiting 60 seconds before scanning again...")
                time.sleep(60)
                continue
                
            # 3. Calculate Slippage-Safe Trade Size
            slippage_cap = get_slippage_cap(best_symbol)
            trade_amount = min(available_cash, slippage_cap)
            
            print("\n================= ENTRY SIGNAL ===================")
            print(f"⭐ BEST TARGET: {best_symbol} (Velocity: +{best_velocity:,.0f} shares/min)")
            print(f"💰 TTP BALANCE: ${available_cash:,.2f}")
            print(f"🛡️ SLIPPAGE CAP: ${slippage_cap:,.2f} (1% ADV)")
            if available_cash > slippage_cap:
                print(f"⚠️ WARNING: Balance exceeds safe liquidity! Capping trade at ${slippage_cap:,.2f}!")
                print(f"⚠️ DO NOT invest the remaining ${available_cash - slippage_cap:,.2f} in this stock!")
            print(f"\n✅ ACTION: BUY ${trade_amount:,.2f} OF {best_symbol} NOW!")
            print("==================================================")
            
            input("\nPress ENTER once you have executed the trade on your broker...")
            current_position = best_symbol
            print(f"\n[+] Tracking {current_position}. Monitoring Order Flow Velocity...")
            
        else:
            # We are in a position, we must monitor for plateau!
            mom, vel = get_ofi_velocity(current_position)
            
            # Print status update on the same line
            print(f"\r  > Monitoring {current_position} | Flow: {mom:8.0f} | Velocity: {vel:8.0f} | Status: ", end="")
            
            if vel < 0:
                print("PLATEAU DETECTED! 🚨🚨")
                print("\n================= EXIT SIGNAL ====================")
                print(f"📉 The buying pressure on {current_position} has stalled and reversed!")
                print(f"✅ ACTION: SELL YOUR ENTIRE POSITION OF {current_position} NOW!")
                print("==================================================")
                
                input("\nPress ENTER once you have successfully sold the position...")
                
                while True:
                    try:
                        cash_input = input("\n[3] Enter your NEW total available TTP Cash Balance after that trade: $")
                        available_cash = float(cash_input.replace(',', ''))
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                        
                current_position = None # Reset for next buy
                print("\nReady to hunt the next runner...")
            else:
                print("Holding strong. ✅", end="")
                time.sleep(15) # Poll every 15 seconds while holding

if __name__ == "__main__":
    start_interactive_app()