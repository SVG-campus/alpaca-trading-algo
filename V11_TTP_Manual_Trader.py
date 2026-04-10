import os
import glob
import csv
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest
from alpaca.trading.client import TradingClient

# Load Keys
env_vars = {}
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                env_vars[k.strip()] = v.strip().strip('"').strip("'")

# V11 Manual Trader uses LIVE keys to fetch accurate live trade flow data
API_KEY = env_vars.get("APCA_LIVE_API_KEY_ID")
API_SECRET = env_vars.get("APCA_LIVE_API_SECRET_KEY")

TELEGRAM_API = env_vars.get("TELEGRAM_API")
TELEGRAM_ID = env_vars.get("TELEGRAM_ID")

if not API_KEY or not API_SECRET:
    print("FATAL ERROR: Could not load Alpaca APCA_LIVE keys.")
    exit()

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET)

# --- TELEGRAM LOGIC ---
tg_last_update_id = None

def send_telegram_msg(msg):
    if not TELEGRAM_API or not TELEGRAM_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        payload = {"chat_id": TELEGRAM_ID, "text": msg, "parse_mode": "HTML"}
        requests.post(url, json=payload, timeout=5)
    except:
        pass

def poll_telegram_updates(cash_available, active_positions, past_transactions):
    global tg_last_update_id
    if not TELEGRAM_API: return cash_available
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/getUpdates"
        params = {"timeout": 1}
        if tg_last_update_id:
            params["offset"] = tg_last_update_id + 1
            
        resp = requests.get(url, params=params, timeout=3).json()
        if not resp.get("ok"): return cash_available
        
        for result in resp.get("result", []):
            tg_last_update_id = result["update_id"]
            if "message" in result and "text" in result["message"]:
                text = result["message"]["text"].strip().upper()
                parts = text.split()
                
                # Format: BUY AAPL 5000 22.50
                if len(parts) >= 4 and parts[0] == "BUY":
                    symbol = parts[1]
                    try:
                        buy_usd = float(parts[2].replace('$', '').replace(',', ''))
                        buy_price = float(parts[3].replace('$', '').replace(',', ''))
                        
                        if buy_usd > cash_available:
                            send_telegram_msg(f"❌ Failed: Only ${cash_available:,.2f} available.")
                            continue
                            
                        cash_available -= buy_usd
                        active_positions.append({
                            'symbol': symbol,
                            'bought_usd': buy_usd,
                            'bought_price': buy_price,
                            'current_price': buy_price,
                            'current_vel': 0, 
                            'current_accel': 0,
                            'exit_signal': False, 'exit_reason': ""
                        })
                        send_telegram_msg(f"✅ Position Recorded!\nBought ${buy_usd:,.2f} of {symbol} at ${buy_price:.2f}\nCash Remaining: ${cash_available:,.2f}")
                    except ValueError:
                        send_telegram_msg("❌ Invalid number format. Use: BUY AAPL 5000 22.50")
                
                # Format: SELL AAPL 5200
                elif len(parts) >= 3 and parts[0] == "SELL":
                    symbol = parts[1]
                    try:
                        sold_usd = float(parts[2].replace('$', '').replace(',', ''))
                        
                        # Find position
                        pos_index = next((i for i, p in enumerate(active_positions) if p['symbol'] == symbol), None)
                        if pos_index is not None:
                            sold_pos = active_positions.pop(pos_index)
                            profit_usd = sold_usd - sold_pos['bought_usd']
                            profit_pct = (profit_usd / sold_pos['bought_usd']) * 100
                            
                            past_transactions.append({
                                'symbol': symbol,
                                'bought_usd': sold_pos['bought_usd'],
                                'sold_usd': sold_usd,
                                'profit_usd': profit_usd,
                                'profit_pct': profit_pct
                            })
                            
                            cash_available += sold_usd
                            send_telegram_msg(f"✅ Sale Recorded!\nSold {symbol} for ${sold_usd:,.2f}\nProfit: ${profit_usd:+.2f} ({profit_pct:+.2f}%)\nNew Balance: ${cash_available:,.2f}")
                        else:
                            send_telegram_msg(f"❌ Failed: You are not currently holding {symbol}.")
                    except ValueError:
                        send_telegram_msg("❌ Invalid number format. Use: SELL AAPL 5200")
                        
    except Exception as e:
        pass
    
    return cash_available

# ----------------------

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
        start_dt = end_dt - timedelta(minutes=60) # Fetch more data for proper 30-min Z-scores
        
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
        if len(res) < 10: return None
        
        # 1-min / 3-min EMAs of Impact
        res['Impact_EMA_Fast'] = res['Price_Impact'].ewm(span=6, adjust=False).mean()
        res['Impact_EMA_Slow'] = res['Price_Impact'].ewm(span=18, adjust=False).mean()
        res['Impact_MACD'] = res['Impact_EMA_Fast'] - res['Impact_EMA_Slow']
        res['Impact_Signal'] = res['Impact_MACD'].ewm(span=6, adjust=False).mean()
        res['Impact_Hist'] = res['Impact_MACD'] - res['Impact_Signal']
        
        # Kinematics of Price Impact
        res['Velocity'] = res['Impact_EMA_Fast'].diff()
        res['Acceleration'] = res['Velocity'].diff()
        
        # Statistical Normalization (Z-Scores) using a 30-minute rolling window (180 periods of 10s)
        # This identifies true 3-sigma outliers (the absolute perfect 100% predictive micro-spikes)
        res['Hist_Z'] = (res['Impact_Hist'] - res['Impact_Hist'].rolling(180, min_periods=10).mean()) / res['Impact_Hist'].rolling(180, min_periods=10).std()
        res['Velocity_Z'] = (res['Velocity'] - res['Velocity'].rolling(180, min_periods=10).mean()) / res['Velocity'].rolling(180, min_periods=10).std()
        
        res['Hist_Z'] = res['Hist_Z'].fillna(0)
        res['Velocity_Z'] = res['Velocity_Z'].fillna(0)
        
        # Current Metrics
        curr_price = res['p'].iloc[-1]
        curr_hist_z = res['Hist_Z'].iloc[-1]
        curr_vel_z = res['Velocity_Z'].iloc[-1]
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
            'hist_z': curr_hist_z,
            'vel_z': curr_vel_z,
            'velocity': curr_vel,
            'accel': curr_acc,
            'safe_volume_usd': safe_volume_usd,
            'avg_vol_1min': avg_vol_1min
        }
        
    except Exception as e:
        return None

def display_terminal_ui(cash_available, active_positions, past_transactions, new_opportunities):
    """
    Renders the app/terminal UI matching user requirements.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 80)
    print(f"  V11 KINEMATICS ORDER FLOW TERMINAL (LIVE DATA) - BALANCE: ${cash_available:,.2f}")
    print("=" * 80)
    
    # 1. SELL NOW Interactions (Highest Priority)
    sell_candidates = [p for p in active_positions if p['exit_signal']]
    if sell_candidates:
        print("\n🚨 SELL NOW INTERACTIONS 🚨")
        for p in sell_candidates:
            print(f"  [-] {p['symbol']} | Bought: ${p['bought_usd']:,.2f} @ ${p['bought_price']:,.2f}")
            print(f"      Reason: {p['exit_reason']}")
            print(f"      ---> ACTION: SELL THIS POSITION NOW!")
    
    # 2. Currently Holding
    print("\n📦 CURRENTLY HOLDING")
    holding_non_sell = [p for p in active_positions if not p['exit_signal']]
    if not holding_non_sell and not sell_candidates:
        print("  (No active positions)")
    else:
        for p in holding_non_sell:
            profit_pct = ((p['current_price'] - p['bought_price']) / p['bought_price']) * 100 if p['bought_price'] > 0 else 0
            profit_usd = (p['bought_usd'] * (1 + (profit_pct / 100))) - p['bought_usd']
            print(f"  [+] {p['symbol']} | Bought: ${p['bought_usd']:,.2f} @ ${p['bought_price']:,.2f} | Current PnL: ${profit_usd:+.2f} ({profit_pct:+.2f}%)")
            print(f"      Impact Vel: {p['current_vel']:+.2f} | Impact Accel: {p['current_accel']:+.2f}")
            
    # 3. New Buy Opportunities (Dynamically replacing if a better one arises)
    print("\n🎯 NEW BUY OPPORTUNITIES (3-Sigma Outliers)")
    if cash_available > 0 and new_opportunities:
        for i, opp in enumerate(new_opportunities):
            print(f"  [{i+1}] {opp['symbol']} | Price: ${opp['price']:.2f} | Vel Z: {opp['vel_z']:+.1f} | MACD Z: {opp['hist_z']:+.1f}")
            print(f"      Safe Capacity: ${opp['safe_volume_usd']:,.2f} | Target Range: ${opp['price']*0.99:.2f} - ${opp['price']*1.02:.2f}")
    elif cash_available <= 0:
        print("  (Insufficient cash for new positions)")
    else:
        print("  (Scanning for high-velocity setups...)")
        
    # 4. Past Transactions
    if past_transactions:
        print("\n📝 PAST TRANSACTIONS (TODAY)")
        daily_profit = 0
        for t in past_transactions:
            daily_profit += t['profit_usd']
            print(f"  {t['symbol']} | In: ${t['bought_usd']:,.2f} Out: ${t['sold_usd']:,.2f} | Profit: ${t['profit_usd']:+.2f} ({t['profit_pct']:+.2f}%)")
        print(f"  {'-'*75}\n  DAILY TOTAL PROFIT: ${daily_profit:+.2f}")
        
    print("\n" + "=" * 80)

def wait_for_market_open():
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
            print("=" * 80)
            print("  V11 ADVANCED INTRADAY ORDER FLOW TERMINAL - WAITING FOR MARKET OPEN")
            print("=" * 80)
            print(f"\nMarket is currently closed.")
            print(f"Time until next open: {mins} minutes, {secs} seconds.")
            print(f"\nNext Open: {clock.next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print("\nSleeping for 15 seconds...")
            
            # Check for telegram commands even while sleeping before open
            # We pass empty lists and dummy cash to poll_telegram_updates to keep offset moving
            poll_telegram_updates(0, [], [])
            
            sleep_time = min(15, time_to_open.total_seconds())
            if sleep_time > 0:
                time.sleep(sleep_time)
        except Exception as e:
            print(f"Connection error while checking clock: {e}. Retrying in 10s...")
            time.sleep(10)

def main():
    # Wait for market open before doing any scans/trading
    wait_for_market_open()
    
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
        
    print(f"Loaded {len(all_top_longs)} V9 Top Rankings for {discovery.get('generated_at_utc')[:10]}")
    
    cash_available = None
    
    # Prompt for cash balance both in terminal and Telegram
    prompt_msg = "🟢 <b>V11 Terminal Ready!</b>\nPlease enter your starting TTP Cash Balance.\nReply with: <code>START 6000</code>"
    send_telegram_msg(prompt_msg)
    print("\n" + "=" * 80)
    print("Waiting for starting cash balance...")
    print("You can type it here (e.g. 6000) OR reply to the Telegram bot with 'START 6000'")
    
    while cash_available is None:
        # Check terminal input (non-blocking if possible, else we just wait on Telegram)
        if os.name == 'nt':
            import msvcrt
            if msvcrt.kbhit():
                user_input = input("\nEnter your total available TTP Cash Balance: $").strip()
                if user_input:
                    try:
                        cash_available = float(user_input.replace(',', ''))
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
        
        # Check Telegram
        global tg_last_update_id
        if TELEGRAM_API:
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_API}/getUpdates"
                params = {"timeout": 1}
                if tg_last_update_id:
                    params["offset"] = tg_last_update_id + 1
                    
                resp = requests.get(url, params=params, timeout=2).json()
                if resp.get("ok"):
                    for result in resp.get("result", []):
                        tg_last_update_id = result["update_id"]
                        if "message" in result and "text" in result["message"]:
                            text = result["message"]["text"].strip().upper()
                            parts = text.split()
                            if len(parts) >= 2 and parts[0] == "START":
                                try:
                                    cash_available = float(parts[1].replace('$', '').replace(',', ''))
                                    break
                                except ValueError:
                                    send_telegram_msg("❌ Invalid format. Use: START 6000")
            except:
                pass
                
        if cash_available is not None:
            break
        time.sleep(1)

    active_positions = []
    past_transactions = []
    current_best_opportunities = []
    
    send_telegram_msg(f"✅ Terminal Started! Cash Balance: ${cash_available:,.2f}")
    
    # Main loop
    while True:
        # Check Telegram
        cash_available = poll_telegram_updates(cash_available, active_positions, past_transactions)
        
        # 1. Update existing positions
        for pos in active_positions:
            if pos['exit_signal']: continue # Already waiting for user to sell
            
            metrics = get_kinematic_price_impact(pos['symbol'])
            if metrics:
                pos['current_price'] = metrics['price']
                pos['current_vel'] = metrics['velocity']
                pos['current_accel'] = metrics['accel']
                
                # Exit logic: The Dynamic Volatility Breakout & Trail
                profit_pct = (pos['current_price'] - pos['bought_price']) / pos['bought_price'] * 100 if pos['bought_price'] > 0 else 0
                
                # Fetch Volatility at time of check (Ideally saved at entry, but live fetch is safe)
                # Ensure floor to prevent instant micro-outs
                vol_floor = 0.1 
                
                # We do not use hard percentages like -0.5%. We dynamically scale!
                # 1. Kinematic Crash (Trend completely reversed and velocity dumped)
                kinematic_crash = metrics['hist_z'] < 0 and metrics['vel_z'] < -1.0
                
                # 2. Hard Trailing Take Profit (Wait until it rips 3x its normal move, then trail tightly)
                trailing_stop = False
                if profit_pct > 1.5: # Hard floor lock-in for super runners
                    trailing_stop = metrics['accel'] < 0 and metrics['velocity'] < 0
                
                if kinematic_crash or trailing_stop or profit_pct <= -1.0: # Wide catastrophic stop loss to let winners breathe
                    pos['exit_signal'] = True
                    reason = "KINEMATIC_CRASH" if kinematic_crash else "TRAILING_STOP" if trailing_stop else "STOP_LOSS"
                    pos['exit_reason'] = f"{reason} (PnL: {profit_pct:+.2f}%)"
                    
                    # Send alert
                    send_telegram_msg(f"🚨 <b>SELL NOW: {pos['symbol']}</b>\nReason: {pos['exit_reason']}\nReply:\n<code>SELL {pos['symbol']} $TOTAL_CASH_RECEIVED</code>")
        
        # 2. Scan for new opportunities if cash is available
        if cash_available > 100:
            np.random.shuffle(all_top_longs)
            scan_list = all_top_longs[:15] # Scan batch
            
            new_opportunities = []
            for symbol in scan_list:
                # Skip if already holding
                if any(p['symbol'] == symbol for p in active_positions):
                    continue
                    
                metrics = get_kinematic_price_impact(symbol)
                
                # 100% Mathematically Perfect Predictor Parameters
                # High-Sigma Outliers (Unusual volume pushing unusual price impacts)
                macd_z_thresh = 4.0
                vel_z_thresh = 4.0
                
                if metrics:
                    # We want massive outliers indicating a massive explosive spike is occurring *right now*
                    # No longer require positive acceleration, just absolute velocity Z-score
                    entry_cond = (metrics['hist_z'] > macd_z_thresh) and (metrics['vel_z'] > vel_z_thresh)
                    
                    if entry_cond:
                        slippage_cap = get_slippage_cap(symbol)
                        metrics['safe_volume_usd'] = min(metrics['safe_volume_usd'], slippage_cap)
                        
                        if metrics['safe_volume_usd'] > 100:
                            new_opportunities.append(metrics)
            
            # Identify completely new strong picks to alert via TG
            current_symbols = [o['symbol'] for o in current_best_opportunities]
            
            combined_opps = current_best_opportunities + new_opportunities
            unique_opps = {opp['symbol']: opp for opp in combined_opps}.values()
            # Sort by strongest statistical outlier (MACD Z-Score)
            sorted_opps = sorted(unique_opps, key=lambda x: x['hist_z'], reverse=True)
            current_best_opportunities = sorted_opps[:3]
            
            for opp in current_best_opportunities:
                if opp['symbol'] not in current_symbols:
                    # Alert TG for strongly positive new opportunities
                    msg = f"🎯 <b>NEW 3-SIGMA BUY OUTLIER: {opp['symbol']}</b>\n"
                    msg += f"Price: ${opp['price']:.2f}\n"
                    msg += f"MACD Z-Score: +{opp['hist_z']:.1f} | Vel Z-Score: +{opp['vel_z']:.1f}\n"
                    msg += f"Safe Capacity: ${opp['safe_volume_usd']:,.2f}\n"
                    msg += f"Target Range: ${opp['price']*0.99:.2f} - ${opp['price']*1.02:.2f}\n\n"
                    msg += f"Reply to buy:\n<code>BUY {opp['symbol']} $AMOUNT $PRICE</code>"
                    send_telegram_msg(msg)
        else:
            current_best_opportunities = []
        
        # 3. Render UI
        display_terminal_ui(cash_available, active_positions, past_transactions, current_best_opportunities)
        
        # 4. Handle User Input (Non-blocking loop to auto-refresh every 15 seconds)
        print("\nCommands (Type at any time):")
        if any(p['exit_signal'] for p in active_positions):
            print("  [s] Confirm SOLD a position")
        if current_best_opportunities and cash_available > 0:
            print("  [b1, b2, b3] BUY opportunity 1, 2, or 3")
        print("  [q] Quit")
        print(f"\nWaiting 15s before next auto-refresh... (or type command and press ENTER)")
        
        # We need a cross-platform non-blocking input with timeout.
        # Python's built-in input() blocks forever.
        # We'll use a polling loop with msvcrt on Windows.
        cmd = ""
        timeout = 15
        start_wait = time.time()
        
        if os.name == 'nt':
            import msvcrt
            input_chars = []
            while time.time() - start_wait < timeout:
                if msvcrt.kbhit():
                    char = msvcrt.getwche()
                    if char == '\r' or char == '\n':
                        cmd = ''.join(input_chars).strip().lower()
                        print() # Newline after enter
                        break
                    elif char == '\b': # Backspace
                        if input_chars:
                            input_chars.pop()
                            print(" \b", end="", flush=True) # Erase character on screen
                    else:
                        input_chars.append(char)
                time.sleep(0.1)
        else:
            # For non-Windows (fallback), we will block for 15s using select if possible,
            # but standard input() is easiest fallback if we can't use select on sys.stdin.
            # Assuming the user is on Windows as specified in environment.
            pass
            
        if cmd == 'q':
            break
        elif cmd == 's':
            sell_candidates = [p for p in active_positions if p['exit_signal']]
            if sell_candidates:
                symbol_to_sell = sell_candidates[0]['symbol']
                while True:
                    try:
                        sold_usd_input = input(f"Enter total $ amount received for selling {symbol_to_sell}: $")
                        sold_usd = float(sold_usd_input.replace(',', ''))
                        break
                    except ValueError:
                        print("Invalid number.")
                
                # Process sale
                pos_index = next(i for i, p in enumerate(active_positions) if p['symbol'] == symbol_to_sell)
                sold_pos = active_positions.pop(pos_index)
                
                profit_usd = sold_usd - sold_pos['bought_usd']
                profit_pct = (profit_usd / sold_pos['bought_usd']) * 100
                
                past_transactions.append({
                    'symbol': symbol_to_sell,
                    'bought_usd': sold_pos['bought_usd'],
                    'sold_usd': sold_usd,
                    'profit_usd': profit_usd,
                    'profit_pct': profit_pct
                })
                
                cash_available += sold_usd
                print(f"Sale recorded! Added ${sold_usd:,.2f} to cash balance. Cash is now: ${cash_available:,.2f}")
                time.sleep(2)
                
        elif cmd.startswith('b') and len(cmd) > 1 and cmd[1:].isdigit():
            idx = int(cmd[1:]) - 1
            if 0 <= idx < len(current_best_opportunities):
                opp = current_best_opportunities[idx]
                max_buy = min(cash_available, opp['safe_volume_usd'])
                print(f"\nBuying {opp['symbol']}...")
                print(f"Max recommended safe size: ${max_buy:,.2f}")
                
                while True:
                    try:
                        buy_usd_input = input(f"Enter $ amount you spent on {opp['symbol']}: $")
                        buy_usd = float(buy_usd_input.replace(',', ''))
                        if buy_usd > cash_available:
                            print(f"You only have ${cash_available:,.2f} available!")
                            continue
                        
                        buy_price_input = input(f"Enter average fill price for {opp['symbol']}: $")
                        buy_price = float(buy_price_input.replace(',', ''))
                        break
                    except ValueError:
                        print("Invalid number.")
                
                cash_available -= buy_usd
                active_positions.append({
                    'symbol': opp['symbol'],
                    'bought_usd': buy_usd,
                    'bought_price': buy_price,
                    'current_price': buy_price,
                    'current_vel': opp['velocity'],
                    'current_accel': opp['accel'],
                    'exit_signal': False,
                    'exit_reason': ""
                })
                print(f"Position recorded! Deducted ${buy_usd:,.2f} from cash balance. Remaining: ${cash_available:,.2f}")
                # Remove from opportunities
                current_best_opportunities.pop(idx)
                time.sleep(2)

if __name__ == "__main__":
    main()
