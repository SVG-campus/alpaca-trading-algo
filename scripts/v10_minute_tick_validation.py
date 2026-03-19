import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# -------------------------------------------------------------
# V10 DEEP VALIDATION: INTRADAY STRESS TEST ENGINE
# -------------------------------------------------------------
print("Initializing V10 Intraday Deep Validation Engine...")

# Load Paper Keys
env_vars = {}
with open('.env', 'r') as f:
    for line in f:
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.strip().split('=', 1)
            env_vars[k.strip()] = v.strip().strip('"').strip("'")

api_key = env_vars.get("APCA_PAPER_API_KEY_ID")
api_secret = env_vars.get("APCA_PAPER_API_SECRET_KEY")

if not api_key or not api_secret:
    print("FATAL ERROR: Could not load Alpaca keys.")
    exit()

data_client = StockHistoricalDataClient(api_key, api_secret)

# The specific days we are testing to prove robustness across ALL market conditions
# 1. Mega Bull Day: Nov 6, 2024 (Post-Election Trump Rally, massive gap ups)
# 2. Mega Crash Day: Aug 5, 2024 (Yen Carry Trade unwinding, VIX spiked to 65)
# 3. Chop Day: Feb 12, 2025 (Massive CPI whiplash, volatile both ways)
# 4. Dead/Flat Day: Dec 24, 2024 (Christmas Eve half-day, zero volume/liquidity)

TEST_DAYS = [
    {"name": "Mega Bull", "date": "2024-11-06"},
    {"name": "Mega Crash", "date": "2024-08-05"},
    {"name": "High Chop", "date": "2025-02-12"},
    {"name": "Dead/Flat", "date": "2024-12-24"}
]

# We will take the exact universe logic from the Kaggle job. 
# We'll proxy a high-volatility basket that the GNN would likely select.
test_symbols = ["NVDA", "TSLA", "SMCI", "MSTR", "PLTR", "MARA", "COIN"]

def simulate_intraday_execution(symbol, target_date, tp_pct, sl_pct):
    # Fetch 1-Minute data for the entire target day
    start_dt = datetime.fromisoformat(f"{target_date}T09:30:00-04:00")
    end_dt = datetime.fromisoformat(f"{target_date}T16:00:00-04:00")
    
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt
    )
    
    try:
        bars = data_client.get_stock_bars(req)
        if not bars.data or symbol not in bars.data:
            return None
            
        df = pd.DataFrame([{"t": b.timestamp, "o": b.open, "h": b.high, "l": b.low, "c": b.close} for b in bars.data[symbol]])
        
        # Exact Logic: Wait 5 minutes after open to clear noise
        if len(df) <= 5: return None
        
        # We buy at the OPEN of the 6th minute
        entry_price = df.iloc[5]['o']
        entry_time = df.iloc[5]['t']
        
        # Calculate precise exact dollar targets
        target_price = entry_price * (1 + tp_pct)
        stop_price = entry_price * (1 + sl_pct)
        
        # Walk through the rest of the day minute-by-minute
        for i in range(6, len(df)):
            bar = df.iloc[i]
            
            # Did it hit the stop loss?
            if bar['l'] <= stop_price:
                return {"pnl": sl_pct, "reason": "STOP_LOSS", "time": bar['t'], "bars_held": i - 5}
                
            # Did it hit the take profit?
            if bar['h'] >= target_price:
                return {"pnl": tp_pct, "reason": "TAKE_PROFIT", "time": bar['t'], "bars_held": i - 5}
                
        # If we made it to exactly 5 minutes before close without hitting brackets
        close_idx = len(df) - 5
        if close_idx > 5:
            exit_price = df.iloc[close_idx]['c']
            pnl = (exit_price - entry_price) / entry_price
            return {"pnl": pnl, "reason": "TIME_STOP_5MIN", "time": df.iloc[close_idx]['t'], "bars_held": close_idx - 5}
            
        return None
        
    except Exception as e:
        print(f"API Error fetching {symbol}: {e}")
        return None

results = []

print("\nExecuting Minute-by-Minute Backtest across all Market Regimes...")
print("Testing Fixed Brackets (+14% / -9%) vs Dynamic Trailing Brackets (+8% / -4%)...\n")

# Let's test the +14% / -9% (Current V9) against the old +8% / -4% (V8)
# And let's test a +5% / -2.5% tight bracket.
brackets = [
    {"tp": 0.14, "sl": -0.09, "name": "V9 Aggressive (14/9)"},
    {"tp": 0.08, "sl": -0.04, "name": "V8 Balanced (8/4)"},
    {"tp": 0.05, "sl": -0.025, "name": "Tight Scalp (5/2.5)"}
]

for regime in TEST_DAYS:
    print(f"=== REGIME: {regime['name']} ({regime['date']}) ===")
    
    for bracket in brackets:
        regime_pnl = []
        for sym in test_symbols:
            res = simulate_intraday_execution(sym, regime['date'], bracket['tp'], bracket['sl'])
            if res:
                regime_pnl.append(res['pnl'])
                
        if regime_pnl:
            avg_pnl = np.mean(regime_pnl)
            win_rate = len([p for p in regime_pnl if p > 0]) / len(regime_pnl)
            print(f"  {bracket['name']:<25} | Avg PNL: {avg_pnl*100:>5.2f}% | Win Rate: {win_rate*100:>5.1f}%")
        else:
            print(f"  {bracket['name']:<25} | No valid data.")

print("\nAnalysis Complete.")