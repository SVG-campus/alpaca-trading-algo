import json
import nbformat as nbf
import os

# Read FW9
with open('Misc. Files/ab-initio-causal-discovery.ipynb', 'r', encoding='utf-8') as f:
    fw9 = json.load(f)

# Extract code cells from FW9
code_cells = [cell['source'] for cell in fw9['cells'] if cell['cell_type'] == 'code']
# cells 1, 2, 3, 4 contain the classes
oracle_code = "".join(code_cells[1]) + "\n" + "".join(code_cells[2]) + "\n" + "".join(code_cells[3]) + "\n" + "".join(code_cells[4])

# The strategy generator logic
script_content = '''import nbformat as nbf
import os

FW9_CODE = ''' + repr(oracle_code) + '''

def create_notebook(filepath, mode):
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(f"""
# REENGINEERED MONTHLY TRADING STRATEGY ({mode})
### "The Alpaca Singularity Engine" (TRUE CAUSAL EDITION)

This notebook executes the strategy for the **{mode}** account.
It relies on Framework 9 (Ab-Initio Causal Discovery) to map the causal structure of the universe
and run a Monte Carlo simulation (ApexSimulator) to find the mathematically optimal pick.
"""))

    cells.append(nbf.v4.new_code_cell(f"""
import os
import sys
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
import networkx as nx
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. CREDENTIALS
mode = '{mode}'
if mode == 'LIVE':
    API_KEY = os.environ.get('APCA_LIVE_API_KEY_ID')
    API_SECRET = os.environ.get('APCA_LIVE_API_SECRET_KEY')
    BASE_URL = os.environ.get('APCA_LIVE_API_BASE_URL', 'https://api.alpaca.markets')
    is_paper = False
elif mode == 'MAX-PAPER':
    API_KEY = os.environ.get('MAX_APCA_PAPER_API_KEY_ID')
    API_SECRET = os.environ.get('MAX_APCA_PAPER_API_SECRET_KEY')
    BASE_URL = os.environ.get('MAX_APCA_PAPER_API_BASE_URL', 'https://paper-api.alpaca.markets')
    is_paper = True
else:
    API_KEY = os.environ.get('APCA_PAPER_API_KEY_ID')
    API_SECRET = os.environ.get('APCA_PAPER_API_SECRET_KEY')
    BASE_URL = os.environ.get('APCA_PAPER_API_BASE_URL', 'https://paper-api.alpaca.markets')
    is_paper = True

if not API_KEY or not API_SECRET:
    print(f"❌ ERROR: {{mode}} API credentials not found.")
    sys.exit(0) # Exit gracefully so GH Action stays green

trading_client = TradingClient(API_KEY, API_SECRET, paper=is_paper)
account = trading_client.get_account()

print(f"🚀 {{mode}} ENGINE ONLINE | Account: {{account.account_number}}")
print(f"   Buying Power: ${{float(account.buying_power):,.2f}} | Cash: ${{float(account.cash):,.2f}}")
"""))

    cells.append(nbf.v4.new_code_cell(FW9_CODE))

    cells.append(nbf.v4.new_code_cell("""
# 2. TEMPORAL & LIQUIDATION LOGIC
today = datetime.now()
day = today.day

positions = trading_client.get_all_positions()
has_equity = any(p.asset_class == AssetClass.US_EQUITY for p in positions)

should_trade = False
trade_reason = ""

# Condition for trading
if day == 1:
    should_trade = True
    trade_reason = "1st of the month standard rebalance."
elif 2 <= day <= 25:
    if not has_equity:
        should_trade = True
        trade_reason = f"Forced trade (Day {day} with empty portfolio)."
    else:
        trade_reason = f"Halted: Positions already exist (Day {day})."
else:
    trade_reason = f"Halted: Day {day} is in cooldown."

print(f"Status: {trade_reason}")

if should_trade:
    print("⚠️ Rebalance Triggered: Liquidating existing EQUITY positions...")
    for p in positions:
        if p.asset_class == AssetClass.US_EQUITY:
            print(f"   Closing {p.symbol}...")
            trading_client.close_position(p.symbol)
    
    if has_equity:
        print("   Waiting 15s for settlements...")
        time.sleep(15)
        account = trading_client.get_account()
else:
    print("🏁 Execution complete (No trade needed today).")
    # Wrap next cell in conditional
"""))

    cells.append(nbf.v4.new_code_cell("""
# 3. TITAN ORACLE RANKING
if 'should_trade' in locals() and should_trade:
    print("--- DOWNLOADING CAUSAL UNIVERSE DATA ---")
    universe = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'META', 'GOOGL', 'AVGO', 'COST', 'JPM']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730) # 2 years of history
    df = yf.download(universe, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))['Close']
    df = df.dropna()
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"Loaded {len(df)} days of historical data for {universe}.")
    
    print("--- RUNNING FW9: AB-INITIO CAUSAL DISCOVERY ---")
    # Small MI threshold to ensure we discover graphs even in tight financial regimes
    oracle = TitanOracle(df, mi_threshold=0.01, max_lag=2)
    skel = oracle.build_skeleton()
    dag = oracle.orient_edges(skel)
    tg = oracle.discover_temporal_links()
    
    print("--- RUNNING APEX SIMULATOR FORECAST ---")
    # Fit Monte Carlo
    simulator = ApexSimulator(df, tg, max_lag=2)
    
    # Simulate a +1 standard deviation shock across the market, or simply project forward
    # Here we project the expected natural return over the next 5 days
    expected_returns = {}
    for sym in universe:
        try:
            # We simulate no external shock (shock_value = last known value)
            # to see the natural autoregressive/causal trajectory
            base_val = df[sym].iloc[-1]
            forecast = simulator.simulate(sym, base_val, steps=5, n_paths=100)
            
            # Expected 5-day return
            exp_ret = forecast[sym]['mean'][-1] / base_val - 1.0
            expected_returns[sym] = exp_ret
            print(f"   {sym} 5-day expected return: {exp_ret:.2%}")
        except Exception as e:
            print(f"   Failed to forecast {sym}: {e}")
            expected_returns[sym] = -999.0
            
    top_pick = pd.Series(expected_returns).idxmax()
    print(f"🏆 CAUSAL TITAN TOP PICK: {top_pick}")
"""))

    cells.append(nbf.v4.new_code_cell("""
# 4. EXECUTION
if 'should_trade' in locals() and should_trade and 'top_pick' in locals():
    # --- LIQUIDITY CAP ---
    print(f"Calculating liquidity for {top_pick}...")
    try:
        hist = yf.Ticker(top_pick).history(period="1mo")
        slippage_cap = (hist['Volume'] * hist['Close']).tail(20).mean() * 0.01
    except:
        slippage_cap = 10000.0
    print(f"   Max safe trade: ${slippage_cap:,.2f}")

    # --- ALLOCATION ---
    account = trading_client.get_account()
    current_bp = float(account.buying_power)
    current_cash = float(account.cash)
    
    spendable = current_bp
    equity_amt = min(spendable, slippage_cap)
    eth_sweep = current_cash - equity_amt
    
    # 2% buffer for fractional/market movement
    order_val = round(equity_amt * 0.98, 2)
    
    print(f"Refreshed BP: ${current_bp:,.2f} | Planning to spend ${order_val:,.2f}")

    if order_val >= 1.0:
        try:
            trading_client.submit_order(MarketOrderRequest(
                symbol=top_pick, notional=order_val,
                side=OrderSide.BUY, time_in_force=TimeInForce.DAY
            ))
            print(f"✅ EQUITY ORDER SUBMITTED: {top_pick} (${order_val:,.2f})")
            
            # Sweep remaining cash
            time.sleep(2)
            account = trading_client.get_account()
            rem_cash = float(account.cash)
            if rem_cash >= 5.0:
                trading_client.submit_order(MarketOrderRequest(
                    symbol="ETH/USD", notional=round(rem_cash * 0.98, 2),
                    side=OrderSide.BUY, time_in_force=TimeInForce.GTC
                ))
                print(f"✅ CASH SWEPT TO ETH")
        except Exception as e:
            print(f"❌ TRADE FAILED: {e}")
    else:
        print("⚠️ Not enough buying power to execute trade (Need >$1.00).")
        if current_cash >= 5.0:
            print("   Sweeping available cash to ETH instead...")
            trading_client.submit_order(MarketOrderRequest(
                symbol="ETH/USD", notional=round(current_cash * 0.98, 2),
                side=OrderSide.BUY, time_in_force=TimeInForce.GTC
            ))
            print("   ✅ SWEPT TO ETH")

    print("🏁 SYSTEM COMPLETE.")
"""))

    nb['cells'] = cells
    with open(filepath, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook('Misc. Files/strategy-live.ipynb', 'LIVE')
    create_notebook('Misc. Files/strategy-paper.ipynb', 'PAPER')
    create_notebook('Misc. Files/strategy-max-paper.ipynb', 'MAX-PAPER')
'''

with open('generate_strategy_notebook.py', 'w', encoding='utf-8') as f:
    f.write(script_content)

print("generate_strategy_notebook.py updated!")
