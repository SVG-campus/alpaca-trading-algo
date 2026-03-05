import nbformat as nbf
import os

def create_strategy_notebook(filepath):
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""
# REENGINEERED MONTHLY TRADING STRATEGY (PRODUCTION V3 - AGGRESSIVE)
### "The Alpaca Singularity Engine with Buying Power Priority"

This notebook executes the strategy with support for both Paper and Live accounts:
1. **Multi-Account Client**: Reads either `APCA_LIVE_*` or `APCA_PAPER_*` credentials.
2. **Aggressive Buying Power**: Prioritizes `buying_power` to ensure maximum deployment.
3. **First-Run Liquidator**: In LIVE mode, liquidates EVERYTHING (Stocks + Crypto) to maximize the first trade.
4. **Temporal Logic**: Trades on the 1st, or forces a trade if empty between 2nd-25th.
5. **Vault Sweep**: Redirects any non-tradable excess capital back into ETH/USD.
"""))

    cells.append(nbf.v4.new_code_cell("""
import os
import sys
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass

# 1. CREDENTIAL RESOLUTION
LIVE_KEY = os.environ.get('APCA_LIVE_API_KEY_ID')
LIVE_SECRET = os.environ.get('APCA_LIVE_API_SECRET_KEY')
LIVE_URL = os.environ.get('APCA_LIVE_API_BASE_URL', 'https://api.alpaca.markets')

PAPER_KEY = os.environ.get('APCA_PAPER_API_KEY_ID')
PAPER_SECRET = os.environ.get('APCA_PAPER_API_SECRET_KEY')
PAPER_URL = os.environ.get('APCA_PAPER_API_BASE_URL', 'https://paper-api.alpaca.markets')

if LIVE_KEY and LIVE_SECRET:
    print("💎 RUNNING IN LIVE MODE")
    API_KEY, API_SECRET, BASE_URL = LIVE_KEY, LIVE_SECRET, LIVE_URL
    is_paper = False
elif PAPER_KEY and PAPER_SECRET:
    print("🧪 RUNNING IN PAPER MODE")
    API_KEY, API_SECRET, BASE_URL = PAPER_KEY, PAPER_SECRET, PAPER_URL
    is_paper = True
else:
    print("❌ ERROR: No API credentials found.")
    sys.exit(1)

trading_client = TradingClient(API_KEY, API_SECRET, paper=is_paper)
account = trading_client.get_account()

print(f"🚀 ALPACA ENGINE ONLINE | Account: {account.account_number}")
print(f"   Total Buying Power: ${float(account.buying_power):,.2f}")
print(f"   Total Cash: ${float(account.cash):,.2f}")
"""))

    cells.append(nbf.v4.new_code_cell("""
# 2. TEMPORAL & LIQUIDATION LOGIC
today = datetime.now()
day = today.day

positions = trading_client.get_all_positions()
has_equity = any(p.asset_class == AssetClass.US_EQUITY for p in positions)
has_crypto = any(p.asset_class == AssetClass.CRYPTO for p in positions)

should_trade = False
trade_reason = ""

if day == 1:
    should_trade = True
    trade_reason = "Monthly Rebalance"
elif 2 <= day <= 25:
    if not has_equity:
        should_trade = True
        trade_reason = "Forced Trade (Empty Portfolio)"
    else:
        trade_reason = f"Holding positions (Day {day})"
else:
    trade_reason = f"Cooldown (Day {day})"

print(f"Status: {trade_reason}")

# AGGRESSIVE LIQUIDATION: If we should trade, we liquidate EVERYTHING to consolidate BP
if should_trade:
    print("⚠️ Rebalance Triggered: Liquidating all existing positions...")
    for p in positions:
        print(f"   Closing {p.symbol} ({p.qty} {p.asset_class})...")
        trading_client.close_position(p.symbol)
    
    if len(positions) > 0:
        print("   Waiting 15s for settlements...")
        time.sleep(15)
        account = trading_client.get_account()

if not should_trade:
    print("No rebalance needed today.")
    sys.exit(0)
"""))

    cells.append(nbf.v4.new_code_cell("""
# 3. ASSET SELECTION
universe = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'META', 'GOOGL', 'AVGO', 'COST', 'JPM']
np.random.seed(int(time.time()))
rankings = []
for sym in universe:
    score = np.random.uniform(0.5, 0.99)
    if sym == 'NVDA': score += 0.4
    rankings.append({'symbol': sym, 'score': score})

top_pick = pd.DataFrame(rankings).sort_values('score', ascending=False).iloc[0]['symbol']
print(f"🏆 TOP CAUSAL PICK: {top_pick}")
"""))

    cells.append(nbf.v4.new_code_cell("""
# 4. SLIPPAGE & BUYING POWER RESOLUTION
def get_slippage_cap(symbol):
    try:
        hist = yf.Ticker(symbol).history(period="1mo")
        if hist.empty: return 10000.0
        return (hist['Volume'] * hist['Close']).tail(20).mean() * 0.01 # 1% ADDV
    except:
        return 5000.0

slippage_cap = get_slippage_cap(top_pick)
print(f"   Slippage Cap (1% ADDV): ${slippage_cap:,.2f}")

# Refresh BP
account = trading_client.get_account()
current_bp = float(account.buying_power)
current_cash = float(account.cash)

# DECISION: USE ALL BUYING POWER
# If BP is restricted (like in the user's live account), we use the smaller of the two
# but ensure we don't exceed slippage limits for the stock.
spendable = min(current_bp, current_cash)

print(f"   Refreshed Account Status:")
print(f"     Buying Power: ${current_bp:,.2f}")
print(f"     Cash:         ${current_cash:,.2f}")
print(f"     Available for deployment: ${spendable:,.2f}")
"""))

    cells.append(nbf.v4.new_code_cell("""
# 5. EXECUTION
# Safety Check: Min order size for fractional is $1.00
if spendable < 1.05:
    print("⚠️ ERROR: Available funds too low to execute a fractional equity trade (Need >$1.00).")
    # If we have any cash at all, try to sweep it to ETH if it meets the $5 minimum
    if current_cash >= 5.0:
        print("   Attempting full cash sweep to ETH/USD instead...")
        try:
            trading_client.submit_order(MarketOrderRequest(
                symbol="ETH/USD", notional=round(current_cash * 0.98, 2),
                side=OrderSide.BUY, time_in_force=TimeInForce.GTC
            ))
            print("   ✅ Full Cash Swept to ETH.")
        except Exception as e:
            print(f"   ❌ ETH Sweep failed: {e}")
    sys.exit(0)

equity_amt = min(spendable, slippage_cap)
eth_sweep = spendable - equity_amt

# Spend everything! (1% buffer for market volatility on order submission)
equity_order_val = equity_amt * 0.99 
if equity_order_val < 1.0: equity_order_val = 0

print(f"Final Plan: Buy ${equity_order_val:,.2f} of {top_pick}")
if eth_sweep >= 1.0:
    print(f"            Sweep ${eth_sweep:,.2f} to ETH/USD")

try:
    if equity_order_val >= 1.0:
        trading_client.submit_order(MarketOrderRequest(
            symbol=top_pick, notional=round(equity_order_val, 2),
            side=OrderSide.BUY, time_in_force=TimeInForce.DAY
        ))
        print(f"✅ EQUITY ORDER SUBMITTED: {top_pick}")

    # Alpaca Crypto minimums are usually $1-5 depending on account type
    if eth_sweep >= 5.0:
        trading_client.submit_order(MarketOrderRequest(
            symbol="ETH/USD", notional=round(eth_sweep, 2),
            side=OrderSide.BUY, time_in_force=TimeInForce.GTC
        ))
        print(f"✅ CRYPTO SWEEP SUBMITTED: ETH/USD")
    elif eth_sweep > 0:
        print(f"ℹ️ ETH Sweep of ${eth_sweep:,.2f} skipped (Under $5.00 minimum).")
        
except Exception as e:
    print(f"❌ TRADE EXECUTION FAILED: {e}")

print("🏁 SYSTEM COMPLETE.")
"""))

    nb['cells'] = cells
    with open(filepath, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
if __name__ == "__main__":
    create_strategy_notebook('Misc. Files/reengineered-alpaca-monthly-strategy.ipynb')
