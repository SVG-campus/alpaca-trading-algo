import json
import glob
from pathlib import Path
import pandas as pd

CACHE_PATH = Path("data/latest_discovery.json")

def load_latest_slippage_file():
    # Find the most recently generated slippage CSV
    files = glob.glob("data/slippage_1pct_adv_20d_*.csv")
    if not files:
        raise RuntimeError("No slippage CSV found in data/ directory.")
    # Sort by filename (which includes timestamp) and grab the latest
    return sorted(files)[-1]

def build_alpaca_allocation():
    if not CACHE_PATH.exists():
        raise RuntimeError(f"Discovery cache not found at {CACHE_PATH}")

    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    top_longs = cache.get("top_rankings", [])
    top_shorts = cache.get("top_short_rankings", [])
    
    slippage_file = load_latest_slippage_file()
    df_slip = pd.read_csv(slippage_file)
    
    # NO PRICE FILTERS - Use all stocks available in the slippage report
    valid_symbols = set(df_slip['symbol'].tolist())
    
    # Helper to look up slippage limit and price
    def get_asset_info(sym):
        row = df_slip[df_slip['symbol'] == sym]
        if row.empty: return None
        return {
            "price": float(row['last_close'].iloc[0]),
            "slippage_cap": float(row['cash_value_1pct_adv'].iloc[0])
        }

    # Custom Alpaca Budgets as requested ($50k, $500k, $5m)
    accounts = [50000, 500000, 5000000]
    
    print("==========================================================================")
    print("🦙 ALPACA MONTHLY REBALANCE - CASCADING ALLOCATION MASTER PLAN 🦙")
    print("==========================================================================")
    print(f"Generated from Cache: {cache['generated_at_utc']}")
    print("Rules: Full Kaggle universe enabled. Cascading 80/20 budget via 1% ADV slippage caps.\n")
    
    for total_bp in accounts:
        long_budget = total_bp * 0.80
        short_budget = total_bp * 0.20
        
        print(f"======================================================")
        print(f"💰 ALPACA ACCOUNT SIZE: ${total_bp:,.2f}")
        print(f"   Target Long (80%): ${long_budget:,.2f}")
        print(f"   Target Short (20%): ${short_budget:,.2f}")
        print(f"======================================================")
        
        # ---------------------
        # LONG ALLOCATIONS
        # ---------------------
        print(f"📈 LONG ALLOCATIONS")
        remaining_long = long_budget
        long_orders = []
        
        for pick in top_longs:
            sym = pick["symbol"]
            if sym not in valid_symbols: continue
            if remaining_long < 10.0: break
                
            info = get_asset_info(sym)
            if not info: continue
                
            allocation = min(remaining_long, info["slippage_cap"])
            shares = int(allocation // info["price"])
            
            if shares > 0:
                cost = shares * info["price"]
                long_orders.append({"symbol": sym, "shares": shares, "cost": cost})
                remaining_long -= cost
                print(f"  --> BUY {shares} shares of {sym} @ ~${info['price']:,.2f} = ${cost:,.2f}")
                
        if not long_orders:
            print("  --> No valid LONG targets found.")
            
        if remaining_long > 10.0:
            print(f"  ⚠️ Warning: Exhausted top 50 long list but still have ${remaining_long:,.2f} unallocated cash.")
            
        # ---------------------
        # SHORT ALLOCATIONS
        # ---------------------
        print(f"\n📉 SHORT ALLOCATIONS")
        remaining_short = short_budget
        short_orders = []
        
        for pick in top_shorts:
            sym = pick["symbol"]
            if sym not in valid_symbols: continue
            if remaining_short < 10.0: break
                
            info = get_asset_info(sym)
            if not info: continue
                
            allocation = min(remaining_short, info["slippage_cap"])
            shares = int(allocation // info["price"])
            
            if shares > 0:
                cost = shares * info["price"]
                short_orders.append({"symbol": sym, "shares": shares, "cost": cost})
                remaining_short -= cost
                print(f"  --> SHORT {shares} shares of {sym} @ ~${info['price']:,.2f} = ${cost:,.2f}")
                
        if not short_orders:
            print("  --> No valid SHORT targets found.")
            
        if remaining_short > 10.0:
            print(f"  ⚠️ Warning: Exhausted top 50 short list but still have ${remaining_short:,.2f} unallocated margin.")
            
        print("\n\n")

if __name__ == "__main__":
    build_alpaca_allocation()
