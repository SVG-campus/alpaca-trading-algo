import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print(">> INITIATING MATHEMATICAL OPTIMIZATION OF VOLATILITY FILTRATION")

raw_universe = [
    'NVDA', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'META', 'GOOGL', 'AVGO', 'COST', 'JPM',
    'AMD', 'INTC', 'CRM', 'ADBE', 'NFLX', 'PYPL', 'SQ', 'SHOP', 'UBER', 'ABNB',
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'DHR', 'LLY', 'BMY', 'AMGN',
    'V', 'MA', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'SPGI'
]

end_date = datetime.now()
start_date = end_date - timedelta(days=730)
print("Downloading 2 years of data...")
df_raw = yf.download(raw_universe, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))['Close']
df_raw = df_raw.dropna(axis=1) 
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)

returns = df_raw.pct_change().dropna()
volatility = returns.std() * np.sqrt(252)

# Grid Search for best cutoffs
best_sharpe = -999
best_params = {}

print("Running simulations on volatility cutoffs (Upper: Crash-prone, Lower: Dead stocks)...")

# We simulate a simple Momentum + Mean Reversion proxy to represent the Causal Oracle's ability 
# to pick the best stock FROM the remaining safe pool.
def simulate_portfolio(safe_stocks):
    if len(safe_stocks) < 5: return -999, -999
    # The oracle picks the stock with the best risk-adjusted momentum
    scores = (returns[safe_stocks].mean() / returns[safe_stocks].std())
    top_long = scores.idxmax()
    top_short = scores.idxmin()
    
    # 80/20 Hedge
    port_returns = (returns[top_long] * 0.80) + (returns[top_short] * -0.20)
    
    annual_ret = port_returns.mean() * 252
    annual_vol = port_returns.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    return annual_ret, sharpe

for upper_q in [0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]:
    for lower_q in [0.0, 0.05, 0.10, 0.15, 0.20]:
        if upper_q <= lower_q: continue
        
        upper_bound = volatility.quantile(upper_q)
        lower_bound = volatility.quantile(lower_q)
        
        safe_universe = volatility[(volatility <= upper_bound) & (volatility >= lower_bound)].index.tolist()
        
        ann_ret, sharpe = simulate_portfolio(safe_universe)
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = {
                'upper_q': upper_q,
                'lower_q': lower_q,
                'ann_ret': ann_ret,
                'sharpe': sharpe,
                'pool_size': len(safe_universe)
            }

print("\n>> OPTIMIZATION COMPLETE <<")
print(f"Old Parameters (V6): Drop Top 25% (0.75), Drop Bottom 10% (0.10)")
print(f"Optimal Parameters Discovered:")
print(f"  -> Keep stocks below Top {100 - best_params['upper_q']*100:.0f}% Volatility (Upper Quantile: {best_params['upper_q']})")
print(f"  -> Keep stocks above Bottom {best_params['lower_q']*100:.0f}% Volatility (Lower Quantile: {best_params['lower_q']})")
print(f"  -> Safe Pool Size: {best_params['pool_size']} stocks")
print(f"  -> Simulated APY: {best_params['ann_ret']:.2%}")
print(f"  -> Simulated Sharpe: {best_params['sharpe']:.2f}")

# Write to file so I can read it back easily
with open('Misc. Files/volatility_optimization_results.txt', 'w') as f:
    f.write(f"{best_params['upper_q']},{best_params['lower_q']},{best_params['ann_ret']}")
