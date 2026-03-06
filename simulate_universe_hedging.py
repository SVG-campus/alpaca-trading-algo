import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# We simulate a "Wide Universe" vs "Narrow Universe"
# Narrow: The original 10
narrow_universe = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'META', 'GOOGL', 'AVGO', 'COST', 'JPM']

# Wide: Add 40 more liquid stocks across sectors
wide_universe = narrow_universe + [
    'AMD', 'INTC', 'CRM', 'ADBE', 'NFLX', 'PYPL', 'SQ', 'SHOP', 'UBER', 'ABNB',
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'DHR', 'LLY', 'BMY', 'AMGN',
    'V', 'MA', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'SPGI',
    'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'DIS', 'CMCSA', 'VZ'
]

end_date = datetime.now()
start_date = end_date - timedelta(days=365) # 1 year

print("Downloading Wide Universe Data...")
df_wide = yf.download(wide_universe, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))['Close']
df_wide = df_wide.dropna(axis=1) # Drop any that don't have full history

if isinstance(df_wide.columns, pd.MultiIndex):
    df_wide.columns = df_wide.columns.get_level_values(0)

print(f"Loaded {len(df_wide.columns)} stocks successfully.")

# --- VOLATILITY FILTERING ---
# Calculate daily returns
returns = df_wide.pct_change().dropna()

# Calculate annualized volatility
volatility = returns.std() * np.sqrt(252)

# We want to drop the top 20% most volatile stocks to prevent "crash" scenarios
vol_threshold = volatility.quantile(0.80)
safe_stocks = volatility[volatility <= vol_threshold].index.tolist()

print(f"Volatility Filter: Dropped {len(df_wide.columns) - len(safe_stocks)} highly volatile stocks.")
print(f"Safe Universe Size: {len(safe_stocks)}")

# --- HEDGING SIMULATION (LONG / SHORT) ---
# We will simulate a simplistic causal score (since running full PC-Algorithm on 40 stocks takes ~10 mins)
# We use a proxy: Momentum + Mean Reversion + Low Volatility
scores = (returns.mean() / returns.std()) * np.sqrt(252) # Sharpe Ratio proxy for stability

safe_scores = scores[safe_stocks].sort_values(ascending=False)

top_longs = safe_scores.head(3).index.tolist()
top_shorts = safe_scores.tail(3).index.tolist() # The ones with worst stable trajectory

print(f"Top 3 Long Candidates (Hedging): {top_longs}")
print(f"Top 3 Short Candidates (Hedging): {top_shorts}")

# Calculate theoretical portfolio return of Hedged vs Unhedged
# Simulate past 30 days
recent_returns = returns.tail(30)

unhedged_ret = recent_returns[narrow_universe[0]].mean() # Assume we put it all in the top narrow stock
hedged_ret_long = recent_returns[top_longs].mean(axis=1).mean()
hedged_ret_short = -1 * recent_returns[top_shorts].mean(axis=1).mean() # Shorting

hedged_total = (hedged_ret_long * 0.5) + (hedged_ret_short * 0.5)

print(f"Simulated Unhedged (Top 1) 30-Day Mean Daily Return: {unhedged_ret:.4%}")
print(f"Simulated Hedged (Long/Short) 30-Day Mean Daily Return: {hedged_total:.4%}")
print(f"Hedged Volatility: {(recent_returns[top_longs].mean(axis=1)*0.5 - recent_returns[top_shorts].mean(axis=1)*0.5).std():.4%}")
print(f"Unhedged Volatility: {recent_returns[narrow_universe[0]].std():.4%}")

with open('Misc. Files/universe_expansion_math.txt', 'w') as f:
    f.write(f"Safe Stocks: {safe_stocks}\n")
    f.write(f"Longs: {top_longs}\n")
    f.write(f"Shorts: {top_shorts}\n")
    f.write(f"Hedged Return: {hedged_total}\n")
    f.write(f"Unhedged Return: {unhedged_ret}\n")
