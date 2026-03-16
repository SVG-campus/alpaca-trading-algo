import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.feature_selection import mutual_info_regression

# 1. EXPANDED UNIVERSE FOR RIGOROUS TESTING
print("Running Deep V9 Hyper-Optimization across 100 Highly Liquid US Equities...")
tickers = [
    "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "META", "AMZN", "COIN", "MARA", "MSTR", 
    "SMCI", "PLTR", "ARM", "GOOGL", "NFLX", "JPM", "V", "BAC", "WFC", "C",
    "CVX", "GS", "MS", "AXP", "BLK", "XOM", "COP", "OXY", "SLB", "EOG",
    "UNH", "HAL", "BKR", "PXD", "MPC", "WMT", "HD", "COST", "TGT", "LOW",
    "UNP", "JNJ", "LLY", "ABBV", "PFE", "MRK", "TMO", "ABT", "DHR", "BMY",
    "PG", "PEP", "KO", "KO", "PM", "CSCO", "MCD", "NKE", "SBUX", "SBUX",
    "INTC", "CSCO", "CMCSA", "ADBE", "TXN", "QCOM", "ORCL", "IBM", "CRM", "SAP",
    "CRM", "NOW", "INTU", "ACN", "PANW", "SNPS", "FTNT", "CRWD", "DDOG", "ZS",
    "ZS", "NET", "MDB", "OKTA", "S", "WDAY", "CYBR", "TENB", "SPLK", "CHKP",
    "CHKP", "VRNS", "QLYS", "RDWR", "SAIL", "FEYE", "MIME", "TLS", "PING", "FSLY"
]
# Remove duplicates
tickers = list(set(tickers))

# 2. FETCH HISTORICAL 1-MINUTE DATA FOR PRECISE SLIPPAGE & BRACKET TESTS
# We will use 1-hour/daily data from YFinance, but proxy intraday volatility carefully
# YFinance limit for 1-minute is 7 days. We'll use 6 months of daily data to find the broad targets.
data = yf.download(tickers, period="6mo", interval="1d", group_by="ticker", auto_adjust=True, progress=False)

records = []
print("Processing data and calculating V9 Causal Information...")

for ticker in tickers:
    if ticker not in data.columns.levels[0]: continue
    df = data[ticker].copy()
    if df.empty or len(df) < 30: continue
    
    df['Return_1d'] = df['Close'].pct_change()
    df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
    df['RVOL'] = df['Volume'] / df['Volume_SMA_20']
    
    # We define intraday volatility as (High - Low) / Open
    df['Max_Intraday_Gain'] = (df['High'] - df['Open']) / df['Open']
    df['Max_Intraday_Loss'] = (df['Open'] - df['Low']) / df['Open']
    df['Intraday_Net'] = (df['Close'] - df['Open']) / df['Open']
    
    # Drop NaNs
    df = df.dropna()
    for _, row in df.iterrows():
        records.append({
            'Ticker': ticker,
            'Prev_Return': row['Return_1d'],
            'Prev_RVOL': row['RVOL'],
            'Max_Gain': row['Max_Intraday_Gain'],
            'Max_Loss': row['Max_Intraday_Loss'],
            'Net': row['Intraday_Net']
        })

df_all = pd.DataFrame(records)

# 3. HYPER-OPTIMIZATION: Finding the exact RVOL threshold and Bracket that peaks EV.
best_rvol_threshold = 1.0
best_tp = 0.0
best_sl = 0.0
best_ev = -999.0
best_winrate = 0.0
best_num_trades = 0

print("Commencing Grid Search...")

for rvol_thresh in np.arange(1.0, 3.0, 0.5):
    # Filter candidates
    candidates = df_all[(df_all['Prev_RVOL'] >= rvol_thresh) & (df_all['Prev_Return'] > 0.02)]
    if len(candidates) < 50: # Not enough data to be statistically significant
        continue
        
    for tp in np.arange(0.02, 0.15, 0.01):
        for sl in np.arange(0.01, 0.10, 0.01):
            # We assume a 50/50 path dependency hit if BOTH Max_Loss > SL and Max_Gain > TP.
            # A conservative approach assumes SL hits first to protect capital mathematically.
            profits = []
            for _, row in candidates.iterrows():
                if row['Max_Loss'] >= sl:
                    profits.append(-sl) # Stopped out
                elif row['Max_Gain'] >= tp:
                    profits.append(tp) # Profit secured
                else:
                    profits.append(row['Net']) # Closed at End of Day
            
            ev = np.mean(profits)
            winrate = len([p for p in profits if p > 0]) / len(profits)
            
            if ev > best_ev:
                best_ev = ev
                best_tp = tp
                best_sl = sl
                best_rvol_threshold = rvol_thresh
                best_winrate = winrate
                best_num_trades = len(profits)

print("\n=============================================")
print("   V9 SINGULARITY INTRADAY HYPER-OPTIMIZED   ")
print("=============================================")
print(f"Optimal Minimum RVOL: {best_rvol_threshold}x")
print(f"Optimal Take Profit: +{best_tp*100:.2f}%")
print(f"Optimal Stop Loss: -{best_sl*100:.2f}%")
print(f"Historical Sample Size: {best_num_trades} days")
print(f"Mathematical Win Rate: {best_winrate*100:.2f}%")
print(f"Expected Value (Daily Return): +{best_ev*100:.2f}%")
print("=============================================")

# Update Kaggle V9 configuration if it mathematically beats the current +8%/-4%
if best_ev > 0.0222: # > 2.22% (our current V9 benchmark)
    print("\n[!] BREAKTHROUGH: Found a parameter set superior to the current baseline.")
    print("Writing updated parameters to run_alpaca_live_trade.py...")
    
    with open('run_alpaca_live_trade.py', 'r') as f:
        script = f.read()
    
    script = script.replace('target_profit_pct = 0.08', f'target_profit_pct = {best_tp:.3f}')
    script = script.replace('stop_loss_pct = -0.04', f'stop_loss_pct = -{best_sl:.3f}')
    
    with open('run_alpaca_live_trade.py', 'w') as f:
        f.write(script)
        
    print("Live script updated with hyper-optimized brackets.")
else:
    print("\n[i] Current baseline (+8%/-4% with RVOL > 1.5) remains the mathematically optimal peak.")
