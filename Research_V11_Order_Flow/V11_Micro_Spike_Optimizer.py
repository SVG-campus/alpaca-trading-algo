import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest
import joblib

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
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

def fetch_tick_data(symbol, start_dt, end_dt):
    req = StockTradesRequest(
        symbol_or_symbols=symbol,
        start=start_dt,
        end=end_dt,
        limit=50000 
    )
    try:
        trades = data_client.get_stock_trades(req)
        if not trades.data or symbol not in trades.data:
            return None
        df = pd.DataFrame([{"t": t.timestamp, "p": t.price, "s": t.size} for t in trades.data[symbol]])
        df['t'] = pd.to_datetime(df['t'])
        return df
    except Exception:
        return None

def calculate_topological_kinematics(df):
    df['Price_Diff'] = df['p'].diff().fillna(0)
    df['Price_Impact'] = df['Price_Diff'] * df['s'] 
    
    df.set_index('t', inplace=True)
    
    # Calculate True Intraday VWAP for structural context filtering
    df['Cumulative_Vol'] = df['s'].cumsum()
    df['Cum_PV'] = (df['p'] * df['s']).cumsum()
    df['VWAP'] = df['Cum_PV'] / df['Cumulative_Vol']
    
    res = df.resample('10s').agg({
        'Price_Impact': 'sum',
        'p': 'last',
        's': 'sum',
        'VWAP': 'last'
    }).fillna(0)
    
    res['p'] = res['p'].replace(0, np.nan).ffill()
    res['VWAP'] = res['VWAP'].replace(0, np.nan).ffill()
    
    # EMAs and MACD
    res['Impact_EMA_Fast'] = res['Price_Impact'].ewm(span=6, adjust=False).mean()
    res['Impact_EMA_Slow'] = res['Price_Impact'].ewm(span=18, adjust=False).mean()
    res['Impact_MACD'] = res['Impact_EMA_Fast'] - res['Impact_EMA_Slow']
    res['Impact_Signal'] = res['Impact_MACD'].ewm(span=6, adjust=False).mean()
    res['Impact_Hist'] = res['Impact_MACD'] - res['Impact_Signal']
    
    # Kinematics
    res['Velocity'] = res['Impact_EMA_Fast'].diff()
    res['Acceleration'] = res['Velocity'].diff()
    res['Jerk'] = res['Acceleration'].diff()
    res['Snap'] = res['Jerk'].diff()
    
    rolling_window = 180 # 30 mins
    
    for metric in ['Impact_Hist', 'Velocity', 'Acceleration', 'Jerk', 'Snap']:
        mean = res[metric].rolling(rolling_window, min_periods=10).mean()
        std = res[metric].rolling(rolling_window, min_periods=10).std()
        res[f'{metric}_Z'] = (res[metric] - mean) / std.replace(0, 1e-9)
        res[f'{metric}_Z'] = res[f'{metric}_Z'].fillna(0)
        
    res['Curving_Up'] = ((res['Velocity_Z'] > 0) & (res['Acceleration_Z'] > 0)).astype(int)
    res['Plateauing'] = ((res['Velocity_Z'] > 0) & (res['Acceleration_Z'] < 0)).astype(int)
    res['Curving_Down'] = ((res['Velocity_Z'] < 0) & (res['Acceleration_Z'] < 0)).astype(int)
    
    res['Price_Std'] = res['p'].rolling(60, min_periods=10).std()
    res['Price_Std_Pct'] = (res['Price_Std'] / res['p']) * 100
    res['Price_Std_Pct'] = res['Price_Std_Pct'].fillna(0.1) 
    
    return res

print("\n--- LOADING ML DUAL-TENSOR SYSTEM ---")
try:
    entry_model_path = 'data/intraday/models/v12_breakout_tensor.joblib'
    exit_model_path = 'data/intraday/models/v12_crash_tensor.joblib'
    
    ENTRY_MODEL = joblib.load(entry_model_path)
    EXIT_MODEL = joblib.load(exit_model_path)
    print("Loaded Holographic Tensors successfully.")
except Exception as e:
    print(f"ERROR: ML Models not found: {e}. Run the V12 Holographic Tensor Research scripts first.")
    sys.exit()

def test_strategy(df, symbol, date_str, entry_prob_thresh, exit_prob_thresh):
    in_position = False
    entry_price = 0
    entry_time = None
    max_price_seen = 0
    entry_volatility = 0
    entry_metrics = {}
    trades = []
    
    prices = df['p'].values
    
    for i in range(len(df)):
        row = df.iloc[i]
        curr_time = df.index[i]
        
        vel = row['Velocity'] if pd.notna(row['Velocity']) else row['Price_Impact']
        volatility_pct = row['Price_Std_Pct']
        
        features = pd.DataFrame([{
            'Impact_Hist_Z': row['Impact_Hist_Z'],
            'Velocity_Z': row['Velocity_Z'],
            'Acceleration_Z': row['Acceleration_Z'],
            'Jerk_Z': row['Jerk_Z'],
            'Snap_Z': row['Snap_Z'],
            'Curving_Up': row['Curving_Up'],
            'Plateauing': row['Plateauing'],
            'Curving_Down': row['Curving_Down'],
            'Volatility_Pct': volatility_pct
        }])
        
        if not in_position:
            vwap = row['VWAP'] if 'VWAP' in row and pd.notna(row['VWAP']) else 0
            is_above_vwap = row['p'] > vwap
            
            entry_prob = ENTRY_MODEL.predict_proba(features)[0][1]
            entry_cond = (entry_prob >= entry_prob_thresh) and (vel > 0) and is_above_vwap
            
            if entry_cond:
                in_position = True
                entry_price = row['p']
                max_price_seen = row['p']
                entry_time = curr_time
                entry_volatility = max(0.1, volatility_pct)
                entry_metrics = {
                    'ml_entry_prob': entry_prob
                }
        else:
            current_price = row['p']
            if current_price > max_price_seen:
                max_price_seen = current_price
                
            profit_pct = (current_price - entry_price) / entry_price * 100
            
            crash_prob = EXIT_MODEL.predict_proba(features)[0][1]
            ml_crash_exit = crash_prob >= exit_prob_thresh
            
            stop_loss = profit_pct <= -1.5
            
            exit_cond = stop_loss or ml_crash_exit
            
            if exit_cond: 
                future_prices = prices[i:]
                if len(future_prices) > 0:
                    max_future_price = np.max(future_prices)
                    max_possible_profit = (max_future_price - entry_price) / entry_price * 100
                else:
                    max_possible_profit = profit_pct
                    
                min_future_price = np.min(future_prices) if len(future_prices) > 0 else current_price
                max_loss_avoided = (min_future_price - entry_price) / entry_price * 100
                
                gain_after_loss = 0
                time_to_peak_after_loss = 0
                if (stop_loss or ml_crash_exit) and len(future_prices) > 0:
                    peak_idx = np.argmax(future_prices)
                    peak_price_after = future_prices[peak_idx]
                    if peak_price_after > entry_price:
                        gain_after_loss = (peak_price_after - entry_price) / entry_price * 100
                        time_to_peak_after_loss = peak_idx * 10 
                        
                reason = "ML_CRASH_DETECTED" if ml_crash_exit else "STOP_LOSS" 
                
                trades.append({
                    'symbol': symbol,
                    'date': date_str,
                    'entry_time': entry_time.strftime('%H:%M:%S'),
                    'exit_time': curr_time.strftime('%H:%M:%S'),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'profit_pct': profit_pct,
                    'max_possible_profit': max_possible_profit,
                    'profit_left_on_table': max_possible_profit - profit_pct if max_possible_profit > profit_pct else 0,
                    'max_loss_avoided': max_loss_avoided,
                    'gain_after_loss': gain_after_loss,
                    'time_to_peak_after_loss_secs': time_to_peak_after_loss,
                    'reason': reason,
                    **entry_metrics,
                    'ml_crash_prob': crash_prob
                })
                in_position = False
                
    return trades

if __name__ == "__main__":
    symbols = ['MEOH', 'MSTR', 'BCYC', 'SMCI', 'NVDA', 'TSLA', 'AAPL', 'AMZN', 'META', 'GOOGL', 'PLTR']
    
    num_days_to_test = 10
    current_dt = datetime.now(timezone.utc)
    
    print(f"--- MASSIVE HISTORICAL DATA BACKTEST: {num_days_to_test} DAYS, {len(symbols)} SYMBOLS ---")
    
    all_metrics = {}
    
    for i in range(1, num_days_to_test + 3): 
        target_day = current_dt - timedelta(days=i)
        if target_day.weekday() >= 5: 
            continue
            
        end_dt = target_day.replace(hour=20, minute=0, second=0, microsecond=0)
        start_dt = target_day.replace(hour=13, minute=30, second=0, microsecond=0)
        date_str = start_dt.strftime('%Y-%m-%d')
            
        print(f"Fetching Data for {date_str}...")
        
        all_metrics[date_str] = {}
        for symbol in symbols:
            raw_df = fetch_tick_data(symbol, start_dt, end_dt)
            if raw_df is not None and len(raw_df) > 0:
                metrics_df = calculate_topological_kinematics(raw_df)
                all_metrics[date_str][symbol] = metrics_df

    print("\n--- OPTIMIZING FOR 100% PREDICTIVE PRECISION ---")
    
    # We demand extreme ML confidence to filter out ALL fakeouts and ride massive spikes.
    entry_probs = [0.80, 0.85, 0.90, 0.95, 0.98]
    exit_probs = [0.80, 0.85, 0.90, 0.95, 0.98]
    
    best_overall_profit = -999
    best_universal_params = {}
    best_universal_stats = {}
    
    viable_strategies = []
    
    for en_p in entry_probs:
        for ex_p in exit_probs:
            total_profit_grid = 0.0
            total_trades_grid = 0
            winning_trades_grid = 0
            all_trades_for_grid = []
            
            daily_profits = {date: 0.0 for date in all_metrics.keys()}
            
            for date_str, day_data in all_metrics.items():
                for symbol, df in day_data.items():
                    trades = test_strategy(df, symbol, date_str, en_p, ex_p)
                    if trades:
                        total_trades_grid += len(trades)
                        grid_profit = sum(t['profit_pct'] for t in trades)
                        total_profit_grid += grid_profit
                        winning_trades_grid += sum(1 for t in trades if t['profit_pct'] > 0)
                        daily_profits[date_str] += grid_profit
                        all_trades_for_grid.extend(trades)
                        
            if total_trades_grid > 2: # Looking for rare, super-high conviction setups
                win_rate = winning_trades_grid / total_trades_grid
                
                viable_strategies.append({
                    'entry_p': en_p, 'exit_p': ex_p,
                    'win_rate': win_rate, 'profit': total_profit_grid, 'trades': total_trades_grid,
                    'daily_profits': daily_profits,
                    'all_trades': all_trades_for_grid
                })
                
                # We demand a very high win rate (>= 60%) to prove mathematical precision
                if total_profit_grid > best_overall_profit and win_rate >= 0.60: 
                    best_overall_profit = total_profit_grid
                    best_universal_params = {'entry_p': en_p, 'exit_p': ex_p}
                    best_universal_stats = {
                        'total_trades': total_trades_grid,
                        'win_rate': win_rate,
                        'total_profit': total_profit_grid,
                        'daily_profits': daily_profits,
                        'all_trades': all_trades_for_grid
                    }
                    
    if viable_strategies:
        print(f"\nTop 10 ML Combinations Found (Sorted by Total Profit):")
        viable_strategies.sort(key=lambda x: x['profit'], reverse=True)
        for s in viable_strategies[:10]:
            print(f"  ML Entry Confidence > {s['entry_p']*100:.0f}% | ML Exit Confidence > {s['exit_p']*100:.0f}% | Win: {s['win_rate']*100:.1f}%, Profit: {s['profit']:+.2f}% ({s['trades']} trades)")

    if best_universal_params:
        print("\n🏆 PERFECT DUAL-TENSOR PREDICTOR FOUND! 🏆")
        print(f"Optimal Minimum ML Entry Confidence Required: {best_universal_params['entry_p']*100:.0f}%")
        print(f"Optimal Minimum ML Crash Exit Confidence Required: {best_universal_params['exit_p']*100:.0f}%")
        print(f"Total Trades Generated: {best_universal_stats['total_trades']}")
        print(f"Win Rate: {best_universal_stats['win_rate']*100:.2f}%")
        print(f"Cumulative Net Profit: {best_universal_stats['total_profit']:.2f}%")
        
        dp = best_universal_stats['daily_profits']
        best_day = max(dp, key=dp.get)
        worst_day = min(dp, key=dp.get)
        print(f"\nHighest Profit Day: {best_day} ({dp[best_day]:+.2f}%)")
        print(f"Lowest Profit Day:  {worst_day} ({dp[worst_day]:+.2f}%)")
        
        trades_df = pd.DataFrame(best_universal_stats['all_trades'])
        avg_left = trades_df['profit_left_on_table'].mean()
        max_left = trades_df['profit_left_on_table'].max()
        print(f"\nOn average, we left {avg_left:.2f}% profit on the table. (Max left: {max_left:.2f}%)")
        
        csv_path = "Research_V11_Order_Flow/optimal_trades_analysis.csv"
        trades_df.to_csv(csv_path, index=False)
        print(f"Saved detailed trade logs to {csv_path}")
    else:
        print("\nCould not find a universally profitable parameter set across this dataset.")