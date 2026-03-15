import os
import shutil

os.makedirs('kaggle_jobs/intraday', exist_ok=True)
os.makedirs('kaggle_jobs/swing', exist_ok=True)

# Intraday
shutil.copyfile('run_discovery_kaggle_gpu.py', 'kaggle_jobs/intraday/run_discovery.py')

with open('run_discovery_kaggle_gpu.py', 'r') as f:
    code = f.read()

# Modify code for V7 Swing
code = code.replace(
    '''    # V8 INTRADAY OPTIMIZATION: We NEED volatility to make intraday profit!
    # Instead of dropping the most volatile stocks (which is for swing trading),
    # we DROP the bottom 50% (the flat, dead stocks) and keep the highly active ones,
    # stripping only the top 1% to avoid hyper-toxic anomalies.
    returns = df_raw.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    lower_bound = volatility.quantile(0.50) # Drop bottom 50% dead stocks
    upper_bound = volatility.quantile(0.99) # Drop top 1% anomalies
    
    optimized_universe = volatility[(volatility >= lower_bound) & (volatility <= upper_bound)].index.tolist()''',
    '''    # V7 SWING OPTIMIZATION: We want stable, causal growth for monthly swings!
    # As discovered in V7 Research, we drop the top 10% toxic anomalies
    # and keep the flat/stable ones for safety.
    returns = df_raw.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    vol_threshold = volatility.quantile(0.90) # Keep bottom 90%
    optimized_universe = volatility[volatility <= vol_threshold].index.tolist()'''
)

code = code.replace(
    '''    # Fast causal threshold for intraday noise adaptation
    oracle = TitanOracle(df_opt, mi_threshold=0.015, max_lag=1, tracker=tracker)''',
    '''    # Standard causal threshold for swing trading stability
    oracle = TitanOracle(df_opt, mi_threshold=0.01, max_lag=2, tracker=tracker)'''
)

code = code.replace(
    '''    if tracker: tracker.advance(1, phase="Preparing V8 Intraday node features", force=True)
    node_features = {}
    for symbol in optimized_universe:
        # V8 Intraday Features: Short-term momentum is king for day trading
        momentum_1d = (df_opt[symbol].iloc[-1] / df_opt[symbol].iloc[-2]) - 1
        momentum_5d = (df_opt[symbol].iloc[-1] / df_opt[symbol].iloc[-6]) - 1
        # Intraday ATR approximation (historical proxy)
        vol_5d = df_opt[symbol].tail(5).pct_change().std()
        
        node_features[symbol] = [momentum_1d, momentum_5d, vol_5d]
        if tracker: tracker.advance(1, phase="Preparing V8 Intraday node features")

    # Input dim upgraded from 2 to 3 for V8
    gnn_model = GraphDynamicsEngine(input_dim=3, hidden_dim=64, num_layers=3).to(DEVICE)''',
    '''    if tracker: tracker.advance(1, phase="Preparing V7 Swing node features", force=True)
    node_features = {}
    for symbol in optimized_universe:
        # V7 Swing Features: Longer-term momentum and structural stability
        momentum = (df_opt[symbol].iloc[-1] / df_opt[symbol].iloc[-20]) - 1
        volatility_20d = df_opt[symbol].tail(20).pct_change().std()
        node_features[symbol] = [momentum, volatility_20d]
        if tracker: tracker.advance(1, phase="Preparing V7 Swing node features")

    gnn_model = GraphDynamicsEngine(input_dim=2, hidden_dim=64, num_layers=3).to(DEVICE)'''
)

code = code.replace('V8 Intraday:', 'V7 Swing:')

with open('kaggle_jobs/swing/run_discovery.py', 'w') as f:
    f.write(code)

print("Kaggle jobs prepared.")
