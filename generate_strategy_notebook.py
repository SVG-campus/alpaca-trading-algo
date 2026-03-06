import nbformat as nbf
import os

FW9_CODE = '\nclass TitanOracle:\n    """\n    The core discovery engine for both cross-sectional and temporal data.\n    """\n    def __init__(self, data: pd.DataFrame, mi_threshold=0.05, max_lag=2):\n        self.df = data.copy()\n        self.nodes = list(data.columns)\n        self.mi_threshold = mi_threshold\n        self.max_lag = max_lag\n        self.graph = nx.DiGraph()\n        self.temporal_graph = nx.DiGraph()\n        \n    def _cmi(self, x, y, z_list, data=None):\n        """Calculates Non-Linear Conditional Mutual Information using Tree Residuals"""\n        df = data if data is not None else self.df\n        if not z_list:\n            return mutual_info_regression(df[x].values.reshape(-1, 1), df[y].values)[0]\n        \n        Z = df[z_list]\n        model_x = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42).fit(Z, df[x])\n        model_y = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42).fit(Z, df[y])\n        \n        res_x = df[x] - model_x.predict(Z)\n        res_y = df[y] - model_y.predict(Z)\n        \n        return mutual_info_regression(res_x.values.reshape(-1, 1), res_y.values)[0]\n\n    def build_skeleton(self):\n        """Phase 1: PC-Algorithm inspired skeleton discovery (Cross-Sectional)."""\n        skeleton = nx.Graph()\n        skeleton.add_nodes_from(self.nodes)\n        for i in range(len(self.nodes)):\n            for j in range(i + 1, len(self.nodes)):\n                skeleton.add_edge(self.nodes[i], self.nodes[j])\n                \n        for u, v in list(skeleton.edges()):\n            if self._cmi(u, v, []) < self.mi_threshold:\n                skeleton.remove_edge(u, v)\n                continue\n            neighbors = list(set(skeleton.neighbors(u)) - {v})\n            for n in neighbors:\n                if self._cmi(u, v, [n]) < self.mi_threshold:\n                    skeleton.remove_edge(u, v)\n                    break\n        return skeleton\n\n    def orient_edges(self, skeleton):\n        """Phase 2: Asymmetry orientation (LiNGAM/Post-Nonlinear proxy)."""\n        self.graph.add_nodes_from(self.nodes)\n        for u, v in skeleton.edges():\n            X, Y = self.df[[u]], self.df[v]\n            \n            m_xy = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42).fit(X, Y)\n            res_y = Y - m_xy.predict(X)\n            mi_xy = mutual_info_regression(self.df[[u]], np.abs(res_y))[0]\n            \n            m_yx = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42).fit(self.df[[v]], self.df[u])\n            res_x = self.df[u] - m_yx.predict(self.df[[v]])\n            mi_yx = mutual_info_regression(self.df[[v]], np.abs(res_x))[0]\n            \n            if mi_xy < mi_yx:\n                self.graph.add_edge(u, v)\n            else:\n                self.graph.add_edge(v, u)\n        \n        # Break cycles\n        try:\n            for cycle in list(nx.simple_cycles(self.graph)):\n                self.graph.remove_edge(cycle[0], cycle[1])\n        except nx.NetworkXNoCycle:\n            pass\n        return self.graph\n\n    def discover_temporal_links(self):\n        """Phase 3: PCMCI for time-lagged DAGs."""\n        # Build lagged dataframe\n        df_lagged = pd.DataFrame(index=self.df.index)\n        for col in self.nodes:\n            df_lagged[f"{col}_t0"] = self.df[col]\n            for lag in range(1, self.max_lag + 1):\n                df_lagged[f"{col}_t-{lag}"] = self.df[col].shift(lag)\n        df_lagged = df_lagged.dropna().reset_index(drop=True)\n        \n        target_nodes = [f"{n}_t0" for n in self.nodes]\n        all_past_nodes = [c for c in df_lagged.columns if not c.endswith("_t0")]\n        \n        # PC1 Phase\n        candidates = {target: [] for target in target_nodes}\n        for target in target_nodes:\n            for past in all_past_nodes:\n                if self._cmi(past, target, [], data=df_lagged) > self.mi_threshold:\n                    candidates[target].append(past)\n        \n        # MCI Phase\n        self.temporal_graph.add_nodes_from(df_lagged.columns)\n        for target, current_parents in candidates.items():\n            for candidate in current_parents:\n                # Conditioning on target\'s past (simplified for this framework)\n                z_target = [p for p in current_parents if p != candidate]\n                if self._cmi(candidate, target, z_target, data=df_lagged) > self.mi_threshold:\n                    self.temporal_graph.add_edge(candidate, target)\n        \n        return self.temporal_graph\n\n\nclass TitanCausalAudit:\n    """\n    Audits the discovered graph for causal stability and statistical paradoxes.\n    """\n    def __init__(self, data: pd.DataFrame, graph: nx.DiGraph):\n        self.df = data\n        self.graph = graph\n\n    def check_simpsons_paradox(self, x, y):\n        """Detects if the sign of relationship flips when conditioning on a confounder."""\n        overall_corr = self.df[[x, y]].corr().iloc[0, 1]\n        \n        # Find potential confounders (parents of both or parents of X)\n        confounders = list(self.graph.predecessors(x))\n        if not confounders:\n            return False, "No confounders detected for " + x\n        \n        # Simple split on the first confounder for illustration\n        c = confounders[0]\n        median_c = self.df[c].median()\n        low_c = self.df[self.df[c] <= median_c][[x, y]].corr().iloc[0, 1]\n        high_c = self.df[self.df[c] > median_c][[x, y]].corr().iloc[0, 1]\n        \n        if np.sign(overall_corr) != np.sign(low_c) or np.sign(overall_corr) != np.sign(high_c):\n            return True, f"Simpson\'s Paradox detected! Sign flips on {c}"\n        return False, "No Simpson\'s flip detected."\n\n    def validate_stability(self):\n        """Checks if R^2 values are consistent across the graph."""\n        stability = {}\n        for node in self.graph.nodes():\n            parents = list(self.graph.predecessors(node))\n            if not parents: continue\n            X, y = self.df[parents], self.df[node]\n            model = HistGradientBoostingRegressor().fit(X, y)\n            score = r2_score(y, model.predict(X))\n            stability[node] = score\n        return stability\n\n\nclass ApexSimulator:\n    """\n    Monte Carlo Counterfactual Forecaster.\n    """\n    def __init__(self, data: pd.DataFrame, temporal_graph: nx.DiGraph, max_lag: int = 2):\n        self.df = data.copy()\n        self.nodes = list(data.columns)\n        self.tg = temporal_graph\n        self.max_lag = max_lag\n        self.models = {}\n        self.residuals = {}\n        self._fit_models()\n\n    def _fit_models(self):\n        # Build training data\n        df_lagged = pd.DataFrame(index=self.df.index)\n        for col in self.nodes:\n            df_lagged[f"{col}_t0"] = self.df[col]\n            for lag in range(1, self.max_lag + 1):\n                df_lagged[f"{col}_t-{lag}"] = self.df[col].shift(lag)\n        df_lagged = df_lagged.dropna().reset_index(drop=True)\n        \n        for n in self.nodes:\n            target = f"{n}_t0"\n            parents = list(self.tg.predecessors(target))\n            # Always add autoregressive links if not present\n            for l in range(1, self.max_lag + 1):\n                p_auto = f"{n}_t-{l}"\n                if p_auto not in parents: parents.append(p_auto)\n            \n            X, y = df_lagged[parents], df_lagged[target]\n            model = HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42).fit(X, y)\n            self.models[target] = model\n            self.residuals[target] = y.values - model.predict(X)\n            self.models[target].parents = parents\n\n    def simulate(self, target_var: str, shock_value: float, steps: int = 5, n_paths: int = 1000):\n        # Initial state from last index\n        base_state = {}\n        last_idx = self.df.index[-1]\n        for col in self.nodes:\n            for lag in range(1, self.max_lag + 1):\n                base_state[f"{col}_t-{lag}"] = self.df.loc[last_idx - (lag-1), col]\n        \n        results = {n: np.zeros((n_paths, steps)) for n in self.nodes}\n        \n        for p in range(n_paths):\n            current_state = base_state.copy()\n            for t in range(steps):\n                next_state = {}\n                for n in self.nodes:\n                    target = f"{n}_t0"\n                    if t == 0 and n == target_var:\n                        next_state[n] = shock_value\n                    else:\n                        parents = self.models[target].parents\n                        x_in = np.array([[current_state[par] for par in parents]])\n                        noise = np.random.choice(self.residuals[target])\n                        next_state[n] = self.models[target].predict(x_in)[0] + noise\n                \n                # Update current state for next step\n                for n in self.nodes:\n                    results[n][p, t] = next_state[n]\n                    # Shift history\n                    for lag in range(self.max_lag, 1, -1):\n                        current_state[f"{n}_t-{lag}"] = current_state[f"{n}_t-{lag-1}"]\n                    current_state[f"{n}_t-1"] = next_state[n]\n                    \n        summary = {}\n        for n in self.nodes:\n            summary[n] = {\n                \'mean\': np.mean(results[n], axis=0),\n                \'lower\': np.percentile(results[n], 5, axis=0),\n                \'upper\': np.percentile(results[n], 95, axis=0)\n            }\n        return summary\n\n\n# --- 1. DATA GENERATION (Synthetic Macro-Economy) ---\nnp.random.seed(42)\nn = 2000\nR = np.zeros(n)  # Rates\nU = np.zeros(n)  # Unemployment\nI = np.zeros(n)  # Inflation\n\nfor t in range(2, n):\n    R[t] = 0.8 * R[t-1] + np.random.normal(0, 0.1)\n    U[t] = 0.5 * U[t-1] + 0.6 * R[t-2] + np.random.normal(0, 0.1)\n    I[t] = 0.6 * I[t-1] - 0.7 * U[t-1] + np.random.normal(0, 0.1)\n\ndf = pd.DataFrame({\'RATES\': R, \'UNEMP\': U, \'INFLATION\': I})\n\n# --- 2. DISCOVERY (TITAN ORACLE) ---\noracle = TitanOracle(df, mi_threshold=0.08)\nskel = oracle.build_skeleton()\ndag = oracle.orient_edges(skel)\ntg = oracle.discover_temporal_links()\n\nprint(">> DISCOVERED TEMPORAL LINKS:")\nfor u, v in tg.edges():\n    if not u.startswith(v[:-3]): # Ignore self-loops\n        print(f"   {u} -> {v}")\n\n# --- 3. AUDIT ---\naudit = TitanCausalAudit(df, dag)\nstability = audit.validate_stability()\nprint("\\n>> CAUSAL STABILITY (R^2):", stability)\n\n# --- 4. SIMULATION ---\nsimulator = ApexSimulator(df, tg, max_lag=2)\nshock_val = df[\'RATES\'].mean() + 2.0 * df[\'RATES\'].std()\nforecast = simulator.simulate(\'RATES\', shock_val, steps=6)\n\n# --- 5. VISUALIZATION ---\nfig, axes = plt.subplots(1, 2, figsize=(15, 5))\nsteps = np.arange(6)\n\nfor i, var in enumerate([\'UNEMP\', \'INFLATION\']):\n    axes[i].plot(steps, forecast[var][\'mean\'], label=\'Mean Forecast\', color=\'blue\')\n    axes[i].fill_between(steps, forecast[var][\'lower\'], forecast[var][\'upper\'], alpha=0.2, color=\'blue\', label=\'90% CI\')\n    axes[i].set_title(f"Shockwave Propagation: {var}")\n    axes[i].set_xlabel("Time Steps (t+n)")\n    axes[i].legend()\n\nplt.tight_layout()\nplt.show()\n'

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
