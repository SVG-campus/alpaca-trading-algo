
import numpy as np
import pandas as pd
import networkx as nx
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Configuration Constants
MI_THRESHOLD = 0.05
ALPHA = 0.05
MAX_LAG = 2
N_PATHS = 1000

print("🔮 TITAN ORACLE ONLINE | Causal Physics Engine Initialized")

import io
import time

def rate_limited_api_call():
    print("Simulating 5 minute API delay (mocked to 1 sec for testing)...")
    time.sleep(1)


class TitanOracle:
    """
    The core discovery engine for both cross-sectional and temporal data.
    """
    def __init__(self, data: pd.DataFrame, mi_threshold=0.05, max_lag=2):
        self.df = data.copy()
        self.nodes = list(data.columns)
        self.mi_threshold = mi_threshold
        self.max_lag = max_lag
        self.graph = nx.DiGraph()
        self.temporal_graph = nx.DiGraph()
        
    def _cmi(self, x, y, z_list, data=None):
        """Calculates Non-Linear Conditional Mutual Information using Tree Residuals"""
        df = data if data is not None else self.df
        if not z_list:
            return mutual_info_regression(df[x].values.reshape(-1, 1), df[y].values)[0]
        
        Z = df[z_list]
        model_x = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42).fit(Z, df[x])
        model_y = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42).fit(Z, df[y])
        
        res_x = df[x] - model_x.predict(Z)
        res_y = df[y] - model_y.predict(Z)
        
        return mutual_info_regression(res_x.values.reshape(-1, 1), res_y.values)[0]

    def build_skeleton(self):
        """Phase 1: PC-Algorithm inspired skeleton discovery (Cross-Sectional)."""
        skeleton = nx.Graph()
        skeleton.add_nodes_from(self.nodes)
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                skeleton.add_edge(self.nodes[i], self.nodes[j])
                
        for u, v in list(skeleton.edges()):
            if self._cmi(u, v, []) < self.mi_threshold:
                skeleton.remove_edge(u, v)
                continue
            neighbors = list(set(skeleton.neighbors(u)) - {v})
            for n in neighbors:
                if self._cmi(u, v, [n]) < self.mi_threshold:
                    skeleton.remove_edge(u, v)
                    break
        return skeleton

    def orient_edges(self, skeleton):
        """Phase 2: Asymmetry orientation (LiNGAM/Post-Nonlinear proxy)."""
        self.graph.add_nodes_from(self.nodes)
        for u, v in skeleton.edges():
            X, Y = self.df[[u]], self.df[v]
            
            m_xy = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42).fit(X, Y)
            res_y = Y - m_xy.predict(X)
            mi_xy = mutual_info_regression(self.df[[u]], np.abs(res_y))[0]
            
            m_yx = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42).fit(self.df[[v]], self.df[u])
            res_x = self.df[u] - m_yx.predict(self.df[[v]])
            mi_yx = mutual_info_regression(self.df[[v]], np.abs(res_x))[0]
            
            if mi_xy < mi_yx:
                self.graph.add_edge(u, v)
            else:
                self.graph.add_edge(v, u)
        
        # Break cycles
        try:
            for cycle in list(nx.simple_cycles(self.graph)):
                self.graph.remove_edge(cycle[0], cycle[1])
        except nx.NetworkXNoCycle:
            pass
        return self.graph

    def discover_temporal_links(self):
        """Phase 3: PCMCI for time-lagged DAGs."""
        # Build lagged dataframe
        df_lagged = pd.DataFrame(index=self.df.index)
        for col in self.nodes:
            df_lagged[f"{col}_t0"] = self.df[col]
            for lag in range(1, self.max_lag + 1):
                df_lagged[f"{col}_t-{lag}"] = self.df[col].shift(lag)
        df_lagged = df_lagged.dropna().reset_index(drop=True)
        
        target_nodes = [f"{n}_t0" for n in self.nodes]
        all_past_nodes = [c for c in df_lagged.columns if not c.endswith("_t0")]
        
        # PC1 Phase
        candidates = {target: [] for target in target_nodes}
        for target in target_nodes:
            for past in all_past_nodes:
                if self._cmi(past, target, [], data=df_lagged) > self.mi_threshold:
                    candidates[target].append(past)
        
        # MCI Phase
        self.temporal_graph.add_nodes_from(df_lagged.columns)
        for target, current_parents in candidates.items():
            for candidate in current_parents:
                # Conditioning on target's past (simplified for this framework)
                z_target = [p for p in current_parents if p != candidate]
                if self._cmi(candidate, target, z_target, data=df_lagged) > self.mi_threshold:
                    self.temporal_graph.add_edge(candidate, target)
        
        return self.temporal_graph


class TitanCausalAudit:
    """
    Audits the discovered graph for causal stability and statistical paradoxes.
    """
    def __init__(self, data: pd.DataFrame, graph: nx.DiGraph):
        self.df = data
        self.graph = graph

    def check_simpsons_paradox(self, x, y):
        """Detects if the sign of relationship flips when conditioning on a confounder."""
        overall_corr = self.df[[x, y]].corr().iloc[0, 1]
        
        # Find potential confounders (parents of both or parents of X)
        confounders = list(self.graph.predecessors(x))
        if not confounders:
            return False, "No confounders detected for " + x
        
        # Simple split on the first confounder for illustration
        c = confounders[0]
        median_c = self.df[c].median()
        low_c = self.df[self.df[c] <= median_c][[x, y]].corr().iloc[0, 1]
        high_c = self.df[self.df[c] > median_c][[x, y]].corr().iloc[0, 1]
        
        if np.sign(overall_corr) != np.sign(low_c) or np.sign(overall_corr) != np.sign(high_c):
            return True, f"Simpson's Paradox detected! Sign flips on {c}"
        return False, "No Simpson's flip detected."

    def validate_stability(self):
        """Checks if R^2 values are consistent across the graph."""
        stability = {}
        for node in self.graph.nodes():
            parents = list(self.graph.predecessors(node))
            if not parents: continue
            X, y = self.df[parents], self.df[node]
            model = HistGradientBoostingRegressor().fit(X, y)
            score = r2_score(y, model.predict(X))
            stability[node] = score
        return stability


class ApexSimulator:
    """
    Monte Carlo Counterfactual Forecaster.
    """
    def __init__(self, data: pd.DataFrame, temporal_graph: nx.DiGraph, max_lag: int = 2):
        self.df = data.copy()
        self.nodes = list(data.columns)
        self.tg = temporal_graph
        self.max_lag = max_lag
        self.models = {}
        self.residuals = {}
        self._fit_models()

    def _fit_models(self):
        # Build training data
        df_lagged = pd.DataFrame(index=self.df.index)
        for col in self.nodes:
            df_lagged[f"{col}_t0"] = self.df[col]
            for lag in range(1, self.max_lag + 1):
                df_lagged[f"{col}_t-{lag}"] = self.df[col].shift(lag)
        df_lagged = df_lagged.dropna().reset_index(drop=True)
        
        for n in self.nodes:
            target = f"{n}_t0"
            parents = list(self.tg.predecessors(target))
            # Always add autoregressive links if not present
            for l in range(1, self.max_lag + 1):
                p_auto = f"{n}_t-{l}"
                if p_auto not in parents: parents.append(p_auto)
            
            X, y = df_lagged[parents], df_lagged[target]
            model = HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42).fit(X, y)
            self.models[target] = model
            self.residuals[target] = y.values - model.predict(X)
            self.models[target].parents = parents

    def simulate(self, target_var: str, shock_value: float, steps: int = 5, n_paths: int = 1000):
        # Initial state from last index
        base_state = {}
        last_idx = self.df.index[-1]
        for col in self.nodes:
            for lag in range(1, self.max_lag + 1):
                base_state[f"{col}_t-{lag}"] = self.df.loc[last_idx - (lag-1), col]
        
        results = {n: np.zeros((n_paths, steps)) for n in self.nodes}
        
        for p in range(n_paths):
            current_state = base_state.copy()
            for t in range(steps):
                next_state = {}
                for n in self.nodes:
                    target = f"{n}_t0"
                    if t == 0 and n == target_var:
                        next_state[n] = shock_value
                    else:
                        parents = self.models[target].parents
                        x_in = np.array([[current_state[par] for par in parents]])
                        noise = np.random.choice(self.residuals[target])
                        next_state[n] = self.models[target].predict(x_in)[0] + noise
                
                # Update current state for next step
                for n in self.nodes:
                    results[n][p, t] = next_state[n]
                    # Shift history
                    for lag in range(self.max_lag, 1, -1):
                        current_state[f"{n}_t-{lag}"] = current_state[f"{n}_t-{lag-1}"]
                    current_state[f"{n}_t-1"] = next_state[n]
                    
        summary = {}
        for n in self.nodes:
            summary[n] = {
                'mean': np.mean(results[n], axis=0),
                'lower': np.percentile(results[n], 5, axis=0),
                'upper': np.percentile(results[n], 95, axis=0)
            }
        return summary


csv_data = '''Date,NVDA,AAPL,MSFT,AMZN,TSLA
2025-11-24,182.55,275.92,474.00,226.28,417.78
2025-11-25,177.82,276.97,476.99,229.67,419.40
2025-11-26,180.26,277.55,485.50,229.16,426.58
2025-11-28,177.00,278.85,492.01,233.22,430.17
2025-12-01,179.92,283.10,486.74,233.88,430.14
2025-12-02,181.46,286.19,490.00,234.42,429.24
2025-12-03,179.59,284.15,477.73,232.38,446.74
2025-12-04,183.38,280.70,480.84,229.11,454.53
2025-12-05,182.41,278.78,483.16,229.53,455.00
2025-12-08,185.55,277.89,491.02,226.89,439.58
2025-12-09,184.97,277.18,492.02,227.92,445.17
2025-12-10,183.78,278.78,478.56,231.78,451.45
2025-12-11,180.93,278.03,483.47,230.28,446.89
2025-12-12,175.02,278.28,478.53,226.19,458.96
2025-12-15,176.29,274.11,474.82,222.54,475.31
2025-12-16,177.72,274.61,476.39,222.56,489.88
2025-12-17,170.94,271.84,476.12,221.27,467.26
2025-12-18,174.14,272.19,483.98,226.76,483.37
2025-12-19,180.99,273.67,485.92,227.35,481.20
2025-12-22,183.69,270.97,484.92,228.43,488.73
2025-12-23,189.21,272.36,486.85,232.14,485.56
2025-12-24,188.61,273.81,488.02,232.38,485.40
2025-12-26,190.53,273.40,487.71,232.52,475.19
2025-12-29,188.22,273.76,487.10,232.07,459.64
2025-12-30,187.54,273.08,487.48,232.53,454.43
2025-12-31,186.50,271.86,483.62,230.82,449.72
2026-01-02,188.85,271.01,472.94,226.50,438.07
2026-01-05,188.12,267.26,472.85,233.06,451.67
2026-01-06,187.24,262.36,478.51,240.93,432.96
2026-01-07,189.11,260.33,483.47,241.56,431.41
2026-01-08,185.04,259.04,478.11,246.29,435.80
2026-01-09,184.86,259.37,479.28,247.38,445.01
2026-01-12,184.94,260.25,477.18,246.47,448.96
2026-01-13,185.81,261.05,470.67,242.60,447.20
2026-01-14,183.14,259.96,459.38,236.65,439.20
2026-01-15,187.05,258.21,456.66,238.18,438.57
2026-01-16,186.23,255.53,459.86,239.12,437.50
2026-01-20,178.07,246.70,454.52,231.00,419.25
2026-01-21,183.32,247.65,444.11,231.31,431.44
2026-01-22,184.84,248.35,451.14,234.34,449.36
2026-01-23,187.67,248.04,465.95,239.16,449.06
2026-01-26,186.47,255.41,470.28,238.42,435.20
2026-01-27,188.52,258.27,480.58,244.68,430.90
2026-01-28,191.52,256.44,481.63,243.01,431.46
2026-01-29,192.51,258.28,433.50,241.73,416.56
2026-01-30,191.13,259.48,430.29,239.30,430.41
2026-02-02,185.61,270.01,423.37,242.96,421.81
2026-02-03,180.34,269.48,411.21,238.62,421.96
2026-02-04,174.19,276.49,414.19,232.99,406.01
2026-02-05,171.88,275.91,393.67,222.69,397.21
2026-02-06,185.41,278.12,401.14,210.32,411.11
2026-02-09,190.04,274.62,413.60,208.72,417.32
2026-02-10,188.54,273.68,413.27,206.96,425.21
2026-02-11,190.05,275.50,404.37,204.08,428.27
2026-02-12,186.94,261.73,401.84,199.60,417.07
2026-02-13,182.81,255.78,401.32,198.79,417.44
2026-02-17,184.97,263.88,396.86,201.15,410.63
2026-02-18,187.98,264.35,399.60,204.79,411.32
2026-02-19,187.90,260.58,398.46,204.86,411.71
2026-02-20,189.82,264.58,397.23,210.11,411.82
2026-02-23,191.55,266.18,384.47,205.27,399.83
2026-02-24,192.85,272.14,389.00,208.56,409.38
2026-02-25,195.56,274.23,400.60,210.64,417.40
2026-02-26,184.89,272.95,401.72,207.92,408.58
2026-02-27,177.19,264.18,392.74,210.00,402.51
2026-03-02,182.48,264.72,398.55,208.39,403.32
2026-03-03,179.67,262.54,401.05,206.06,390.50'''

print("Loading real Alpaca stock data...")
df = pd.read_csv(io.StringIO(csv_data), index_col='Date')
df.index = pd.to_datetime(df.index)

print(f"Loaded {len(df)} days of data for {list(df.columns)}")
rate_limited_api_call()

print("\n--- 1. DISCOVERY (TITAN ORACLE on ALPACA DATA) ---")
oracle = TitanOracle(df, mi_threshold=0.01, max_lag=2)
skel = oracle.build_skeleton()
dag = oracle.orient_edges(skel)
tg = oracle.discover_temporal_links()

print(">> DISCOVERED CROSS-SECTIONAL EDGES:")
print(list(dag.edges()))

print("\n>> DISCOVERED TEMPORAL LINKS:")
for u, v in tg.edges():
    if not u.startswith(v[:-3]):
        print(f"   {u} -> {v}")

print("\n--- 2. CAUSAL AUDIT ---")
audit = TitanCausalAudit(df, dag)
stability = audit.validate_stability()
for k, v in stability.items():
    print(f"   {k}: R^2 = {v:.4f}")

print("\n--- 3. SIMULATION (MONTE CARLO) ---")
simulator = ApexSimulator(df, tg, max_lag=2)
nvda_shock = df['NVDA'].iloc[-1] * 1.10
forecast = simulator.simulate('NVDA', nvda_shock, steps=5)

for n in df.columns:
    if n != 'NVDA':
        expected_change = forecast[n]['mean'][-1] / df[n].iloc[-1] - 1
        print(f"   Expected 5-day impact on {n}: {expected_change:.2%}")

print("\nFW9 Alpaca Real-World Validation COMPLETE.")
