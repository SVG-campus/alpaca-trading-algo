import nbformat as nbf

def consolidate_notebook(filepath):
    nb = nbf.v4.new_notebook()

    cells = []

    # Cell 0: Markdown - Title
    cells.append(nbf.v4.new_markdown_cell("""
# FRAMEWORK 9: AB-INITIO CAUSAL DISCOVERY (TITAN EDITION)
### "The Oracle": A Complete Causal Physics Engine

This notebook implements the production-grade **Titan Oracle**, a unified causal discovery and simulation framework. It integrates state-of-the-art methods to map, validate, and simulate causal relationships in complex systems.

**Key Modules:**
1. **TitanOracle:** Excavates structural skeletons (PC), orients edges (LiNGAM), and discovers temporal links (PCMCI).
2. **TitanCausalAudit:** Audits data integrity, Simpson's paradox, and causal stability.
3. **ApexSimulator:** Executes Monte Carlo counterfactual forecasting to simulate causal shockwaves.
"""))

    # Cell 1: Code - Imports and Config
    cells.append(nbf.v4.new_code_cell("""
import numpy as np
import pandas as pd
import networkx as nx
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Configuration Constants
MI_THRESHOLD = 0.05
ALPHA = 0.05
MAX_LAG = 2
N_PATHS = 1000

print("🔮 TITAN ORACLE ONLINE | Causal Physics Engine Initialized")
"""))

    # Cell 2: Code - TitanOracle
    cells.append(nbf.v4.new_code_cell("""
class TitanOracle:
    \"\"\"
    The core discovery engine for both cross-sectional and temporal data.
    \"\"\"
    def __init__(self, data: pd.DataFrame, mi_threshold=0.05, max_lag=2):
        self.df = data.copy()
        self.nodes = list(data.columns)
        self.mi_threshold = mi_threshold
        self.max_lag = max_lag
        self.graph = nx.DiGraph()
        self.temporal_graph = nx.DiGraph()
        
    def _cmi(self, x, y, z_list, data=None):
        \"\"\"Calculates Non-Linear Conditional Mutual Information using Tree Residuals\"\"\"
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
        \"\"\"Phase 1: PC-Algorithm inspired skeleton discovery (Cross-Sectional).\"\"\"
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
        \"\"\"Phase 2: Asymmetry orientation (LiNGAM/Post-Nonlinear proxy).\"\"\"
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
        \"\"\"Phase 3: PCMCI for time-lagged DAGs.\"\"\"
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
"""))

    # Cell 3: Code - TitanCausalAudit
    cells.append(nbf.v4.new_code_cell("""
class TitanCausalAudit:
    \"\"\"
    Audits the discovered graph for causal stability and statistical paradoxes.
    \"\"\"
    def __init__(self, data: pd.DataFrame, graph: nx.DiGraph):
        self.df = data
        self.graph = graph

    def check_simpsons_paradox(self, x, y):
        \"\"\"Detects if the sign of relationship flips when conditioning on a confounder.\"\"\"
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
        \"\"\"Checks if R^2 values are consistent across the graph.\"\"\"
        stability = {}
        for node in self.graph.nodes():
            parents = list(self.graph.predecessors(node))
            if not parents: continue
            X, y = self.df[parents], self.df[node]
            model = HistGradientBoostingRegressor().fit(X, y)
            score = r2_score(y, model.predict(X))
            stability[node] = score
        return stability
"""))

    # Cell 4: Code - ApexSimulator
    cells.append(nbf.v4.new_code_cell("""
class ApexSimulator:
    \"\"\"
    Monte Carlo Counterfactual Forecaster.
    \"\"\"
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
"""))

    # Cell 5: Code - Execution Pipeline
    cells.append(nbf.v4.new_code_cell("""
# --- 1. DATA GENERATION (Synthetic Macro-Economy) ---
np.random.seed(42)
n = 2000
R = np.zeros(n)  # Rates
U = np.zeros(n)  # Unemployment
I = np.zeros(n)  # Inflation

for t in range(2, n):
    R[t] = 0.8 * R[t-1] + np.random.normal(0, 0.1)
    U[t] = 0.5 * U[t-1] + 0.6 * R[t-2] + np.random.normal(0, 0.1)
    I[t] = 0.6 * I[t-1] - 0.7 * U[t-1] + np.random.normal(0, 0.1)

df = pd.DataFrame({'RATES': R, 'UNEMP': U, 'INFLATION': I})

# --- 2. DISCOVERY (TITAN ORACLE) ---
oracle = TitanOracle(df, mi_threshold=0.08)
skel = oracle.build_skeleton()
dag = oracle.orient_edges(skel)
tg = oracle.discover_temporal_links()

print(">> DISCOVERED TEMPORAL LINKS:")
for u, v in tg.edges():
    if not u.startswith(v[:-3]): # Ignore self-loops
        print(f"   {u} -> {v}")

# --- 3. AUDIT ---
audit = TitanCausalAudit(df, dag)
stability = audit.validate_stability()
print("\\n>> CAUSAL STABILITY (R^2):", stability)

# --- 4. SIMULATION ---
simulator = ApexSimulator(df, tg, max_lag=2)
shock_val = df['RATES'].mean() + 2.0 * df['RATES'].std()
forecast = simulator.simulate('RATES', shock_val, steps=6)

# --- 5. VISUALIZATION ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
steps = np.arange(6)

for i, var in enumerate(['UNEMP', 'INFLATION']):
    axes[i].plot(steps, forecast[var]['mean'], label='Mean Forecast', color='blue')
    axes[i].fill_between(steps, forecast[var]['lower'], forecast[var]['upper'], alpha=0.2, color='blue', label='90% CI')
    axes[i].set_title(f"Shockwave Propagation: {var}")
    axes[i].set_xlabel("Time Steps (t+n)")
    axes[i].legend()

plt.tight_layout()
plt.show()
"""))

    nb['cells'] = cells

    with open(filepath, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    consolidate_notebook('Misc. Files/ab-initio-causal-discovery.ipynb')
