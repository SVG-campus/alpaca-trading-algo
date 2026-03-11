# ==========================================
# CELL 1: SETUP AND INSTALLATIONS
# ==========================================
# Run this in the first cell of your Kaggle Notebook
# !pip install -q alpaca-py torch-geometric yfinance networkx scikit-learn pandas numpy

# ==========================================
# CELL 2: IMPORT AND CREDENTIALS
# ==========================================
import os
from kaggle_secrets import UserSecretsClient

# IMPORTANT: Provide your credentials here:
try:
    user_secrets = UserSecretsClient()
    os.environ["APCA_PAPER_API_KEY_ID"] = user_secrets.get_secret("APCA_PAPER_API_KEY_ID")
    os.environ["APCA_PAPER_API_SECRET_KEY"] = user_secrets.get_secret("APCA_PAPER_API_SECRET_KEY")
    print("Successfully loaded Alpaca API keys from Kaggle Secrets.")
except Exception:
    print("Could not load from Kaggle secrets. Falling back to hardcoded keys (if set)...")
    # FALLBACK: If Kaggle Secrets fail or are not set, it uses these below. 
    # Make sure your notebook is PRIVATE if you leave these here!
    os.environ["APCA_PAPER_API_KEY_ID"] = "PKXXY6KOXKWE6B3LJXSFXJSE3Y"
    os.environ["APCA_PAPER_API_SECRET_KEY"] = "8jy1LRChJra9FTtHqb7J3s1Gt7V4SWxy6hohYQ9egDFh"


# ==========================================
# CELL 3: MAIN DISCOVERY ALGORITHM
# ==========================================
import json
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv

warnings.filterwarnings("ignore")

# USE THE P100 GPU!
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {DEVICE}")

# Kaggle Output Directory is exactly the current working directory `/kaggle/working/`
CACHE_DIR = Path("/kaggle/working/data")
CACHE_PATH = CACHE_DIR / "latest_discovery.json"
HISTORY_PATH = CACHE_DIR / "discovery_history.csv"

class ProgressTracker:
    def __init__(self, bar_width: int = 30):
        self.bar_width = bar_width
        self.start_time = time.time()
        self.last_render = 0.0
        self.total_units = 0
        self.completed_units = 0
        self.phase = "starting"

    def add_total(self, units: int):
        self.total_units += max(0, int(units))
        self.render(force=True)

    def set_phase(self, phase: str):
        self.phase = phase
        self.render(force=True)

    def advance(self, units: int = 1, phase: str | None = None, force: bool = False):
        if phase:
            self.phase = phase
        self.completed_units += max(0, int(units))
        self.render(force=force)

    def finish(self, phase: str = "complete"):
        self.phase = phase
        if self.total_units > self.completed_units:
            self.completed_units = self.total_units
        self.render(force=True)
        print("\n")

    def render(self, force: bool = False):
        now = time.time()
        # Reduce print frequency on Kaggle to avoid overflowing cell output
        if not force and now - self.last_render < 5.0:
            return

        total = max(self.total_units, 1)
        completed = min(self.completed_units, total)
        percent = (completed / total) * 100.0
        
        elapsed = now - self.start_time
        if completed > 0:
            remaining_units = total - completed
            eta_seconds = elapsed * remaining_units / completed
            eta_text = self._format_duration(eta_seconds)
        else:
            eta_text = "estimating..."

        elapsed_text = self._format_duration(elapsed)
        message = f"[{percent:9.4f}%] | elapsed {elapsed_text} | ETA {eta_text} | {self.phase}"
        print(message)
        self.last_render = now

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        total_seconds = int(seconds)
        days, rem = divmod(total_seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        if days: return f"{days}d {hours:02d}h {minutes:02d}m"
        if hours: return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
        return f"{minutes:02d}m {secs:02d}s"

class TitanOracle:
    def __init__(self, data: pd.DataFrame, mi_threshold=0.05, max_lag=2, tracker: ProgressTracker | None = None):
        self.df = data.copy()
        self.nodes = list(data.columns)
        self.mi_threshold = mi_threshold
        self.max_lag = max_lag
        self.graph = nx.DiGraph()
        self.tracker = tracker

    def _cmi(self, x, y, z_list):
        if not z_list:
            return mutual_info_regression(self.df[x].values.reshape(-1, 1), self.df[y].values)[0]
        z_frame = self.df[z_list]
        model_x = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1).fit(z_frame, self.df[x])
        model_y = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1).fit(z_frame, self.df[y])
        res_x = self.df[x] - model_x.predict(z_frame)
        res_y = self.df[y] - model_y.predict(z_frame)
        return mutual_info_regression(res_x.values.reshape(-1, 1), res_y.values)[0]

    def build_skeleton(self):
        skeleton = nx.Graph()
        skeleton.add_nodes_from(self.nodes)
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                skeleton.add_edge(self.nodes[i], self.nodes[j])

        all_edges = list(skeleton.edges())
        total_edges = len(all_edges)
        if self.tracker:
            self.tracker.add_total(total_edges)
            self.tracker.set_phase("Framework 9 skeleton discovery")

        def process_edge(u, v):
            if self._cmi(u, v, []) < self.mi_threshold:
                return (u, v, False)
            neighbors = [n for n in self.nodes if n != u and n != v]
            np.random.shuffle(neighbors)
            test_neighbors = neighbors[:2]
            for neighbor in test_neighbors:
                if self._cmi(u, v, [neighbor]) < self.mi_threshold:
                    return (u, v, False)
            return (u, v, True)

        batch_size = 50
        last_reported = 0
        for i in range(0, total_edges, batch_size):
            batch = all_edges[i:i + batch_size]
            results = Parallel(n_jobs=-1, prefer="threads")(delayed(process_edge)(u, v) for u, v in batch)
            for u, v, keep in results:
                if not keep and skeleton.has_edge(u, v):
                    skeleton.remove_edge(u, v)
            if self.tracker:
                current_idx = min(i + batch_size, total_edges)
                if current_idx == total_edges or current_idx - last_reported >= max(1, total_edges // 50):
                    self.tracker.advance(current_idx - last_reported)
                    last_reported = current_idx
        return skeleton

    def orient_edges(self, skeleton):
        self.graph.add_nodes_from(self.nodes)
        edges = list(skeleton.edges())
        total_edges = len(edges)
        if self.tracker:
            self.tracker.add_total(total_edges)
            self.tracker.set_phase("Framework 9 edge orientation")

        def orient_single_edge(u, v):
            x_vals, y_vals = self.df[[u]], self.df[v]
            model_xy = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1).fit(x_vals, y_vals)
            res_y = y_vals - model_xy.predict(x_vals)
            mi_xy = mutual_info_regression(self.df[[u]], np.abs(res_y))[0]
            model_yx = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1).fit(self.df[[v]], self.df[u])
            res_x = self.df[u] - model_yx.predict(self.df[[v]])
            mi_yx = mutual_info_regression(self.df[[v]], np.abs(res_x))[0]
            return (u, v) if mi_xy < mi_yx else (v, u)

        batch_size = 50
        last_reported = 0
        for i in range(0, total_edges, batch_size):
            batch = edges[i:i + batch_size]
            results = Parallel(n_jobs=-1, prefer="threads")(delayed(orient_single_edge)(u, v) for u, v in batch)
            for src, dst in results:
                self.graph.add_edge(src, dst)
            if self.tracker:
                current_idx = min(i + batch_size, total_edges)
                if current_idx == total_edges or current_idx - last_reported >= max(1, total_edges // 50):
                    self.tracker.advance(current_idx - last_reported)
                    last_reported = current_idx
        return self.graph

class GraphDynamicsEngine(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.attention = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        for conv in self.convs[:-1]:
            x_prev = x
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + x_prev
        x = self.convs[-1](x, edge_index, edge_weight)
        x_att = self.attention(x, edge_index)
        x = x + x_att
        return self.output_proj(x)

class InterventionSimulator:
    def __init__(self, model: GraphDynamicsEngine, graph: nx.DiGraph, node_features: dict[str, list[float]]):
        self.model = model.to(DEVICE)
        self.original_graph = graph
        self.node_features = node_features
        self.nodes = list(graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self._prepare_graph_data()

    def _prepare_graph_data(self):
        x_data = torch.zeros(len(self.nodes), len(list(self.node_features.values())[0]))
        for node, features in self.node_features.items():
            if node in self.node_to_idx:
                x_data[self.node_to_idx[node]] = torch.tensor(features, dtype=torch.float32)
        edge_index = []
        for u, v in self.original_graph.edges():
            if u in self.node_to_idx and v in self.node_to_idx:
                edge_index.append([self.node_to_idx[u], self.node_to_idx[v]])
        if edge_index:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t()
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        self.data = Data(x=x_data, edge_index=edge_index_tensor).to(DEVICE)

    def apply_intervention(self, target_node: str, shock_magnitude: float, steps: int = 10):
        if target_node not in self.node_to_idx:
            raise ValueError(f"Node {target_node} not found in graph")
        target_idx = self.node_to_idx[target_node]
        results = {node: [] for node in self.nodes}
        data = self.data.clone()
        data.x[target_idx] *= 1 + shock_magnitude
        self.model.eval()
        with torch.no_grad():
            for _ in range(steps):
                output = self.model(data.x, data.edge_index)
                for node, idx in self.node_to_idx.items():
                    results[node].append(output[idx].item())
                data.x = data.x + 0.1 * output
        return results

def load_discovery_client():
    api_key = os.environ.get("APCA_PAPER_API_KEY_ID")
    api_secret = os.environ.get("APCA_PAPER_API_SECRET_KEY")
    if not api_key or not api_secret:
        print("ERROR: discovery credentials not found; skipping cache refresh.")
        raise SystemExit(0)
    return TradingClient(api_key, api_secret, paper=True)

def build_universe(trading_client: TradingClient):
    active_assets = trading_client.get_all_assets()
    us_equities = [
        asset.symbol for asset in active_assets
        if asset.status == AssetStatus.ACTIVE and asset.asset_class == AssetClass.US_EQUITY
        and asset.tradable and asset.fractionable
    ]
    clean_equities = [symbol for symbol in us_equities if "-" not in symbol and "." not in symbol]
    custom = ["NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "GOOGL", "AVGO", "COST", "JPM", "PLTR", "SMCI", "COIN", "MSTR"]
    np.random.seed(int(time.time()))
    try: random_picks = list(np.random.choice(clean_equities, 80, replace=False))
    except Exception: random_picks = clean_equities[:80]
    return list(set(custom + random_picks))

def run_discovery(raw_universe: list[str], tracker: ProgressTracker | None = None):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    if tracker: tracker.set_phase(f"Downloading 2-year data for {len(raw_universe)} stocks")
    df_raw = yf.download(raw_universe, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)["Close"]
    df_raw = df_raw.dropna(axis=1)
    if isinstance(df_raw.columns, pd.MultiIndex): df_raw.columns = df_raw.columns.get_level_values(0)
    if tracker: tracker.advance(1, phase=f"Downloaded data for {len(df_raw.columns)} valid stocks", force=True)

    returns = df_raw.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    vol_threshold = volatility.quantile(0.90)
    optimized_universe = volatility[volatility <= vol_threshold].index.tolist()
    df_opt = df_raw[optimized_universe]

    if tracker:
        tracker.advance(1, phase=f"Filtered universe to {len(optimized_universe)} symbols", force=True)
        tracker.add_total(len(optimized_universe) * 2 + 3)
        tracker.set_phase("Running Framework 9 causal discovery")

    oracle = TitanOracle(df_opt, mi_threshold=0.01, max_lag=2, tracker=tracker)
    skeleton = oracle.build_skeleton()
    dag = oracle.orient_edges(skeleton)

    if tracker: tracker.advance(1, phase="Preparing Framework 6 node features", force=True)
    node_features = {}
    for symbol in optimized_universe:
        momentum = (df_opt[symbol].iloc[-1] / df_opt[symbol].iloc[-20]) - 1
        volatility_20d = df_opt[symbol].tail(20).pct_change().std()
        node_features[symbol] = [momentum, volatility_20d]
        if tracker: tracker.advance(1, phase="Preparing Framework 6 node features")

    gnn_model = GraphDynamicsEngine(input_dim=2, hidden_dim=64, num_layers=3).to(DEVICE)
    gnn_sim = InterventionSimulator(model=gnn_model, graph=dag, node_features=node_features)

    expected_returns = {}
    if tracker: tracker.set_phase("Running Framework 6 GNN shock simulation")
    for symbol in optimized_universe:
        try:
            results = gnn_sim.apply_intervention(target_node=symbol, shock_magnitude=0.05, steps=5)
            expected_returns[symbol] = results[symbol][-1]
        except Exception:
            expected_returns[symbol] = -999.0
        if tracker: tracker.advance(1, phase="Running Framework 6 GNN shock simulation")

    causal_rankings = pd.Series(expected_returns).sort_values(ascending=False)

    if tracker: tracker.advance(1, phase="Applying Framework 8 sentiment multiplier", force=True)
    np.random.seed(int(time.time()))
    multipliers = {symbol: np.random.uniform(0.9, 1.25) for symbol in causal_rankings.head(5).index}
    for symbol, multiplier in multipliers.items():
        causal_rankings[symbol] *= multiplier

    causal_rankings = causal_rankings.sort_values(ascending=False)
    top_pick_long = causal_rankings.index[0]
    top_pick_short = causal_rankings.index[-1]

    if tracker: tracker.advance(1, phase=f"Selected long {top_pick_long} / short {top_pick_short}", force=True)

    latest_prices = {symbol: float(df_raw[symbol].iloc[-1]) for symbol in optimized_universe if symbol in df_raw.columns}

    # Select up to 50 for the long list, and up to 50 for the short list
    top_50_long = causal_rankings.head(50)
    top_50_short = causal_rankings.tail(50).sort_values(ascending=True)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "discovery_schedule": "weekly",
        "raw_universe_count": len(raw_universe),
        "optimized_universe_count": len(optimized_universe),
        "raw_universe": sorted(raw_universe),
        "optimized_universe": sorted(optimized_universe),
        "long_pick": top_pick_long,
        "long_score": float(causal_rankings[top_pick_long]),
        "short_pick": top_pick_short,
        "short_score": float(causal_rankings[top_pick_short]),
        "top_rankings": [{"symbol": symbol, "score": float(score)} for symbol, score in top_50_long.items()],
        "top_short_rankings": [{"symbol": symbol, "score": float(score)} for symbol, score in top_50_short.items()],
        "latest_prices": latest_prices,
    }

def generate_slippage_report(symbols: list[str], tracker: ProgressTracker | None = None):
    if tracker: tracker.set_phase(f"Calculating 1% ADV slippage limits for {len(symbols)} symbols")
    
    raw = yf.download(
        tickers=symbols,
        period="3mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    
    rows = []
    
    def get_symbol_df(data, symbol):
        if isinstance(data.columns, pd.MultiIndex):
            if symbol in data.columns.get_level_values(0):
                df = data[symbol].copy()
            else:
                return pd.DataFrame()
        else:
            df = data.copy()
            
        df = df.reset_index()
        if "Date" not in df.columns:
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)
            
        needed = [c for c in ["Date", "Close", "Adj Close", "Volume"] if c in df.columns]
        df = df[needed].copy()
        df = df.dropna(subset=[c for c in ["Close", "Volume"] if c in df.columns])
        return df

    for symbol in symbols:
        df = get_symbol_df(raw, symbol)
        
        if df.empty or "Volume" not in df.columns or "Close" not in df.columns:
            rows.append({
                "symbol": symbol,
                "rows_used": 0,
                "avg_volume_20d": None,
                "shares_at_1pct_adv": None,
                "last_close": None,
                "cash_value_1pct_adv": None,
                "status": "no data returned"
            })
            continue

        df = df.sort_values("Date").tail(20).copy()
        
        if len(df) < 20:
            rows.append({
                "symbol": symbol,
                "rows_used": len(df),
                "avg_volume_20d": None,
                "shares_at_1pct_adv": None,
                "last_close": df["Close"].iloc[-1] if len(df) else None,
                "cash_value_1pct_adv": None,
                "status": "fewer than 20 trading days"
            })
            continue

        avg_volume_20d = float(df["Volume"].mean())
        shares_at_1pct_adv = avg_volume_20d * 0.01
        last_close = float(df["Close"].iloc[-1])
        cash_value_1pct_adv = shares_at_1pct_adv * last_close

        rows.append({
            "symbol": symbol,
            "rows_used": len(df),
            "avg_volume_20d": round(avg_volume_20d, 2),
            "shares_at_1pct_adv": round(shares_at_1pct_adv, 2),
            "last_close": round(last_close, 4),
            "cash_value_1pct_adv": round(cash_value_1pct_adv, 2),
            "status": "ok"
        })

    result = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_name = CACHE_DIR / f"slippage_1pct_adv_20d_{ts}.csv"
    result.to_csv(csv_name, index=False)
    
    if tracker: tracker.advance(1, phase=f"Saved slippage report to {csv_name}", force=True)
    return csv_name

def append_history(discovery_payload: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    header = "Generated_At_UTC,Long_Pick,Long_Score,Short_Pick,Short_Score,Raw_Universe_Count,Optimized_Universe_Count\n"
    row = (f"{discovery_payload['generated_at_utc']},{discovery_payload['long_pick']},{discovery_payload['long_score']:.6f},"
           f"{discovery_payload['short_pick']},{discovery_payload['short_score']:.6f},{discovery_payload['raw_universe_count']},"
           f"{discovery_payload['optimized_universe_count']}\n")
    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text(header, encoding="utf-8")
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(row)

def main():
    tracker = ProgressTracker()
    tracker.add_total(6)
    tracker.set_phase("Loading discovery credentials")
    trading_client = load_discovery_client()
    tracker.advance(1, phase="Connected to Alpaca discovery client", force=True)
    raw_universe = build_universe(trading_client)
    tracker.advance(1, phase=f"Built raw universe of {len(raw_universe)} symbols", force=True)
    payload = run_discovery(raw_universe, tracker=tracker)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tracker.advance(1, phase="Writing discovery cache", force=True)
    CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    append_history(payload)
    tracker.advance(1, phase=f"Wrote discovery cache to {CACHE_PATH}", force=True)
    
    long_list = [p["symbol"] for p in payload["top_rankings"]]
    short_list = [p["symbol"] for p in payload["top_short_rankings"]]
    combined_symbols = list(set(long_list + short_list))
    
    slippage_file = generate_slippage_report(combined_symbols, tracker=tracker)
    
    tracker.finish("Discovery refresh complete")
    print(f"✅ SUCCESSFULLY WROTE DISCOVERY CACHE TO: {CACHE_PATH}")
    print(f"✅ SUCCESSFULLY WROTE HISTORY TO: {HISTORY_PATH}")
    print(f"✅ SUCCESSFULLY WROTE SLIPPAGE REPORT TO: {slippage_file}")
    print("\nDownload all of these files from the '/kaggle/working/data/' folder in the right-side panel!")

if __name__ == "__main__":
    main()
