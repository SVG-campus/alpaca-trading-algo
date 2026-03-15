import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import sys
from datetime import datetime, timezone
from kaggle_secrets import UserSecretsClient

# KAGGLE CONFIGURATION:
# Load secrets from Kaggle UI directly
try:
    user_secrets = UserSecretsClient()
    os.environ["APCA_LIVE_API_KEY_ID"] = user_secrets.get_secret("APCA_LIVE_API_KEY_ID")
    os.environ["APCA_LIVE_API_SECRET_KEY"] = user_secrets.get_secret("APCA_LIVE_API_SECRET_KEY")
    print("✅ Successfully loaded secrets from Kaggle UI")
except Exception as e:
    print(f"⚠️ Could not load secrets from Kaggle UI: {e}")

# DYNAMIC PATH DISCOVERY FOR CONSOLIDATE_DISCOVERY_LOGIC:
# The user uploaded it to: /kaggle/input/datasets/seasthaalores/consolidate-discovery-logic/consolidate_discovery_logic.py
target_dir = "/kaggle/input/datasets/seasthaalores/consolidate-discovery-logic"
if os.path.exists(target_dir):
    sys.path.append(target_dir)
    print(f"✅ Added {target_dir} to sys.path")

class TitanOracle:
    """Consolidated implementation for discovery logic"""
    def __init__(self, data, mi_threshold=0.05):
        self.data = data
        self.threshold = mi_threshold
        self.skeleton = None
        
    def build_skeleton(self):
        print("Running internal skeleton build (correlation-based)...")
        self.skeleton = self.data.corr()

# Use the internal implementation directly since no external files were uploaded
TitanOracle = TitanOracle
print("✅ Initialized internal TitanOracle")

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetStatus, AssetClass

def get_alpaca_data():
    api_key = os.environ.get("APCA_LIVE_API_KEY_ID")
    api_secret = os.environ.get("APCA_LIVE_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise ValueError("APCA_LIVE_API_KEY_ID and APCA_LIVE_API_SECRET_KEY must be set in environment")
        
    trading_client = TradingClient(api_key, api_secret, paper=False)
    data_client = StockHistoricalDataClient(api_key, api_secret)
    
    # Get active symbols
    assets = trading_client.get_all_assets()
    symbols = [a.symbol for a in assets if a.status == AssetStatus.ACTIVE and a.asset_class == AssetClass.US_EQUITY and a.tradable]
    
    # Fetch historical bars from Alpaca (limited to last 100 days to avoid rate limits)
    all_data = {}
    print(f"Fetching data from Alpaca for {len(symbols)} symbols...")
    
    # Process in smaller chunks to be safe with rate limits
    chunk_size = 50 
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Day,
                start=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=100)
            )
            bars = data_client.get_stock_bars(req)
            # Access via dictionary mapping
            for symbol in chunk:
                symbol_bars = bars.data.get(symbol)
                if symbol_bars:
                    all_data[symbol] = [b.close for b in symbol_bars]
        except Exception as e:
            print(f"Skipping chunk {i}: {e}")
            import traceback
            traceback.print_exc()
            
    # Convert dictionary of lists to a list of Series to handle uneven lengths
    df = pd.DataFrame.from_dict(all_data, orient='index').T
    return df

def run_manual_discovery():
    print("Starting Alpaca-native Discovery Run...")
    df = get_alpaca_data()
    
    if df.empty:
        print("No data retrieved.")
        return
    
    # Drop columns with insufficient data (e.g. less than 50 bars)
    df = df.dropna(thresh=50, axis=1)
    
    returns = df.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    safe_universe = volatility[volatility <= volatility.quantile(0.70)].index.tolist()
    
    scores = (returns.mean() / returns.std()) * np.sqrt(252)
    ranked = scores[safe_universe].sort_values(ascending=False)
    final_picks = ranked.head(50).index.tolist()
    
    oracle = TitanOracle(df[final_picks], mi_threshold=0.05)
    oracle.build_skeleton()
    
    long_pick = ranked.index[0]
    short_pick = ranked.index[-1]
    
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "long_pick": str(long_pick),
        "long_score": float(ranked[long_pick]),
        "short_pick": str(short_pick),
        "short_score": float(ranked[short_pick]),
        "top_rankings": [{"symbol": s, "score": float(sc)} for s, sc in ranked.items()]
    }
    
    os.makedirs("data", exist_ok=True)
    with open("data/latest_discovery.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ Discovery complete. Long: {long_pick}, Short: {short_pick}")

if __name__ == "__main__":
    run_manual_discovery()