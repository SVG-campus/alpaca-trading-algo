import os
import time
import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mutual_info_score
from alpaca.trading.client import TradingClient

warnings.filterwarnings("ignore")

class TitanOracle:
    """
    Consolidated discovery logic (Framework 9) 
    Designed for Kaggle/Cloud environments.
    """
    def __init__(self, data: pd.DataFrame, mi_threshold=0.01):
        self.df = data.copy()
        self.nodes = list(data.columns)
        self.mi_threshold = mi_threshold

    def _cmi(self, x, y, z_list):
        # Implementation of conditional mutual information using random forest residuals
        # This is the core causal discovery logic.
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import mutual_info_regression
        
        if not z_list:
            return mutual_info_regression(self.df[x].values.reshape(-1, 1), self.df[y].values)[0]
        z_frame = self.df[z_list]
        model_x = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1).fit(z_frame, self.df[x])
        model_y = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1).fit(z_frame, self.df[y])
        res_x = self.df[x] - model_x.predict(z_frame)
        res_y = self.df[y] - model_y.predict(z_frame)
        return mutual_info_regression(res_x.values.reshape(-1, 1), res_y.values)[0]

    def run_discovery(self):
        # Placeholder for full DAG discovery loop
        # This will be the main entry point for signal detection
        print("Running discovery on provided dataset...")
        return {"status": "success", "message": "Discovery complete"}

def get_alpaca_universe(api_key, api_secret, paper=True):
    client = TradingClient(api_key, api_secret, paper=paper)
    assets = client.get_all_assets()
    return [a.symbol for a in assets if a.status == 'active' and a.tradable]