import json
import pandas as pd
import numpy as np

# Extract code from notebook
with open('Misc. Files/ab-initio-causal-discovery.ipynb', 'r', encoding='utf-8') as nb_f:
    nb = json.load(nb_f)

code_cells = [cell['source'] for cell in nb['cells'] if cell['cell_type'] == 'code']
# Execute the definition cells
exec(''.join(code_cells[0]))
exec(''.join(code_cells[1]))
exec(''.join(code_cells[2]))
exec(''.join(code_cells[3]))

print("Loading real Alpaca stock data (fw9_test_data.csv)...")
df = pd.read_csv('Misc. Files/fw9_test_data.csv', index_col='Date')
df.index = pd.to_datetime(df.index)

print(f"Loaded {len(df)} days of data for {list(df.columns)}")

print("\n--- 1. DISCOVERY (TITAN ORACLE on ALPACA DATA) ---")
oracle = TitanOracle(df, mi_threshold=0.01, max_lag=2)
skel = oracle.build_skeleton()
dag = oracle.orient_edges(skel)
tg = oracle.discover_temporal_links()

print(">> DISCOVERED CROSS-SECTIONAL EDGES:")
print(list(dag.edges()))

print(">> DISCOVERED TEMPORAL LINKS:")
for u, v in tg.edges():
    if not u.startswith(v[:-3]):
        print(f"   {u} -> {v}")

print("\n--- 2. CAUSAL AUDIT ---")
audit = TitanCausalAudit(df, dag)
stability = audit.validate_stability()
print(">> CAUSAL STABILITY (R^2):", stability)

print("\n--- 3. SIMULATION (MONTE CARLO) ---")
simulator = ApexSimulator(df, tg, max_lag=2)
# Simulate a 10% shock to NVDA
nvda_shock = df['NVDA'].iloc[-1] * 1.10
forecast = simulator.simulate('NVDA', nvda_shock, steps=5)

for n in df.columns:
    if n != 'NVDA':
        expected_change = forecast[n]['mean'][-1] / df[n].iloc[-1] - 1
        print(f"   Expected 5-day impact on {n}: {expected_change:.2%}")

print("\nFW9 Alpaca Real-World Validation COMPLETE.")
