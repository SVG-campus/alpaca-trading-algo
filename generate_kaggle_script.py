import json
import io

with open('Misc. Files/fw9_test_data.csv', 'r') as f:
    csv_data = f.read()

with open('Misc. Files/ab-initio-causal-discovery.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [cell['source'] for cell in nb['cells'] if cell['cell_type'] == 'code']
imports = ''.join(code_cells[0]).replace('import matplotlib.pyplot as plt', '').replace('import seaborn as sns', '')
classes = ''.join(code_cells[1]) + '\n' + ''.join(code_cells[2]) + '\n' + ''.join(code_cells[3])

kaggle_script = f"""{imports}
import io
import time

def rate_limited_api_call():
    print("Simulating 5 minute API delay (mocked to 1 sec for testing)...")
    time.sleep(1)

{classes}

csv_data = '''{csv_data}'''

print("Loading real Alpaca stock data...")
df = pd.read_csv(io.StringIO(csv_data), index_col='Date')
df.index = pd.to_datetime(df.index)

print(f"Loaded {{len(df)}} days of data for {{list(df.columns)}}")
rate_limited_api_call()

print("\\n--- 1. DISCOVERY (TITAN ORACLE on ALPACA DATA) ---")
oracle = TitanOracle(df, mi_threshold=0.01, max_lag=2)
skel = oracle.build_skeleton()
dag = oracle.orient_edges(skel)
tg = oracle.discover_temporal_links()

print(">> DISCOVERED CROSS-SECTIONAL EDGES:")
print(list(dag.edges()))

print("\\n>> DISCOVERED TEMPORAL LINKS:")
for u, v in tg.edges():
    if not u.startswith(v[:-3]):
        print(f"   {{u}} -> {{v}}")

print("\\n--- 2. CAUSAL AUDIT ---")
audit = TitanCausalAudit(df, dag)
stability = audit.validate_stability()
for k, v in stability.items():
    print(f"   {{k}}: R^2 = {{v:.4f}}")

print("\\n--- 3. SIMULATION (MONTE CARLO) ---")
simulator = ApexSimulator(df, tg, max_lag=2)
nvda_shock = df['NVDA'].iloc[-1] * 1.10
forecast = simulator.simulate('NVDA', nvda_shock, steps=5)

for n in df.columns:
    if n != 'NVDA':
        expected_change = forecast[n]['mean'][-1] / df[n].iloc[-1] - 1
        print(f"   Expected 5-day impact on {{n}}: {{expected_change:.2%}}")

print("\\nFW9 Alpaca Real-World Validation COMPLETE.")
"""

with open('kaggle_fw9.py', 'w', encoding='utf-8') as f:
    f.write(kaggle_script)
