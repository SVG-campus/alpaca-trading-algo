import os
import json
from pathlib import Path

USERNAME = os.environ.get('KAGGLE_USERNAME', 'seasthaalores')

intraday_meta = {
  "id": f"{USERNAME}/alpaca-intraday-v8",
  "title": "Alpaca Intraday V8 Singularity",
  "code_file": "run_discovery.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}

swing_meta = dict(intraday_meta)
swing_meta["id"] = f"{USERNAME}/alpaca-swing-v7"
swing_meta["title"] = "Alpaca Swing V7 Singularity"

with open('kaggle_jobs/intraday/kernel-metadata.json', 'w') as f:
    json.dump(intraday_meta, f, indent=2)
    
with open('kaggle_jobs/swing/kernel-metadata.json', 'w') as f:
    json.dump(swing_meta, f, indent=2)
    
print("Kaggle metadata files created.")