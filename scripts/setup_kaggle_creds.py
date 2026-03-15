import os
import json
from pathlib import Path

def setup_kaggle_credentials():
    # Read .env
    env_vars = {}
    with open('.env', 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.strip().split('=', 1)
                env_vars[key.strip()] = val.strip().strip('"').strip("'")
    
    username = env_vars.get('KAGGLE_USERNAME')
    key = env_vars.get('KAGGLE_KEY')
    
    if not username or not key:
        print("Missing KAGGLE_USERNAME or KAGGLE_KEY in .env")
        return False
        
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    creds_path = kaggle_dir / 'kaggle.json'
    creds = {"username": username, "key": key}
    
    with open(creds_path, 'w') as f:
        json.dump(creds, f)
        
    # Set permissions on Linux/Mac, Windows doesn't matter as much but good practice
    try:
        os.chmod(creds_path, 0o600)
    except:
        pass
        
    print(f"Kaggle credentials saved to {creds_path}")
    return True

if __name__ == '__main__':
    setup_kaggle_credentials()