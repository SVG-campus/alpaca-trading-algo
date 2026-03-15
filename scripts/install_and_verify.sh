#!/bin/bash
# GCP VM Diagnostic & Setup Script
# Run this on your GCP VM to guarantee the environment is ready for Kaggle/Alpaca

echo "Updating pip and installing required dependencies..."
pip3 install --upgrade pip
pip3 install kaggle alpaca-py yfinance pandas numpy scikit-learn

echo "Verifying Kaggle Authentication..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: ~/.kaggle/kaggle.json not found. Please upload it."
    exit 1
else
    echo "Kaggle JSON found. Checking permissions..."
    chmod 600 ~/.kaggle/kaggle.json
    kaggle competitions list --count 1
    if [ $? -eq 0 ]; then
        echo "✅ Kaggle CLI Authenticated."
    else
        echo "❌ Kaggle Authentication Failed."
        exit 1
    fi
fi

echo "Verifying Python path and environment..."
/usr/bin/python3 -c "import kaggle; print('✅ Kaggle Python module loaded successfully.')"
if [ $? -eq 0 ]; then
    echo "Environment verified. Everything is ready for automated runs."
else
    echo "❌ Python module not found. Check environment path."
    exit 1
fi