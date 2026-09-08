"""
=============================================================================
Phase 1: Setup, Data Sourcing & Problem Confirmation
Project: Machine Learning Techniques for Financial Data
Domain: Stock Market Next-Day Closing Price Prediction
=============================================================================
"""

import os
import sys

# Ensure portable site-packages is in sys.path if not automatically loaded
PORTABLE_PACKAGES = r"C:\Users\Hp\PythonPortable\Lib\site-packages"
if os.path.exists(PORTABLE_PACKAGES) and PORTABLE_PACKAGES not in sys.path:
    sys.path.insert(0, PORTABLE_PACKAGES)

import yfinance as yf  # pyrefly: ignore [missing-import] # type: ignore
import pandas as pd  # pyrefly: ignore [missing-import] # type: ignore
import numpy as np  # pyrefly: ignore [missing-import] # type: ignore

# Set random seed for reproducibility
np.random.seed(42)

# 1. Create Directory Hierarchy
dirs = [
    os.path.join("outputs", "charts"),
    os.path.join("outputs", "tables"),
    os.path.join("outputs", "models")
]
for d in dirs:
    os.makedirs(d, exist_ok=True)
print("[OK] Output directories initialized: /outputs/charts, /outputs/tables, /outputs/models")

# 2. Define Parameters
TICKER = "RELIANCE.NS"
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

print(f"\nFetching historical OHLCV data for {TICKER} from {START_DATE} to {END_DATE} via yfinance...")

# 3. Download Raw OHLCV Data
ticker_data = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)

# If multi-index columns returned by yfinance, flatten them
if isinstance(ticker_data.columns, pd.MultiIndex):
    ticker_data.columns = [col[0] for col in ticker_data.columns]

# Reset index to have 'Date' as an explicit column
raw_df = ticker_data.reset_index()

# Ensure standard column naming and order
required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
raw_df = raw_df[[c for c in required_cols if c in raw_df.columns]]

# Convert Date to standard string YYYY-MM-DD
raw_df['Date'] = pd.to_datetime(raw_df['Date']).dt.strftime('%Y-%m-%d')

# Save Raw Dataset
raw_csv_path = os.path.join("outputs", "tables", "raw_data.csv")
raw_df.to_csv(raw_csv_path, index=False)
print(f"[OK] Raw dataset saved to {raw_csv_path}")

# Display Summary Information
print("\n" + "="*80)
print("RAW DATASET PREVIEW & METADATA")
print("="*80)
print(f"Dataset Shape: {raw_df.shape[0]} trading days, {raw_df.shape[1]} columns")
print(f"Date Range: {raw_df['Date'].min()} to {raw_df['Date'].max()}")
print("\nData Types:")
print(raw_df.dtypes.to_string())

print("\n--- FIRST 5 ROWS ---")
print(raw_df.head().to_string(index=False))

print("\n--- LAST 5 ROWS ---")
print(raw_df.tail().to_string(index=False))

print("\n--- DESCRIPTIVE STATISTICS (.describe()) ---")
print(raw_df[['Open', 'High', 'Low', 'Close', 'Volume']].describe().round(2).to_string())
