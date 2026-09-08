"""
=============================================================================
Phase 2: Data Cleaning, Feature Engineering & Target Variable Creation
Project: Machine Learning Techniques for Financial Data
Domain: Stock Market Next-Day Opening Price Prediction (Open_{t+1})
=============================================================================
Problem Statement: To design and implement a Machine Learning model that 
predicts the next-day opening price (Open_{t+1}) of a target equity asset 
by analyzing historical OHLCV data, inter-day momentum indicators, 
volatility metrics, and overnight market sentiment.
=============================================================================
"""

import os
import sys

# Ensure portable site-packages is in sys.path
PORTABLE_PACKAGES = r"C:\Users\Hp\PythonPortable\Lib\site-packages"
if os.path.exists(PORTABLE_PACKAGES) and PORTABLE_PACKAGES not in sys.path:
    sys.path.insert(0, PORTABLE_PACKAGES)

import pandas as pd  # pyrefly: ignore [missing-import] # type: ignore
import numpy as np  # pyrefly: ignore [missing-import] # type: ignore
import matplotlib.pyplot as plt  # pyrefly: ignore [missing-import] # type: ignore
import seaborn as sns  # pyrefly: ignore [missing-import] # type: ignore

# Set style for academic, publication-ready figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# 1. Load Raw Dataset from Phase 1
raw_csv_path = os.path.join("outputs", "tables", "raw_data.csv")
df = pd.read_csv(raw_csv_path)
print(f"[OK] Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Data Cleaning & Validation
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.drop_duplicates(subset=['Date']).reset_index(drop=True)
df = df.ffill().bfill()
print(f"[OK] Chronological sorting verified and missing values checked.")

# 3. Feature Engineering aligned with Problem Statement

# A. Base OHLCV Features: Open, High, Low, Close, Volume
# (Retained as primary price anchor points)

# B. Inter-Day Momentum Indicators:
# 1. 5-Day Simple Moving Average of Opening Price
df['SMA_5_Open'] = df['Open'].rolling(window=5).mean()
# 2. 10-Day Simple Moving Average of Opening Price
df['SMA_10_Open'] = df['Open'].rolling(window=10).mean()
# 3. Daily Percentage Return (Inter-day closing price momentum)
df['Daily_Return'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100

# C. Volatility Metrics:
# 4. Daily Intraday Volatility Spread (High - Low)
df['Daily_Volatility'] = df['High'] - df['Low']
# 5. 10-Day Rolling Average Volatility Spread
df['Rolling_Volatility_10'] = df['Daily_Volatility'].rolling(window=10).mean()

# D. Overnight & Intraday Market Sentiment Signals:
# 6. Intraday Closing Sentiment (Close - Open) -> Bullish/Bearish session pressure
df['Intraday_Sentiment'] = df['Close'] - df['Open']
# 7. Previous Session Overnight Gap (Open_t - Close_{t-1}) -> Pre-market gap persistence
df['Overnight_Gap'] = df['Open'] - df['Close'].shift(1)

# 4. Target Variable Creation: Next-Day Opening Price (Open_{t+1})
# Shift Open column by -1 day (lead of 1)
df['Next_Day_Open'] = df['Open'].shift(-1)

# 5. Handle Boundary NaNs (drop first 10 rows for rolling windows, last 1 row for lead target)
pre_drop_len = len(df)
df_clean = df.dropna().reset_index(drop=True)
dropped_rows = pre_drop_len - len(df_clean)
print(f"\n[OK] Dropped {dropped_rows} boundary rows containing NaNs from lag/lead/rolling operations.")
print(f"Final Cleaned & Engineered Dataset Shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

# Format Date back to YYYY-MM-DD string
df_clean['Date'] = df_clean['Date'].dt.strftime('%Y-%m-%d')

# 6. Save Processed Dataset
processed_csv_path = os.path.join("outputs", "tables", "processed_data.csv")
df_clean.to_csv(processed_csv_path, index=False)
print(f"[OK] Processed dataset saved to: {processed_csv_path}")

# 7. Display Sample Target Alignment Table
sample_table = df_clean[['Date', 'Open', 'Close', 'Next_Day_Open']].copy()
sample_table.columns = ['Date', "Today's Open", "Today's Close", 'Next-Day Open (Target)']

print("\n" + "="*80)
print("SAMPLE TARGET ALIGNMENT TABLE: NEXT-DAY OPENING PRICE (First 10 Rows)")
print("="*80)
preview_df = sample_table.head(10).copy()
preview_df["Today's Open"] = preview_df["Today's Open"].map(lambda x: f"INR {x:,.2f}")
preview_df["Today's Close"] = preview_df["Today's Close"].map(lambda x: f"INR {x:,.2f}")
preview_df["Next-Day Open (Target)"] = preview_df["Next-Day Open (Target)"].map(lambda x: f"INR {x:,.2f}")
print(preview_df.to_string(index=False))

# 8. Feature Matrix (X) and Target Vector (y)
feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_5_Open', 'SMA_10_Open', 'Daily_Return',
    'Daily_Volatility', 'Rolling_Volatility_10',
    'Intraday_Sentiment', 'Overnight_Gap'
]
X = df_clean[feature_cols]
y = df_clean['Next_Day_Open']

print("\n" + "="*80)
print("FEATURE MATRIX (X) & TARGET VECTOR (y) OVERVIEW")
print("="*80)
print(f"Feature Matrix X Shape: {X.shape} ({X.shape[1]} engineered features)")
print(f"Features: {list(X.columns)}")
print(f"Target Vector y Shape:  {y.shape} (Next-Day Open Price: Open_{{t+1}})")
print("\nFirst 5 Rows of Feature Matrix (X):")
print(X.head().round(2).to_string())

# 9. Correlation Heatmap Generation
plt.figure(figsize=(12, 9))
analysis_cols = feature_cols + ['Next_Day_Open']
corr_matrix = df_clean[analysis_cols].corr()

# Create Heatmap
ax = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    linewidths=0.6,
    linecolor='white',
    square=True,
    annot_kws={"size": 9, "weight": "bold"}
)

plt.title("Feature Correlation Heatmap with Target Variable (Next-Day Open)", fontsize=13, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9, fontweight='bold')
plt.yticks(rotation=0, fontsize=9, fontweight='bold')
plt.tight_layout()

heatmap_path = os.path.join("outputs", "charts", "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n[OK] Correlation heatmap generated and saved to: {heatmap_path}")

# Display Target Correlation Vector
print("\n" + "="*80)
print("PEARSON CORRELATION WITH TARGET VARIABLE (Next_Day_Open)")
print("="*80)
corr_with_target = corr_matrix['Next_Day_Open'].sort_values(ascending=False)
for feat, score in corr_with_target.items():
    print(f"{feat:<25}: {score:+.4f}")
