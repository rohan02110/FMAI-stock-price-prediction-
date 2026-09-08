"""
=============================================================================
Phase 4: Evaluation & Complete Visualization Suite (Target: Next-Day Open)
Project: Machine Learning Techniques for Financial Data
Domain: Stock Market Next-Day Opening Price Prediction
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
import matplotlib.dates as mdates  # pyrefly: ignore [missing-import] # type: ignore
import seaborn as sns  # pyrefly: ignore [missing-import] # type: ignore
import joblib  # pyrefly: ignore [missing-import] # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # pyrefly: ignore [missing-import] # type: ignore

# Configure academic, presentation-grade aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'

# 1. Load Processed Data & Predictions
processed_csv = os.path.join("outputs", "tables", "processed_data.csv")
predictions_csv = os.path.join("outputs", "tables", "predictions.csv")

df = pd.read_csv(processed_csv)
df['Date'] = pd.to_datetime(df['Date'])

pred_df = pd.read_csv(predictions_csv)
pred_df['Date'] = pd.to_datetime(pred_df['Date'])

# Load models
models_dir = os.path.join("outputs", "models")
lr_model = joblib.load(os.path.join(models_dir, "linear_regression.pkl"))
rf_model = joblib.load(os.path.join(models_dir, "random_forest.pkl"))

# 2. Compute Evaluation Metrics on Test Set
actual_test = pred_df['Actual_Open_Price']
lr_pred_test = pred_df['LR_Predicted_Open']
rf_pred_test = pred_df['RF_Predicted_Open']

lr_mae = mean_absolute_error(actual_test, lr_pred_test)
lr_rmse = np.sqrt(mean_squared_error(actual_test, lr_pred_test))
lr_r2 = r2_score(actual_test, lr_pred_test)

rf_mae = mean_absolute_error(actual_test, rf_pred_test)
rf_rmse = np.sqrt(mean_squared_error(actual_test, rf_pred_test))
rf_r2 = r2_score(actual_test, rf_pred_test)

metrics_df = pd.DataFrame([
    {
        'Model': 'Linear Regression (Primary)',
        'MAE (INR)': round(lr_mae, 4),
        'RMSE (INR)': round(lr_rmse, 4),
        'R2 Score': round(lr_r2, 4)
    },
    {
        'Model': 'Random Forest Regressor (Benchmark)',
        'MAE (INR)': round(rf_mae, 4),
        'RMSE (INR)': round(rf_rmse, 4),
        'R2 Score': round(rf_r2, 4)
    }
])

metrics_csv_path = os.path.join("outputs", "tables", "metrics_summary.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"[OK] Metrics summary saved to: {metrics_csv_path}")

print("\n" + "="*80)
print("MODEL EVALUATION METRICS COMPARISON (TARGET: NEXT-DAY OPENING PRICE)")
print("="*80)
print(metrics_df.to_string(index=False))

# 3. Generate Visualizations Suite
charts_dir = os.path.join("outputs", "charts")

# Chart 1: Historical Opening Price Trend (2021 -> 2025)
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(df['Date'], df['Open'], color='#1f77b4', linewidth=1.8, label='Reliance Industries (Opening Price)')
ax.fill_between(df['Date'], df['Open'], color='#1f77b4', alpha=0.1)
ax.set_title("Chart 1: Historical Opening Price Trend (2021 - 2025)", pad=12)
ax.set_xlabel("Trading Date")
ax.set_ylabel("Opening Price (INR)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
chart1_path = os.path.join(charts_dir, "chart1_closing_price_trend.png")
plt.savefig(chart1_path, dpi=300)
plt.close()
print(f"[OK] Saved: {chart1_path}")

# Chart 2: Inter-Day Momentum & Returns Trend
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(df['Date'], df['Daily_Return'], color='#2ca02c', linewidth=0.9, alpha=0.85, label='Daily Percentage Return (%)')
ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_title("Chart 2: Inter-Day Momentum & Return Volatility Trend (2021 - 2025)", pad=12)
ax.set_xlabel("Trading Date")
ax.set_ylabel("Daily Return (%)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
chart2_path = os.path.join(charts_dir, "chart2_daily_returns_volatility.png")
plt.savefig(chart2_path, dpi=300)
plt.close()
print(f"[OK] Saved: {chart2_path}")

# Chart 3: High vs. Low Intraday Volatility Spread Trend
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(df['Date'], df['High'], color='#d62728', linewidth=1.2, alpha=0.85, label='Daily High Price')
ax.plot(df['Date'], df['Low'], color='#1f77b4', linewidth=1.2, alpha=0.85, label='Daily Low Price')
ax.fill_between(df['Date'], df['Low'], df['High'], color='#9467bd', alpha=0.18, label='Intraday Volatility Spread')
ax.set_title("Chart 3: Daily High vs. Low Price Spread & Volatility Range (2021 - 2025)", pad=12)
ax.set_xlabel("Trading Date")
ax.set_ylabel("Price (INR)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
chart3_path = os.path.join(charts_dir, "chart3_high_vs_low_trend.png")
plt.savefig(chart3_path, dpi=300)
plt.close()
print(f"[OK] Saved: {chart3_path}")

# Chart 4: Trading Volume Trend over Time
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.bar(df['Date'], df['Volume'] / 1e6, color='#ff7f0e', alpha=0.65, width=1.8, label='Trading Volume (Million Shares)')
ax.plot(df['Date'], (df['Volume'].rolling(20).mean()) / 1e6, color='#d62728', linewidth=1.8, label='20-Day Avg Volume')
ax.set_title("Chart 4: Trading Volume Trend over Time (2021 - 2025)", pad=12)
ax.set_xlabel("Trading Date")
ax.set_ylabel("Volume (Million Shares)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
chart4_path = os.path.join(charts_dir, "chart4_volume_trend.png")
plt.savefig(chart4_path, dpi=300)
plt.close()
print(f"[OK] Saved: {chart4_path}")

# Chart 5: Technical Momentum Moving Averages (5-Day vs 10-Day Open SMA)
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(df['Date'], df['Open'], color='#7f7f7f', linewidth=1.0, alpha=0.6, label='Daily Open Price')
ax.plot(df['Date'], df['SMA_5_Open'], color='#1f77b4', linewidth=1.6, label='5-Day Open SMA (Short-term Momentum)')
ax.plot(df['Date'], df['SMA_10_Open'], color='#e377c2', linewidth=1.8, label='10-Day Open SMA (Medium-term Trend)')
ax.set_title("Chart 5: Technical Momentum Trend (5-Day vs. 10-Day Opening Price Moving Average)", pad=12)
ax.set_xlabel("Trading Date")
ax.set_ylabel("Price (INR)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
chart5_path = os.path.join(charts_dir, "chart5_moving_averages_trend.png")
plt.savefig(chart5_path, dpi=300)
plt.close()
print(f"[OK] Saved: {chart5_path}")

# Chart 6: Actual vs. Predicted Next-Day Opening Price (Test Horizon)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(pred_df['Date'], pred_df['Actual_Open_Price'], color='#111111', linewidth=2.2, label='Actual Next-Day Open', marker='o', markersize=3, alpha=0.9)
ax.plot(pred_df['Date'], pred_df['LR_Predicted_Open'], color='#1f77b4', linewidth=1.8, linestyle='--', label=f'Linear Regression (MAE: INR {lr_mae:.2f}, R2: {lr_r2:.4f})')
ax.plot(pred_df['Date'], pred_df['RF_Predicted_Open'], color='#d62728', linewidth=1.6, linestyle=':', label=f'Random Forest (MAE: INR {rf_mae:.2f}, R2: {rf_r2:.4f})')
ax.set_title("Chart 6: Actual vs. Predicted Next-Day Opening Price (Test Set Horizon)", pad=12)
ax.set_xlabel("Testing Period Date (2025)")
ax.set_ylabel("Price (INR)")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=30)
ax.legend(loc='upper left', frameon=True, fontsize=10)
plt.tight_layout()
chart6_path = os.path.join(charts_dir, "chart6_actual_vs_predicted.png")
plt.savefig(chart6_path, dpi=300)
plt.close()
print(f"[OK] Saved: {chart6_path}")

# Residual Diagnostic Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
lr_residuals = pred_df['Actual_Open_Price'] - pred_df['LR_Predicted_Open']
rf_residuals = pred_df['Actual_Open_Price'] - pred_df['RF_Predicted_Open']

ax1.scatter(pred_df['Date'], lr_residuals, color='#1f77b4', alpha=0.6, label='LR Residuals', s=25)
ax1.scatter(pred_df['Date'], rf_residuals, color='#d62728', alpha=0.4, label='RF Residuals', s=25)
ax1.axhline(0, color='black', linestyle='--', linewidth=1)
ax1.set_title("Residual Errors over Time (Actual - Predicted Open)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Residual Error (INR)")
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
ax1.legend()

sns.kdeplot(lr_residuals, ax=ax2, color='#1f77b4', fill=True, alpha=0.3, label=f'LR (Mean={lr_residuals.mean():.2f})')
sns.kdeplot(rf_residuals, ax=ax2, color='#d62728', fill=True, alpha=0.3, label=f'RF (Mean={rf_residuals.mean():.2f})')
ax2.axvline(0, color='black', linestyle='--', linewidth=1)
ax2.set_title("Residual Error Density Distribution")
ax2.set_xlabel("Residual Error (INR)")
ax2.set_ylabel("Density")
ax2.legend()

plt.tight_layout()
residual_path = os.path.join(charts_dir, "residual_plot.png")
plt.savefig(residual_path, dpi=300)
plt.close()
print(f"[OK] Saved: {residual_path}")
