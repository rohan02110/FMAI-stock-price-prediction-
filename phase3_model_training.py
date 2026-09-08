"""
=============================================================================
Phase 3: Model Building, Training & Prediction (Target: Next-Day Open)
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
import joblib  # pyrefly: ignore [missing-import] # type: ignore
from sklearn.linear_model import LinearRegression  # pyrefly: ignore [missing-import] # type: ignore
from sklearn.ensemble import RandomForestRegressor  # pyrefly: ignore [missing-import] # type: ignore
from sklearn.preprocessing import StandardScaler  # pyrefly: ignore [missing-import] # type: ignore

# Set random seed for full reproducibility
np.random.seed(42)

# 1. Load Processed Dataset from Phase 2
processed_csv_path = os.path.join("outputs", "tables", "processed_data.csv")
df = pd.read_csv(processed_csv_path)
print(f"[OK] Processed dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Define Feature Columns and Target Variable (Next_Day_Open)
feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_5_Open', 'SMA_10_Open', 'Daily_Return',
    'Daily_Volatility', 'Rolling_Volatility_10',
    'Intraday_Sentiment', 'Overnight_Gap'
]
target_col = 'Next_Day_Open'

X = df[feature_cols]
y = df[target_col]
dates = df['Date']

# 2. Chronological Time-Series Partitioning (70% Train, 15% Validation, 15% Test)
n_total = len(df)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)
n_test = n_total - n_train - n_val

train_idx = range(0, n_train)
val_idx = range(n_train, n_train + n_val)
test_idx = range(n_train + n_val, n_total)

X_train, y_train, dates_train = X.iloc[train_idx], y.iloc[train_idx], dates.iloc[train_idx]
X_val, y_val, dates_val = X.iloc[val_idx], y.iloc[val_idx], dates.iloc[val_idx]
X_test, y_test, dates_test = X.iloc[test_idx], y.iloc[test_idx], dates.iloc[test_idx]

print("\n" + "="*80)
print("TIME-SERIES DATASET PARTITIONING (CHRONOLOGICAL SPLIT)")
print("="*80)
print(f"Total Observations: {n_total}")
print(f"Training Set   (70%): {len(X_train)} samples ({dates_train.min()} to {dates_train.max()})")
print(f"Validation Set (15%): {len(X_val)} samples ({dates_val.min()} to {dates_val.max()})")
print(f"Test Set       (15%): {len(X_test)} samples ({dates_test.min()} to {dates_test.max()})")

# 3. Feature Scaling via StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. Model 1: Primary Model - Linear Regression
print("\n" + "="*80)
print("TRAINING MODEL 1: LINEAR REGRESSION (TARGET: Next-Day Open Price)")
print("="*80)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

print(f"Intercept (beta_0): {lr_model.intercept_:.4f}")
print("Learned Feature Coefficients:")
for feat, coef in zip(feature_cols, lr_model.coef_):
    print(f"  {feat:<25}: {coef:+.4f}")

# 5. Model 2: Secondary Comparison Model - Random Forest Regressor
print("\n" + "="*80)
print("TRAINING MODEL 2: RANDOM FOREST REGRESSOR (COMPARISON)")
print("="*80)
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

print("Random Forest Feature Importances:")
rf_importances = sorted(zip(feature_cols, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)
for feat, imp in rf_importances:
    print(f"  {feat:<25}: {imp*100:6.2f}%")

# 6. Generate Predictions on Test Set
y_test_pred_lr = lr_model.predict(X_test_scaled)
y_test_pred_rf = rf_model.predict(X_test_scaled)

# 7. Construct Predictions Table
predictions_df = pd.DataFrame({
    'Date': dates_test.values,
    'Actual_Open_Price': y_test.values,
    'LR_Predicted_Open': y_test_pred_lr,
    'LR_Error': np.abs(y_test.values - y_test_pred_lr),
    'RF_Predicted_Open': y_test_pred_rf,
    'RF_Error': np.abs(y_test.values - y_test_pred_rf)
})

# Save predictions table to CSV
pred_csv_path = os.path.join("outputs", "tables", "predictions.csv")
predictions_df.to_csv(pred_csv_path, index=False)
print(f"\n[OK] Predictions saved to: {pred_csv_path}")

# 8. Save Trained Models & Scaler Artifacts
model_dir = os.path.join("outputs", "models")
joblib.dump(lr_model, os.path.join(model_dir, "linear_regression.pkl"))
joblib.dump(rf_model, os.path.join(model_dir, "random_forest.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
print(f"[OK] Trained models and scaler serialized to: {model_dir}")

# 9. Display Sample Results Tables
print("\n" + "="*80)
print("TEST SET PREDICTION TABLE: LINEAR REGRESSION (First 10 Samples - Target: Next-Day Open)")
print("="*80)
sample_lr_view = pd.DataFrame({
    'Date': predictions_df['Date'].head(10),
    'Actual Open (INR)': predictions_df['Actual_Open_Price'].head(10).map(lambda x: f"INR {x:,.2f}"),
    'Predicted Open (INR)': predictions_df['LR_Predicted_Open'].head(10).map(lambda x: f"INR {x:,.2f}"),
    'Error (INR)': predictions_df['LR_Error'].head(10).map(lambda x: f"INR {x:,.2f}")
})
print(sample_lr_view.to_string(index=False))
