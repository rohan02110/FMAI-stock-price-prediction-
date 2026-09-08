"""
=============================================================================
Financial ML Stock & IPO Trend Predictor Dashboard
Problem Statement: Predicting Next-Day Opening Price (Open_{t+1}) and Trend
Built with Streamlit and Plotly
=============================================================================
"""

import os
import sys

# Ensure portable site-packages is in sys.path
PORTABLE_PACKAGES = r"C:\Users\Hp\PythonPortable\Lib\site-packages"
if os.path.exists(PORTABLE_PACKAGES) and PORTABLE_PACKAGES not in sys.path:
    sys.path.insert(0, PORTABLE_PACKAGES)

import streamlit as st  # pyrefly: ignore [missing-import] # type: ignore
import yfinance as yf  # pyrefly: ignore [missing-import] # type: ignore
import pandas as pd  # pyrefly: ignore [missing-import] # type: ignore
import numpy as np  # pyrefly: ignore [missing-import] # type: ignore
import plotly.graph_objects as go  # pyrefly: ignore [missing-import] # type: ignore
from plotly.subplots import make_subplots  # pyrefly: ignore [missing-import] # type: ignore
from sklearn.linear_model import LinearRegression  # pyrefly: ignore [missing-import] # type: ignore
from sklearn.ensemble import RandomForestRegressor  # pyrefly: ignore [missing-import] # type: ignore
from sklearn.preprocessing import StandardScaler  # pyrefly: ignore [missing-import] # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # pyrefly: ignore [missing-import] # type: ignore

# Set page configuration
st.set_page_config(
    page_title="Financial ML - Stock & IPO Trend Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-label { font-size: 14px; font-weight: 600; color: #6c757d; text-transform: uppercase; }
    .metric-val { font-size: 26px; font-weight: 800; color: #212529; margin-top: 4px; }
    .trend-bullish { color: #28a745; font-weight: 800; }
    .trend-bearish { color: #dc3545; font-weight: 800; }
    .trend-neutral { color: #ffc107; font-weight: 800; }
    .stButton>button {
        width: 100%;
        background-color: #0d6efd;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("🔍 Model & Asset Selector")

# Preset Recent IPOs and Major Equities
IPO_PRESETS = {
    "Swiggy Ltd. (SWIGGY.NS)": "SWIGGY.NS",
    "Hyundai Motor India (HYUNDAI.NS)": "HYUNDAI.NS",
    "Bajaj Housing Finance (BAJAJHFL.NS)": "BAJAJHFL.NS",
    "Ola Electric (OLAELC.NS)": "OLAELC.NS",
    "Tata Technologies (TATATECH.NS)": "TATATECH.NS",
    "IREDA (IREDA.NS)": "IREDA.NS",
    "Jio Financial Services (JIOFIN.NS)": "JIOFIN.NS",
    "Zomato Ltd. (ZOMATO.NS)": "ZOMATO.NS",
    "Paytm / One97 (PAYTM.NS)": "PAYTM.NS",
    "Nykaa (NYKAA.NS)": "NYKAA.NS",
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Motors (TATAMOTORS.NS)": "TATAMOTORS.NS",
    "Custom Ticker (Enter Below)": "CUSTOM"
}

selected_preset = st.sidebar.selectbox("Select Target IPO / Equity:", list(IPO_PRESETS.keys()))

if selected_preset == "Custom Ticker (Enter Below)":
    ticker_input = st.sidebar.text_input("Enter Yahoo Finance Ticker:", value="TCS.NS")
else:
    ticker_input = IPO_PRESETS[selected_preset]

timeframe = st.sidebar.selectbox(
    "Historical Data Horizon:",
    ["1y (1 Year)", "2y (2 Years)", "5y (5 Years)", "6mo (6 Months)", "max (Full History)"],
    index=1
)
period_code = timeframe.split()[0]

model_choice = st.sidebar.radio(
    "Select ML Forecasting Model:",
    ["Linear Regression (Primary OLS)", "Random Forest Regressor (Ensemble)", "Compare Both Models"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Specification:**
- **Target Variable:** $Open_{t+1}$ (Next-Day Open)
- **Features (12):** OHLCV + Moving Averages + Inter-Day Returns + Intraday & Overnight Sentiment Gaps
- **Temporal Split:** 70% Train, 15% Validation, 15% Test
""")

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_and_prepare_data(ticker, period):
    data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    data = data.reset_index()
    if 'Date' not in data.columns or len(data) < 25:
        return None
    
    df = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.ffill().bfill()
    
    # 12 Engineered Features
    df['SMA_5_Open'] = df['Open'].rolling(window=5).mean()
    df['SMA_10_Open'] = df['Open'].rolling(window=10).mean()
    df['Daily_Return'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100
    df['Daily_Volatility'] = df['High'] - df['Low']
    df['Rolling_Volatility_10'] = df['Daily_Volatility'].rolling(window=10).mean()
    df['Intraday_Sentiment'] = df['Close'] - df['Open']
    df['Overnight_Gap'] = df['Open'] - df['Close'].shift(1)
    
    # Target: Next-Day Open (Open_{t+1})
    df['Next_Day_Open'] = df['Open'].shift(-1)
    
    return df

# Fetch and Process Data
with st.spinner(f"Fetching market data for {ticker_input}..."):
    raw_df = load_and_prepare_data(ticker_input, period_code)

if raw_df is None or len(raw_df) < 25:
    st.error(f"❌ Could not retrieve sufficient historical data for ticker `{ticker_input}`. For newly listed IPOs, try selecting a shorter timeframe like `6mo` or verify the ticker symbol.")
    st.stop()

# Training Dataset (drop edge NaNs)
df_model = raw_df.dropna().reset_index(drop=True)

feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_5_Open', 'SMA_10_Open', 'Daily_Return',
    'Daily_Volatility', 'Rolling_Volatility_10',
    'Intraday_Sentiment', 'Overnight_Gap'
]

X = df_model[feature_cols]
y = df_model['Next_Day_Open']

# Chronological Time-Series Split (70/15/15)
n_total = len(df_model)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)
n_test = n_total - n_train - n_val

X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
X_test, y_test = X.iloc[n_train + n_val:], y.iloc[n_train + n_val:]
test_dates = df_model['Date'].iloc[n_train + n_val:]

# Fit Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

# -----------------------------------------------------------------------------
# LIVE INFERENCE FOR THE UPCOMING TRADING DAY (LATEST SESSION)
# -----------------------------------------------------------------------------
latest_row = raw_df.iloc[-1]
latest_features = raw_df.iloc[[-1]][feature_cols]
latest_features_scaled = scaler.transform(latest_features)

pred_next_open_lr = lr_model.predict(latest_features_scaled)[0]
pred_next_open_rf = rf_model.predict(latest_features_scaled)[0]

if "Random Forest" in model_choice:
    pred_next_open = pred_next_open_rf
    active_mae = rf_mae
    active_r2 = rf_r2
    active_model_name = "Random Forest Regressor"
else:
    pred_next_open = pred_next_open_lr
    active_mae = lr_mae
    active_r2 = lr_r2
    active_model_name = "Linear Regression (Primary)"

current_close = latest_row['Close']
expected_gap = pred_next_open - current_close
expected_gap_pct = (expected_gap / current_close) * 100

# Determine Trend Signal
if expected_gap_pct > 0.30:
    trend_signal = "🟢 BULLISH GAP-UP"
    trend_class = "trend-bullish"
    trend_desc = f"Expected to open +₹{expected_gap:.2f} (+{expected_gap_pct:.2f}%) higher"
elif expected_gap_pct < -0.30:
    trend_signal = "🔴 BEARISH GAP-DOWN"
    trend_class = "trend-bearish"
    trend_desc = f"Expected to open -₹{abs(expected_gap):.2f} ({expected_gap_pct:.2f}%) lower"
else:
    trend_signal = "🟡 NEUTRAL / FLAT OPEN"
    trend_class = "trend-neutral"
    trend_desc = f"Expected to open flat near ₹{pred_next_open:,.2f} ({expected_gap_pct:+.2f}%)"

# -----------------------------------------------------------------------------
# MAIN DASHBOARD UI
# -----------------------------------------------------------------------------
st.title(f"📈 Stock & IPO Trend Predictor: `{ticker_input}`")
st.caption(f"Machine Learning Next-Day Opening Price Forecast ($Open_{{t+1}}$) | Last Market Close: {latest_row['Date'].strftime('%d-%b-%Y')}")

# Top Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Last Session Close</div>
        <div class="metric-val">₹{current_close:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Next Open</div>
        <div class="metric-val">₹{pred_next_open:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Expected Gap (₹ / %)</div>
        <div class="metric-val {trend_class}">{expected_gap:+.2f} ({expected_gap_pct:+.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Trend</div>
        <div class="metric-val {trend_class}">{trend_signal.split()[1]}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Model Accuracy (R²)</div>
        <div class="metric-val">{active_r2*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INTERACTIVE CHARTS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Actual vs. Predicted Test Horizon", "🕯️ Price & Technical Indicators", "📋 Prediction Error Table & Metrics"])

with tab1:
    st.subheader(f"Test-Set Forecast Horizon: Actual vs. Predicted Next-Day Open")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=test_dates, y=y_test,
        mode='lines+markers', name='Actual Next-Day Open',
        line=dict(color='#111111', width=2), marker=dict(size=4)
    ))
    if model_choice in ["Linear Regression (Primary OLS)", "Compare Both Models"]:
        fig1.add_trace(go.Scatter(
            x=test_dates, y=y_pred_lr,
            mode='lines', name=f'Linear Regression (MAE: ₹{lr_mae:.2f}, R²: {lr_r2:.4f})',
            line=dict(color='#0d6efd', width=2, dash='dash')
        ))
    if model_choice in ["Random Forest Regressor (Ensemble)", "Compare Both Models"]:
        fig1.add_trace(go.Scatter(
            x=test_dates, y=y_pred_rf,
            mode='lines', name=f'Random Forest (MAE: ₹{rf_mae:.2f}, R²: {rf_r2:.4f})',
            line=dict(color='#dc3545', width=1.8, dash='dot')
        ))
    fig1.update_layout(
        template='plotly_white',
        xaxis_title="Trading Date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader(f"Historical Candlestick & Technical Moving Averages")
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.75, 0.25])
    
    # Candlestick
    fig2.add_trace(go.Candlestick(
        x=raw_df['Date'],
        open=raw_df['Open'], high=raw_df['High'],
        low=raw_df['Low'], close=raw_df['Close'],
        name='OHLC'
    ), row=1, col=1)
    
    # Moving Averages
    fig2.add_trace(go.Scatter(
        x=raw_df['Date'], y=raw_df['SMA_5_Open'],
        name='5-Day Open SMA', line=dict(color='#0d6efd', width=1.5)
    ), row=1, col=1)
    
    fig2.add_trace(go.Scatter(
        x=raw_df['Date'], y=raw_df['SMA_10_Open'],
        name='10-Day Open SMA', line=dict(color='#e83e8c', width=1.5)
    ), row=1, col=1)
    
    # Volume
    fig2.add_trace(go.Bar(
        x=raw_df['Date'], y=raw_df['Volume'],
        name='Volume', marker=dict(color='#6c757d', opacity=0.5)
    ), row=2, col=1)
    
    fig2.update_layout(
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        height=550,
        hovermode="x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("Model Comparison Metrics (Test Set)")
        metrics_summary_df = pd.DataFrame({
            'Model': ['Linear Regression (Primary)', 'Random Forest Regressor'],
            'MAE (₹)': [f"₹{lr_mae:.2f}", f"₹{rf_mae:.2f}"],
            'RMSE (₹)': [f"₹{np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}", f"₹{np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}"],
            'R² Score': [f"{lr_r2:.4f}", f"{rf_r2:.4f}"]
        })
        st.table(metrics_summary_df)
    
    with col_m2:
        st.subheader("Top Feature Importances (Random Forest)")
        importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(6)
        imp_df = pd.DataFrame({'Feature': importances.index, 'Importance': [f"{v*100:.2f}%" for v in importances.values]})
        st.table(imp_df)
        
    st.subheader("Recent Test Set Prediction Results")
    test_results_table = pd.DataFrame({
        'Date': test_dates.dt.strftime('%d-%b-%Y').values[-15:],
        'Actual Open': [f"₹{v:,.2f}" for v in y_test.values[-15:]],
        'LR Predicted Open': [f"₹{v:,.2f}" for v in y_pred_lr[-15:]],
        'LR Error': [f"₹{abs(a - p):,.2f}" for a, p in zip(y_test.values[-15:], y_pred_lr[-15:])],
        'RF Predicted Open': [f"₹{v:,.2f}" for v in y_pred_rf[-15:]],
        'RF Error': [f"₹{abs(a - p):,.2f}" for a, p in zip(y_test.values[-15:], y_pred_rf[-15:])]
    })
    st.dataframe(test_results_table, use_container_width=True)

# -----------------------------------------------------------------------------
# FOOTER / ACADEMIC DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown("---")
st.info("💡 **Academic Notice & Conclusion:** The Machine Learning model identifies relationships in historical financial data, volatility metrics, and overnight sentiment to provide an estimated next-day opening price and trend signal. However, stock prices are affected by unpredictable macroeconomic factors, so the model should be considered an analytical aid rather than a guarantee of future market performance.")
