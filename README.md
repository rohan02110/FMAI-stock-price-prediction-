# 📈 Financial Machine Learning: Next-Day Stock Opening Price Prediction & Vision AI Damage Verification

An end-to-end Machine Learning and Multimodal Vision AI system built for quantitative financial modeling, stock price forecasting, and computer vision forensic verification.

---

## 🌟 Key Highlights & Modules

### 1. 📊 Next-Day Stock Opening Price Predictor (`app.py`, `main.py`, Phases 1–5)
- **Target Variable**: $Open_{t+1}$ (Next-Day Opening Price)
- **Asset**: Reliance Industries Limited (`RELIANCE.NS`) across 2021–2025 trading sessions.
- **Engineered Features**: 12 quantitative features including:
  - Inter-day momentum (5-day & 10-day Open Simple Moving Averages, Percentage Daily Returns)
  - Volatility metrics (Intraday High-Low spread, 10-day rolling volatility spread)
  - Sentiment & overnight signals (Intraday close-to-open sentiment, overnight gap persistence)
- **Model Evaluation**:
  - **Linear Regression (Primary OLS)**: $R^2 = 0.9880$, $MAE = ₹5.69$, $RMSE = ₹9.33$
  - **Random Forest Regressor (Ensemble)**: $R^2 = 0.9807$, $MAE = ₹8.77$, $RMSE = ₹11.81$
- **Interactive UI**: Real-time Streamlit dashboard (`app.py`) with Plotly charts, IPO presets, live candlestick analysis, and trend gap signals.

### 2. 🔍 Vision AI Broken Element & Window Re-Verification (`vision_damage_dashboard.py`, `broken_element_verifier.py`)
- Two-stage forensic verification pipeline designed to eliminate false positives (e.g., reflective window glares, tree shadows, frame dividers falsely flagged as fractures).
- Multi-scale patch block pixelation and Google Gemini Vision AI integration.

---

## 📁 Repository Structure

```
├── app.py                             # Interactive Financial ML Streamlit Dashboard
├── vision_damage_dashboard.py         # Forensic Vision AI Damage Verification Dashboard
├── broken_element_verifier.py         # Multimodal Gemini Vision AI & Patch Verifier Engine
├── test_reverification.py             # Vision AI pipeline CLI test suite
├── main.py                            # Master End-to-End Financial Modeling Pipeline
│
├── phase1_data_sourcing.py            # Phase 1: Yahoo Finance Automated Data Acquisition
├── phase2_feature_engineering.py      # Phase 2: Feature Engineering & Target Alignment
├── phase3_model_training.py           # Phase 3: Time-Series Split (70/15/15) & Model Training
├── phase4_evaluation_visualization.py # Phase 4: Statistical Evaluation & Visualization Suite
├── phase5_report_assembly.py          # Phase 5: Final Comprehensive Report Assembly
│
├── outputs/
│   ├── charts/                        # High-resolution academic charts & diagnostic plots
│   ├── tables/                        # Processed datasets, test predictions & metrics summary
│   ├── models/                        # Serialized models (.pkl) & scaler artifacts
│   ├── vision_tests/                  # Synthetic facade & patch inspection outputs
│   └── Final_Report.md                # Comprehensive academic project report
│
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Financial Modeling Pipeline (Phases 1–5)
```bash
python main.py
```

### 3. Launch Interactive Financial Dashboard
```bash
streamlit run app.py
```

### 4. Launch Vision AI Verification Dashboard
```bash
streamlit run vision_damage_dashboard.py
```

---

## 📊 Evaluation Results Summary

| Model | MAE (INR) | RMSE (INR) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Primary OLS)** | **₹5.69** | **₹9.33** | **0.9880** |
| **Random Forest Regressor** | **₹8.77** | **₹11.81** | **0.9807** |
