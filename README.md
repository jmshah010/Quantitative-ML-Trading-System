
# 📊 Quantitative ML Trading System (Intraday)

**End-to-end Machine Learning Pipeline · Technical Indicators · Backtesting Engine · Streamlit UI**

This repository implements a **production-style intraday trading research framework**, combining technical indicators, supervised ML models, signal generation logic, and a lightweight backtesting engine. The objective is to provide a **real-world quant research workflow**, suitable for interviews, portfolio showcases, and professional applications in algorithmic trading, data science, and financial engineering.

The system automates the complete pipeline:
**Data → Feature Engineering → Labeling → ML Modeling → Signal Generation → Backtesting → Visualization.**

---

## 🔥 Key Highlights

### ✔ Full ML Pipeline for Market Supervision

* 40+ engineered technical features
* Multiple labeling schemes (threshold, tercile, quintile)
* Cross-validated model training
* Gaussian Naive Bayes, Logistic Regression, RF, GBM
* Feature importance for interpretability

### ✔ Realistic Signal Logic

* ML probability outputs → signal scoring
* Trend confirmations (EMA stack, ADX, VWAP, MACD)
* Volume/momentum filters
* Strong/Moderate Long/Short signals

### ✔ Professional Backtesting Engine

* ATR-based SL/TP
* Dynamic position sizing
* Trailing ATR stop (optional)
* Trade logs, PnL tables, equity curve
* Full performance analytics (Sharpe, Sortino, Calmar, PF, WR)

### ✔ Streamlit Frontend

* Upload any OHLCV CSV
* Full workflow automation
* Signal visualization
* Equity curve, drawdown, distribution plots
* Downloadable reports

---

# 🧩 System Architecture

```
            ┌────────────────┐
            │ Raw OHLCV Data │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │ Data Cleaner    │
            │ • Sort          │
            │ • Validate      │
            │ • Format        │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │ Feature Engine  │
            │ • 40+ Indicators│
            │ • Candle Stats  │
            │ • Trend/Volume  │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │ Label Generator │
            │ • Threshold     │
            │ • Tercile       │
            │ • Quintile      │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │ ML Models       │
            │ • GNB, LR, RF   │
            │ • GBM           │
            │ • Evaluation    │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │ Signal Engine   │
            │ • ML + Trend    │
            │ • Vol Filters   │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │ Backtester      │
            │ • SL/TP/ATR     │
            │ • PnL & Equity  │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │ Streamlit UI    │
            └─────────────────┘
```

---

# 📁 Folder Structure (Professional Layout)

```
Quantitative-ML-Trading-System/
│
├── data/
│   ├── raw/                 # Raw OHLCV, JSON, CSV
│   └── processed/           # Cleaned data (optional)
│
├── models/
│   ├── gnb_model.pkl
│   ├── gnb_scaler.pkl
│   └── feature_cols.pkl
│
├── src/
│   ├── app.py               # Streamlit interface
│   └── QuantStrategy_ML_Model.py
│
├── notebooks/
│   └── json_to_csv_converter.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/Quantitative-ML-Trading-System.git
cd Quantitative-ML-Trading-System
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Launch Streamlit App

```bash
streamlit run src/app.py
```

---

# 📄 Required Input Data Format

Your uploaded CSV must contain:

| Column                  | Description   |
| ----------------------- | ------------- |
| datetime / datetime_ist | Timestamp     |
| open                    | Opening price |
| high                    | High price    |
| low                     | Low price     |
| close                   | Closing price |
| volume                  | Volume        |

---

# 🔍 Pipeline Explained (Step-by-Step)

### **1. Data Ingestion & Validation**

* Detects datetime column automatically
* Removes duplicates
* Ensures correctly sorted intraday index
* Validates required columns

### **2. Technical Feature Engineering**

40+ features including:

* EMA9/20/50/200
* ATR, ATR%
* MACD, RSI, Stochastic
* OBV, volume trends
* Bollinger Bands metrics
* Candle body/wicks/ranges
* Trend strength (ADX)
* Distance to support & resistance

### **3. Label Generation**

Supports multiple ML labeling schemes:

| Label Method     | Use-case              |
| ---------------- | --------------------- |
| Threshold Return | Simple up/down        |
| Tercile          | Balanced classes      |
| Quintile         | Finer prediction bins |

### **4. Machine Learning Modeling**

Models trained & compared:

* Gaussian Naive Bayes
* Logistic Regression
* Random Forest
* Gradient Boosting

Model outputs include:

* Accuracy
* Precision, Recall, F1
* AUC
* Confusion matrix
* Cross-validation scores

### **5. Signal Generation**

Signals combine predicted class + trend + volume:

* **Strong Long**
* **Moderate Long**
* **Strong Short**
* **Moderate Short**

### **6. Backtesting Framework**

* ATR-based stop loss
* Dynamic SL/TP
* Trailing ATR (optional)
* Position sizing
* Trade logging
* Performance statistics
* Equity curve

### **7. Visualization Layer**

* Price with buy/sell markers
* Equity curve
* Drawdown plot
* Monthly PnL heatmap
* Win/loss pie
* Feature importance
* ML probability charts

---

# 📈 Performance Metrics Calculated

| Metric        | Purpose                    |
| ------------- | -------------------------- |
| Sharpe Ratio  | Risk-adjusted return       |
| Sortino Ratio | Downside-risk efficiency   |
| Max Drawdown  | Largest loss streak        |
| Calmar Ratio  | Return vs drawdown         |
| Profit Factor | Gross win / gross loss     |
| Expectancy    | Avg return per trade       |
| Win Rate      | Accuracy of trading system |

---

# 🎯 Intended Use-Cases

* Quant research prototyping
* ML strategy development
* Streamlit dashboarding for markets
* Backtesting & trade analysis
* Educational demonstration of ML in trading

---

# 🧪 Sample Workflow

1. Upload intraday OHLCV CSV
2. Select labeling method + model
3. Generate features
4. Train ML model
5. View metrics
6. Run backtest
7. Analyze trades
8. Export logs

---

# 📝 Roadmap & Future Enhancements

* XGBoost & CatBoost model support
* Portfolio-level backtesting
* Market regime detection using HMM
* Auto feature-selection (Boruta / SHAP)
* Live data integration
* API-based execution (AngelOne, Fyers, IBKR)

---

# ⚠️ Disclaimer

This project is strictly for **research and educational** purposes.
It is **not** intended for live trading or financial advice.

---

# 🤝 Contributing

Contributions are welcome.
Feel free to open issues or submit PRs for improvements.

