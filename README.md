# 📊 ML-Based Intraday Trading Strategy (Streamlit)

A professional, end-to-end Machine Learning–powered intraday trading system built with **Python**, **Streamlit**, and **scikit-learn**.
This project transforms raw OHLCV data into **signals, trades, performance analytics, and interactive visual dashboards** — providing a real trading-research experience suitable for quant, data science, and algorithmic trading roles.

---

## 🚀 Features

### **1. Data Handling & Validation**

* Upload any intraday OHLCV CSV
* Auto-detects datetime column
* Cleans, sorts, validates required fields
* Filters duplicate timestamps

### **2. Technical Indicator Engine**

Generates 40+ indicators, including:

* EMA9, EMA20, EMA50, EMA200
* VWAP, ATR, ATR%
* Bollinger Bands (width, position)
* RSI, MACD, Stochastic, ROC
* OBV, volume trends, volume ratios
* Candle structure (body, shadows, ranges)
* Support/resistance distances
* ADX & trend strength

### **3. Labeling Methods**

Supports 3 market-supervised labeling methods:

* Threshold returns
* Tercile labeling
* Quintile labeling
  Configurable prediction horizon and thresholds.

### **4. Machine Learning Models**

Trains and compares:

* Gaussian Naive Bayes
* Logistic Regression
* Random Forest
* Gradient Boosting

Shows accuracy, AUC, classification reports, and confusion matrices.

### **5. Advanced Signal Generation**

Combines:

* ML predictions
* Trend confirmation
* Momentum/volume filters
* Optional regime filters (trending vs ranging)

Generates:

* Strong Long / Moderate Long
* Strong Short / Moderate Short
* Confidence-based signals

### **6. Professional Backtesting Engine**

* ATR-based SL & TP
* Dynamic position sizing (risk % per trade)
* Optional ATR trailing stop
* Tracks PnL, PnL%, equity, shares
* Saves complete trade logs

### **7. Portfolio & Performance Analytics**

Automatically computes:

* Total return & %
* Win rate
* Sharpe ratio
* Sortino ratio
* Max drawdown
* Calmar ratio
* Expectancy
* Profit factor
* Max loss streak

### **8. Interactive Visualizations**

Includes:

* Equity curve
* Drawdown chart
* Return distribution
* Win/Loss pie
* Monthly PnL heatmap
* Cumulative PnL by trade
* Signal chart with buys/sells
* ML confidence plot
* Volume analysis
* Feature importance (for tree models)

### **9. Downloads**

* Trade log CSV
* Backtest result CSV

---

## 🗂 Project Structure

```
project/
│── app.py                      # Main Streamlit application
│── Quant Stratergy ML Model.py # Main Streamlit application
│── requirements.txt            # Dependencies
│── sample_data.csv             # Example OHLCV dataset (optional)
│── README.md                   # Documentation
```

---

## 🛠 Installation

### **1. Clone the repo**

```bash
git clone https://github.com/yourname/yourrepo.git
cd yourrepo
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run Streamlit app**

```bash
streamlit run app.py
```

---

## 📁 Required CSV Format

Your dataset must include:

| Column                  | Description |
| ----------------------- | ----------- |
| datetime / datetime_ist | Timestamp   |
| open                    | Open price  |
| high                    | High price  |
| low                     | Low price   |
| close                   | Close price |
| volume                  | Volume      |

---

## 🧠 How It Works (Pipeline)

1. **Upload OHLCV data**
2. **Clean + validate data**
3. **Generate 40+ technical indicators**
4. **Label the data (up/down)**
5. **Build ML dataset**
6. **Train all ML models**
7. **Choose best model**
8. **Generate trading signals**
9. **Run backtest**
10. **Analyze results with visual dashboards**
11. **Download logs for deeper analysis**

---

## 🎯 Use Cases

* Quantitative trading research
* ML-based signal development
* Strategy prototyping
* Backtesting framework
* Trading analytics dashboard
* Educational resource for ML + finance

---

## 📌 Notes

This project is for **research and educational purposes only**.
It does *not* predict the future or guarantee profit.

---

## 🤝 Contributing

Pull requests and issue reports are welcome.
Feel free to extend the ML models, add indicators, or improve visualization.

---

