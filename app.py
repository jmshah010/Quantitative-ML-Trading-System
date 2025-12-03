# app.py
# ==========================================
# SUPER 3-CONFIRMATION INTRADAY MODEL
# Gaussian Naive Bayes + EMA + VWAP + Price Action
# Timeframe: 15-min
# With metrics, plots, pickle, and Streamlit dashboard
# CSV format: datetime_ist,timestamp,open,high,low,close,volume
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import streamlit as st

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


# ------------------------------------------
# 1. DATA LOADING
# ------------------------------------------

def load_data_from_csv(file) -> pd.DataFrame:
    """
    file: path string or file-like object from st.file_uploader
    Expects columns:
      - datetime_ist OR atetime_ist (typo safe)
      - timestamp
      - open, high, low, close, volume
    """
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]

    # Handle datetime column name
    datetime_col = None
    if 'datetime_ist' in df.columns:
        datetime_col = 'datetime_ist'
    elif 'atetime_ist' in df.columns:  # handle typo just in case
        datetime_col = 'atetime_ist'

    if datetime_col is None:
        raise ValueError("CSV must have a 'datetime_ist' (or 'atetime_ist') column.")

    df[datetime_col] = pd.to_datetime(df[datetime_col])  # keeps +05:30 tz if present
    df = df.sort_values(datetime_col).reset_index(drop=True)
    df.set_index(datetime_col, inplace=True)
    df.index.name = "datetime_ist"

    # We ignore 'timestamp' for modeling; keep if you want
    return df


# ------------------------------------------
# 2. INDICATORS
# ------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # EMA 20 & 50
    d['ema20'] = d['close'].ewm(span=20, adjust=False).mean()
    d['ema50'] = d['close'].ewm(span=50, adjust=False).mean()

    # VWAP (cumulative – simple version)
    cum_vol = d['volume'].cumsum()
    cum_vp = (d['close'] * d['volume']).cumsum()
    d['vwap'] = cum_vp / cum_vol.replace(0, np.nan)

    # ATR(14)
    high_low = d['high'] - d['low']
    high_close_prev = (d['high'] - d['close'].shift(1)).abs()
    low_close_prev = (d['low'] - d['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    d['atr14'] = tr.rolling(window=14).mean()

    # RSI(14)
    delta = d['close'].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=d.index).rolling(14).mean()
    roll_down = pd.Series(loss, index=d.index).rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    d['rsi14'] = 100.0 - (100.0 / (1.0 + rs))

    # Volume features
    d['vol_ma20'] = d['volume'].rolling(20).mean()
    d['vol_ratio'] = d['volume'] / d['vol_ma20']

    # Simple swing levels
    d['roll_max_20'] = d['high'].rolling(20).max()
    d['roll_min_20'] = d['low'].rolling(20).min()

    # Candlestick body / range
    d['body'] = d['close'] - d['open']
    d['range'] = d['high'] - d['low']

    return d


# ------------------------------------------
# 3. LABELS FOR ML (direction)
# ------------------------------------------

def create_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    d = df.copy()
    d['future_close'] = d['close'].shift(-horizon)
    d['direction'] = np.where(d['future_close'] > d['close'], 1, 0)
    return d


# ------------------------------------------
# 4. FEATURE MATRIX
# ------------------------------------------

def build_ml_dataset(df: pd.DataFrame):
    feature_cols = [
        'close', 'ema20', 'ema50', 'vwap',
        'atr14', 'rsi14', 'vol_ratio',
        'body', 'range', 'roll_max_20', 'roll_min_20'
    ]
    d = df.dropna().copy()
    X = d[feature_cols].values
    y = d['direction'].values
    return d, X, y, feature_cols


# ------------------------------------------
# 5. TRAIN GAUSSIAN NB
# ------------------------------------------

def train_gnb(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    report = classification_report(y_test, y_pred, digits=3, output_dict=True)
    acc = accuracy_score(y_test, y_pred)

    return model, scaler, report, acc


# ------------------------------------------
# 6. APPLY MODEL ON FULL DATA
# ------------------------------------------

def apply_model(df_ml: pd.DataFrame, model, scaler, feature_cols):
    d = df_ml.copy()
    X_all = d[feature_cols].values
    X_all_scaled = scaler.transform(X_all)
    probs = model.predict_proba(X_all_scaled)
    d['prob_down'] = probs[:, 0]
    d['prob_up'] = probs[:, 1]
    d['ml_signal'] = np.where(d['prob_up'] >= d['prob_down'], 1, 0)
    d['ml_confidence'] = d[['prob_up', 'prob_down']].max(axis=1)
    return d


# ------------------------------------------
# 7. CANDLE PATTERNS
# ------------------------------------------

def is_bullish_engulfing(curr: pd.Series, prev: pd.Series) -> bool:
    return (
        (prev['close'] < prev['open']) and
        (curr['close'] > curr['open']) and
        (curr['close'] >= prev['open']) and
        (curr['open'] <= prev['close'])
    )

def is_bearish_engulfing(curr: pd.Series, prev: pd.Series) -> bool:
    return (
        (prev['close'] > prev['open']) and
        (curr['close'] < curr['open']) and
        (curr['close'] <= prev['open']) and
        (curr['open'] >= prev['close'])
    )


# ------------------------------------------
# 8. GENERATE 3-CONFIRMATION SIGNALS
# ------------------------------------------

def generate_signals(d: pd.DataFrame,
                     ml_conf_threshold: float = 0.70,
                     vol_threshold: float = 1.0) -> pd.DataFrame:
    d = d.copy()
    d['signal'] = 0

    # Previous swing levels
    d['prev_max_20'] = d['roll_max_20'].shift(1)
    d['prev_min_20'] = d['roll_min_20'].shift(1)

    for i in range(1, len(d)):
        idx = d.index[i]
        row = d.iloc[i]
        prev = d.iloc[i - 1]

        close = row['close']
        ema20 = row['ema20']
        ema50 = row['ema50']
        vwap = row['vwap']
        ml_signal = row['ml_signal']
        ml_conf = row['ml_confidence']
        vol_ratio = row['vol_ratio']
        high = row['high']
        low = row['low']
        prev_max_20 = row['prev_max_20']
        prev_min_20 = row['prev_min_20']

        # 1) ML confirmation
        long_ml_ok = (ml_signal == 1 and ml_conf >= ml_conf_threshold)
        short_ml_ok = (ml_signal == 0 and ml_conf >= ml_conf_threshold)

        # 2) Trend confirmation
        long_trend_ok = (close > ema20 > ema50) and (close > vwap)
        short_trend_ok = (close < ema20 < ema50) and (close < vwap)

        # 3) Price action confirmation
        breakout_up = pd.notna(prev_max_20) and (high > prev_max_20)
        breakout_down = pd.notna(prev_min_20) and (low < prev_min_20)

        bull_engulf = is_bullish_engulfing(row, prev)
        bear_engulf = is_bearish_engulfing(row, prev)

        vol_ok = pd.notna(vol_ratio) and (vol_ratio >= vol_threshold)

        long_pa_ok = (breakout_up or bull_engulf) and vol_ok
        short_pa_ok = (breakout_down or bear_engulf) and vol_ok

        sig = 0
        if long_ml_ok and long_trend_ok and long_pa_ok:
            sig = 1
        elif short_ml_ok and short_trend_ok and short_pa_ok:
            sig = -1

        d.at[idx, 'signal'] = sig

    return d


# ------------------------------------------
# 9. BACKTEST ENGINE
# ------------------------------------------

def backtest(df: pd.DataFrame,
             sl_mult: float = 1.0,
             tp_mult: float = 1.5,
             risk_per_trade: float = 1.0) -> pd.DataFrame:
    d = df.copy()
    d['position'] = 0
    d['pnl'] = 0.0
    d['cum_pnl'] = 0.0

    in_trade = False
    direction = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0

    for i in range(1, len(d)):
        idx = d.index[i]
        row = d.iloc[i]
        prev = d.iloc[i - 1]

        if not in_trade:
            sig = prev['signal']
            atr = prev['atr14']

            if sig == 1 and not np.isnan(atr):
                direction = 1
                entry_price = row['close']
                sl = atr * sl_mult
                tp = atr * tp_mult
                stop_price = entry_price - sl
                target_price = entry_price + tp
                in_trade = True
                d.at[idx, 'position'] = 1

            elif sig == -1 and not np.isnan(atr):
                direction = -1
                entry_price = row['close']
                sl = atr * sl_mult
                tp = atr * tp_mult
                stop_price = entry_price + sl
                target_price = entry_price - tp
                in_trade = True
                d.at[idx, 'position'] = -1

        else:
            high = row['high']
            low = row['low']
            exit_price = None
            trade_pnl = 0.0

            if direction == 1:
                if low <= stop_price:
                    exit_price = stop_price
                elif high >= target_price:
                    exit_price = target_price

            elif direction == -1:
                if high >= stop_price:
                    exit_price = stop_price
                elif low <= target_price:
                    exit_price = target_price

            if exit_price is not None:
                if direction == 1:
                    R = (exit_price - entry_price) / (stop_price - entry_price)
                else:
                    R = (entry_price - exit_price) / (entry_price - stop_price)

                trade_pnl = R * risk_per_trade
                d.at[idx, 'pnl'] = trade_pnl

                in_trade = False
                direction = 0
                entry_price = 0.0
                stop_price = 0.0
                target_price = 0.0
                d.at[idx, 'position'] = 0
            else:
                d.at[idx, 'position'] = direction

        if i > 0:
            d.at[idx, 'cum_pnl'] = d['pnl'].iloc[:i+1].sum()

    return d


# ------------------------------------------
# 10. METRICS
# ------------------------------------------

def compute_metrics(results: pd.DataFrame):
    trades = results[results['pnl'] != 0]
    num_trades = len(trades)

    if num_trades == 0:
        return {
            "num_trades": 0,
            "win_rate": np.nan,
            "avg_R": np.nan,
            "avg_win_R": np.nan,
            "avg_loss_R": np.nan,
            "max_dd": np.nan,
            "sharpe": np.nan
        }

    winners = trades[trades['pnl'] > 0]
    losers = trades[trades['pnl'] < 0]

    win_rate = len(winners) / num_trades
    avg_R = trades['pnl'].mean()
    avg_win_R = winners['pnl'].mean() if len(winners) > 0 else 0
    avg_loss_R = losers['pnl'].mean() if len(losers) > 0 else 0

    equity = results['cum_pnl']
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = drawdown.min()

    rets = trades['pnl']
    if rets.std() > 0:
        sharpe = rets.mean() / rets.std() * np.sqrt(len(rets))
    else:
        sharpe = np.nan

    return {
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_R": avg_R,
        "avg_win_R": avg_win_R,
        "avg_loss_R": avg_loss_R,
        "max_dd": max_dd,
        "sharpe": sharpe
    }


# ------------------------------------------
# 11. PLOTS
# ------------------------------------------

def plot_equity_and_drawdown(results: pd.DataFrame):
    equity = results['cum_pnl']
    peak = equity.cummax()
    drawdown = equity - peak

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(equity.index, equity)
    axes[0].set_title("Equity Curve (R-multiples)")
    axes[0].set_ylabel("Cumulative R")

    axes[1].plot(drawdown.index, drawdown)
    axes[1].set_title("Drawdown (R)")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Time")

    plt.tight_layout()
    return fig


# ------------------------------------------
# 12. PICKLE SAVE / LOAD
# ------------------------------------------

MODEL_PATH = "gnb_model.pkl"
SCALER_PATH = "gnb_scaler.pkl"
FEATURE_PATH = "feature_cols.pkl"

def save_model_artifacts(model, scaler, feature_cols):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(FEATURE_PATH, "wb") as f:
        pickle.dump(feature_cols, f)

def load_model_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURE_PATH)):
        return None, None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURE_PATH, "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols


# ------------------------------------------
# 13. REAL-TIME STYLE PREDICTION
# ------------------------------------------

def predict_realtime_signal(df_raw: pd.DataFrame,
                            model,
                            scaler,
                            feature_cols,
                            ml_conf_threshold: float = 0.70,
                            vol_threshold: float = 1.0):
    d = add_indicators(df_raw.copy())
    d = d.dropna()

    if len(d) == 0:
        return None

    X_all = d[feature_cols].values
    X_all_scaled = scaler.transform(X_all)
    probs = model.predict_proba(X_all_scaled)
    d['prob_down'] = probs[:, 0]
    d['prob_up'] = probs[:, 1]
    d['ml_signal'] = np.where(d['prob_up'] >= d['prob_down'], 1, 0)
    d['ml_confidence'] = d[['prob_up', 'prob_down']].max(axis=1)

    d = generate_signals(d, ml_conf_threshold=ml_conf_threshold,
                         vol_threshold=vol_threshold)

    last = d.iloc[-1]
    return {
        "datetime": last.name,
        "prob_up": float(last['prob_up']),
        "prob_down": float(last['prob_down']),
        "ml_signal": int(last['ml_signal']),       # 1 = UP bias, 0 = DOWN bias
        "ml_confidence": float(last['ml_confidence']),
        "final_signal": int(last['signal'])        # 1 = BUY, -1 = SELL, 0 = NO TRADE
    }


# ------------------------------------------
# 14. STREAMLIT UI
# ------------------------------------------

def main():
    st.title("📊 Intraday 3-Confirmation ML Strategy (15-min, IST)")

    st.sidebar.header("⚙️ Parameters")
    test_size = st.sidebar.slider("Test size (fraction for test set)", 0.1, 0.4, 0.2, 0.05)
    ml_conf_threshold = st.sidebar.slider("ML Confidence Threshold", 0.5, 0.95, 0.70, 0.01)
    vol_threshold = st.sidebar.slider("Volume Ratio Threshold", 0.5, 2.0, 1.0, 0.1)
    sl_mult = st.sidebar.slider("SL ATR Multiplier", 0.5, 3.0, 1.0, 0.1)
    tp_mult = st.sidebar.slider("TP ATR Multiplier", 0.5, 4.0, 1.5, 0.1)

    st.markdown("### 🧪 1. Train Model & Backtest")

    train_file = st.file_uploader(
        "Upload historical 15-min CSV (datetime_ist,timestamp,open,high,low,close,volume)",
        type=["csv"],
        key="train"
    )

    if train_file is not None:
        if st.button("Train & Backtest"):
            try:
                df = load_data_from_csv(train_file)
                st.write("Data preview:", df.head())

                df = add_indicators(df)
                df = create_labels(df, horizon=1)
                df_ml, X, y, feature_cols = build_ml_dataset(df)

                model, scaler, report, acc = train_gnb(X, y, test_size=test_size)
                st.success(f"Model trained. Test Accuracy: {acc:.3f}")

                st.subheader("Classification Report (Test)")
                st.json(report)

                df_ml = apply_model(df_ml, model, scaler, feature_cols)
                df_signals = generate_signals(df_ml,
                                              ml_conf_threshold=ml_conf_threshold,
                                              vol_threshold=vol_threshold)
                results = backtest(df_signals,
                                   sl_mult=sl_mult,
                                   tp_mult=tp_mult,
                                   risk_per_trade=1.0)

                metrics = compute_metrics(results)

                st.subheader("📈 Strategy Metrics")
                st.write(f"Number of trades: {metrics['num_trades']}")
                st.write(f"Win rate: {metrics['win_rate'] * 100 if not np.isnan(metrics['win_rate']) else np.nan:.2f}%")
                st.write(f"Average R per trade: {metrics['avg_R']:.3f}")
                st.write(f"Average Win (R): {metrics['avg_win_R']:.3f}")
                st.write(f"Average Loss (R): {metrics['avg_loss_R']:.3f}")
                st.write(f"Max Drawdown (R): {metrics['max_dd']:.3f}")
                st.write(f"Sharpe (per trade R): {metrics['sharpe']:.3f}")

                st.subheader("📉 Equity Curve & Drawdown")
                fig = plot_equity_and_drawdown(results)
                st.pyplot(fig)

                # Save model
                save_model_artifacts(model, scaler, feature_cols)
                st.success("✅ Model, scaler, and feature list saved as pickle files in current directory.")
            except Exception as e:
                st.error(f"Error during training/backtest: {e}")

    st.markdown("---")
    st.markdown("### ⚡ 2. Use Saved Model on New Data (Backtest or Latest Signal)")

    model, scaler, feature_cols = load_model_artifacts()
    if model is None:
        st.info("No saved model found yet. Train the model in section 1 first.")
    else:
        new_file = st.file_uploader(
            "Upload NEW 15-min CSV (same format) for testing / latest signal",
            type=["csv"],
            key="new"
        )

        if new_file is not None:
            try:
                df_new = load_data_from_csv(new_file)
                st.write("New data preview:", df_new.head())

                # For full backtest on new data using saved model:
                df_new_ind = add_indicators(df_new)
                df_new_ind = create_labels(df_new_ind, horizon=1)  # labels not used for live but fine
                df_ml_new, _, _, _ = build_ml_dataset(df_new_ind)

                df_ml_new = apply_model(df_ml_new, model, scaler, feature_cols)
                df_signals_new = generate_signals(df_ml_new,
                                                  ml_conf_threshold=ml_conf_threshold,
                                                  vol_threshold=vol_threshold)
                results_new = backtest(df_signals_new,
                                       sl_mult=sl_mult,
                                       tp_mult=tp_mult,
                                       risk_per_trade=1.0)

                metrics_new = compute_metrics(results_new)

                st.subheader("📈 Metrics on New Data (Using Saved Model)")
                st.write(f"Number of trades: {metrics_new['num_trades']}")
                st.write(f"Win rate: {metrics_new['win_rate'] * 100 if not np.isnan(metrics_new['win_rate']) else np.nan:.2f}%")
                st.write(f"Average R per trade: {metrics_new['avg_R']:.3f}")
                st.write(f"Average Win (R): {metrics_new['avg_win_R']:.3f}")
                st.write(f"Average Loss (R): {metrics_new['avg_loss_R']:.3f}")
                st.write(f"Max Drawdown (R): {metrics_new['max_dd']:.3f}")
                st.write(f"Sharpe (per trade R): {metrics_new['sharpe']:.3f}")

                fig2 = plot_equity_and_drawdown(results_new)
                st.pyplot(fig2)

                # "Realtime" style last candle signal
                st.subheader("🚨 Latest Candle Signal (Realtime Style)")
                rt_info = predict_realtime_signal(df_new, model, scaler, feature_cols,
                                                  ml_conf_threshold=ml_conf_threshold,
                                                  vol_threshold=vol_threshold)
                if rt_info is None:
                    st.write("Not enough data to compute signal.")
                else:
                    st.json({
                        "datetime_ist": str(rt_info["datetime"]),
                        "prob_up": rt_info["prob_up"],
                        "prob_down": rt_info["prob_down"],
                        "ml_signal (1=UP,0=DOWN)": rt_info["ml_signal"],
                        "ml_confidence": rt_info["ml_confidence"],
                        "final_signal (1=BUY,-1=SELL,0=NO TRADE)": rt_info["final_signal"],
                    })

            except Exception as e:
                st.error(f"Error using saved model on new data: {e}")


if __name__ == "__main__":
    main()
