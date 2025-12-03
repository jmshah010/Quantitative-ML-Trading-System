# ==========================================
# PROFESSIONAL INTRADAY ML TRADING STRATEGY
# Enhanced with Institutional-Grade Features
# ==========================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import streamlit as st

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ------------------------------------------
# 1. DATA LOADING & VALIDATION
# ------------------------------------------

def load_data_from_csv(file) -> pd.DataFrame:
    """Load and validate CSV data with robust error handling"""
    try:
        df = pd.read_csv(file)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Handle datetime column variations
        datetime_col = None
        for col in ["datetime_ist", "atetime_ist", "datetime", "date"]:
            if col in df.columns:
                datetime_col = col
                break
        
        if datetime_col is None:
            raise ValueError("CSV must have a datetime column (datetime_ist, datetime, or date)")
        
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col).reset_index(drop=True)
        df.set_index(datetime_col, inplace=True)
        df.index.name = "datetime_ist"
        
        # Validate required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Remove duplicates and handle missing values
        df = df[~df.index.duplicated(keep='first')]
        df = df.dropna(subset=required)
        
        return df
    
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


# ------------------------------------------
# 2. ADVANCED TECHNICAL INDICATORS
# ------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive technical indicators"""
    d = df.copy()
    
    # === Moving Averages ===
    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()
    d["sma20"] = d["close"].rolling(20).mean()
    
    # === VWAP (reset daily for intraday) ===
    cum_vol = d["volume"].cumsum()
    cum_vp = (d["close"] * d["volume"]).cumsum()
    d["vwap"] = cum_vp / cum_vol.replace(0, np.nan)
    
    # === Volatility Indicators ===
    # ATR
    high_low = d["high"] - d["low"]
    high_close_prev = (d["high"] - d["close"].shift(1)).abs()
    low_close_prev = (d["low"] - d["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(window=14).mean()
    d["atr_pct"] = (d["atr14"] / d["close"]) * 100
    
    # Bollinger Bands
    d["bb_middle"] = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["bb_upper"] = d["bb_middle"] + (2 * bb_std)
    d["bb_lower"] = d["bb_middle"] - (2 * bb_std)
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["bb_middle"]
    d["bb_position"] = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"])
    
    # === Momentum Indicators ===
    # RSI
    delta = d["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=d.index).rolling(14).mean()
    roll_down = pd.Series(loss, index=d.index).rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    d["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    
    # MACD
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    
    # Stochastic
    low14 = d["low"].rolling(14).min()
    high14 = d["high"].rolling(14).max()
    d["stoch_k"] = 100 * (d["close"] - low14) / (high14 - low14)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()
    
    # Rate of Change
    d["roc"] = ((d["close"] - d["close"].shift(10)) / d["close"].shift(10)) * 100
    
    # === Volume Indicators ===
    d["vol_ma20"] = d["volume"].rolling(20).mean()
    d["vol_ma50"] = d["volume"].rolling(50).mean()
    d["vol_ratio"] = d["volume"] / d["vol_ma20"]
    d["vol_trend"] = d["vol_ma20"] / d["vol_ma50"]
    
    # On-Balance Volume
    d["obv"] = (np.sign(d["close"].diff()) * d["volume"]).fillna(0).cumsum()
    d["obv_ema"] = d["obv"].ewm(span=20, adjust=False).mean()
    
    # === Price Action Features ===
    d["body"] = d["close"] - d["open"]
    d["range"] = d["high"] - d["low"]
    d["upper_shadow"] = d["high"] - d[["open", "close"]].max(axis=1)
    d["lower_shadow"] = d[["open", "close"]].min(axis=1) - d["low"]
    d["body_ratio"] = d["body"].abs() / d["range"].replace(0, np.nan)
    
    # === Support/Resistance ===
    d["roll_max_20"] = d["high"].rolling(20).max()
    d["roll_min_20"] = d["low"].rolling(20).min()
    d["roll_max_50"] = d["high"].rolling(50).max()
    d["roll_min_50"] = d["low"].rolling(50).min()
    
    # Distance from swing points
    d["dist_from_high"] = (d["close"] - d["roll_max_20"]) / d["close"]
    d["dist_from_low"] = (d["close"] - d["roll_min_20"]) / d["close"]
    
    # === Trend Indicators ===
    d["adx"] = calculate_adx(d)
    d["trend_strength"] = np.where(d["ema20"] > d["ema50"], 1, -1)
    
    return d


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    
    return adx


# ------------------------------------------
# 3. ADVANCED LABELING WITH MULTIPLE METHODS
# ------------------------------------------

def create_labels(df: pd.DataFrame, 
                  horizon: int = 1,
                  return_threshold_pct: float = 0.2,
                  method: str = "threshold") -> pd.DataFrame:
    """
    Create labels with multiple methods:
    - threshold: classify based on return threshold
    - tercile: classify based on return terciles
    - quintile: classify based on return quintiles
    """
    d = df.copy()
    d["future_close"] = d["close"].shift(-horizon)
    d["future_ret_pct"] = (d["future_close"] - d["close"]) / d["close"] * 100.0
    
    if method == "threshold":
        up = d["future_ret_pct"] > return_threshold_pct
        down = d["future_ret_pct"] < -return_threshold_pct
        d["direction"] = np.where(up, 1, np.where(down, 0, np.nan))
    
    elif method == "tercile":
        terciles = d["future_ret_pct"].quantile([0.33, 0.67])
        d["direction"] = np.where(
            d["future_ret_pct"] > terciles[0.67], 1,
            np.where(d["future_ret_pct"] < terciles[0.33], 0, np.nan)
        )
    
    elif method == "quintile":
        quintiles = d["future_ret_pct"].quantile([0.2, 0.8])
        d["direction"] = np.where(
            d["future_ret_pct"] > quintiles[0.8], 1,
            np.where(d["future_ret_pct"] < quintiles[0.2], 0, np.nan)
        )
    
    return d


# ------------------------------------------
# 4. FEATURE ENGINEERING & SELECTION
# ------------------------------------------

def build_ml_dataset(df: pd.DataFrame):
    """Build feature matrix with engineered features"""
    feature_cols = [
        # Price & MAs
        "close", "ema9", "ema20", "ema50", "vwap",
        # Volatility
        "atr14", "atr_pct", "bb_width", "bb_position",
        # Momentum
        "rsi14", "macd", "macd_hist", "stoch_k", "stoch_d", "roc",
        # Volume
        "vol_ratio", "vol_trend", "obv_ema",
        # Price Action
        "body", "range", "body_ratio",
        # Support/Resistance
        "dist_from_high", "dist_from_low",
        # Trend
        "adx", "trend_strength"
    ]
    
    # Add interaction features
    df_eng = df.copy()
    df_eng["rsi_bb"] = df_eng["rsi14"] * df_eng["bb_position"]
    df_eng["vol_momentum"] = df_eng["vol_ratio"] * df_eng["roc"]
    df_eng["trend_vol"] = df_eng["trend_strength"] * df_eng["vol_ratio"]
    
    feature_cols.extend(["rsi_bb", "vol_momentum", "trend_vol"])
    
    d = df_eng.dropna(subset=feature_cols + ["direction"]).copy()
    X = d[feature_cols].values
    y = d["direction"].values.astype(int)
    
    return d, X, y, feature_cols


# ------------------------------------------
# 5. MULTI-MODEL TRAINING WITH CV
# ------------------------------------------

def train_models(X, y, test_size=0.2, var_smoothing=1e-9):
    """Train multiple models and compare performance"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "GaussianNB": GaussianNB(var_smoothing=var_smoothing),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        except:
            auc = np.nan
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "model": model,
            "accuracy": acc,
            "auc": auc,
            "report": report,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "y_test": y_test
        }
    
    return results, scaler


# ------------------------------------------
# 6. ENHANCED SIGNAL GENERATION
# ------------------------------------------

def generate_signals_advanced(d: pd.DataFrame,
                              model,
                              scaler,
                              feature_cols,
                              ml_conf_threshold: float = 0.65,
                              vol_threshold: float = 1.0,
                              use_regime_filter: bool = True) -> pd.DataFrame:
    """Advanced signal generation with regime detection"""
    d = d.copy()
    
    # Apply ML model
    X_all = d[feature_cols].values
    X_all_scaled = scaler.transform(X_all)
    probs = model.predict_proba(X_all_scaled)
    d["prob_down"] = probs[:, 0]
    d["prob_up"] = probs[:, 1]
    d["ml_signal"] = np.where(d["prob_up"] >= d["prob_down"], 1, 0)
    d["ml_confidence"] = d[["prob_up", "prob_down"]].max(axis=1)
    
    # Regime detection
    if use_regime_filter:
        d["regime"] = detect_market_regime(d)
    else:
        d["regime"] = "trending"
    
    d["signal"] = 0
    d["signal_type"] = "NONE"
    
    for i in range(1, len(d)):
        idx = d.index[i]
        row = d.iloc[i]
        prev = d.iloc[i - 1]
        
        # ML conditions
        long_ml = row["ml_signal"] == 1 and row["ml_confidence"] >= ml_conf_threshold
        short_ml = row["ml_signal"] == 0 and row["ml_confidence"] >= ml_conf_threshold
        
        # Trend conditions
        long_trend = (row["close"] > row["ema20"] > row["ema50"]) and (row["close"] > row["vwap"])
        short_trend = (row["close"] < row["ema20"] < row["ema50"]) and (row["close"] < row["vwap"])
        
        # Momentum conditions
        long_momentum = row["rsi14"] > 50 and row["macd_hist"] > 0
        short_momentum = row["rsi14"] < 50 and row["macd_hist"] < 0
        
        # Volume conditions
        strong_vol = pd.notna(row["vol_ratio"]) and row["vol_ratio"] >= vol_threshold
        
        # ADX filter (trend strength)
        strong_trend = pd.notna(row["adx"]) and row["adx"] > 25
        
        # Regime filter
        regime_ok = row["regime"] == "trending" if use_regime_filter else True
        
        # Generate signals with multiple confirmation levels
        if long_ml and long_trend and strong_vol and strong_trend and regime_ok:
            d.at[idx, "signal"] = 1
            d.at[idx, "signal_type"] = "STRONG_LONG"
        elif long_ml and long_trend and (strong_vol or long_momentum):
            d.at[idx, "signal"] = 1
            d.at[idx, "signal_type"] = "MODERATE_LONG"
        elif short_ml and short_trend and strong_vol and strong_trend and regime_ok:
            d.at[idx, "signal"] = -1
            d.at[idx, "signal_type"] = "STRONG_SHORT"
        elif short_ml and short_trend and (strong_vol or short_momentum):
            d.at[idx, "signal"] = -1
            d.at[idx, "signal_type"] = "MODERATE_SHORT"
    
    return d


def detect_market_regime(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """Detect market regime: trending vs ranging"""
    adx = df["adx"]
    bb_width = df["bb_width"]
    
    regime = np.where(
        (adx > 25) | (bb_width > bb_width.rolling(window).mean()),
        "trending",
        "ranging"
    )
    
    return pd.Series(regime, index=df.index)


# ------------------------------------------
# 7. ADVANCED BACKTESTING WITH POSITION SIZING
# ------------------------------------------

def backtest_advanced(df: pd.DataFrame,
                     initial_capital: float = 100000,
                     sl_mult: float = 1.0,
                     tp_mult: float = 1.5,
                     risk_per_trade_pct: float = 1.0,
                     use_trailing_stop: bool = False,
                     trailing_mult: float = 0.5):
    """Enhanced backtest with position sizing and trailing stops"""
    d = df.copy()
    d["position"] = 0
    d["pnl"] = 0.0
    d["pnl_pct"] = 0.0
    d["equity"] = initial_capital
    d["action"] = "FLAT"
    d["shares"] = 0
    
    capital = initial_capital
    in_trade = False
    direction = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    trailing_stop = 0.0
    shares = 0
    entry_time = None
    
    trades = []
    
    for i in range(1, len(d)):
        idx = d.index[i]
        row = d.iloc[i]
        prev = d.iloc[i - 1]
        
        d.at[idx, "action"] = "FLAT"
        d.at[idx, "equity"] = capital
        
        if not in_trade:
            sig = prev["signal"]
            atr = prev["atr14"]
            
            if sig != 0 and not np.isnan(atr):
                # Position sizing based on risk
                risk_amount = capital * (risk_per_trade_pct / 100)
                sl = atr * sl_mult
                shares = int(risk_amount / sl)
                
                if shares > 0:
                    direction = sig
                    entry_price = row["close"]
                    
                    if direction == 1:
                        stop_price = entry_price - sl
                        target_price = entry_price + (atr * tp_mult)
                        trailing_stop = stop_price
                        d.at[idx, "action"] = "BUY"
                    else:
                        stop_price = entry_price + sl
                        target_price = entry_price - (atr * tp_mult)
                        trailing_stop = stop_price
                        d.at[idx, "action"] = "SELL"
                    
                    in_trade = True
                    entry_time = idx
                    d.at[idx, "position"] = direction
                    d.at[idx, "shares"] = shares
        
        else:
            high = row["high"]
            low = row["low"]
            close = row["close"]
            exit_price = None
            exit_reason = ""
            
            d.at[idx, "position"] = direction
            d.at[idx, "shares"] = shares
            d.at[idx, "action"] = "HOLD"
            
            # Update trailing stop
            if use_trailing_stop:
                if direction == 1:
                    new_trailing = close - (row["atr14"] * trailing_mult)
                    trailing_stop = max(trailing_stop, new_trailing)
                else:
                    new_trailing = close + (row["atr14"] * trailing_mult)
                    trailing_stop = min(trailing_stop, new_trailing)
            
            # Check exit conditions
            if direction == 1:
                if low <= (trailing_stop if use_trailing_stop else stop_price):
                    exit_price = trailing_stop if use_trailing_stop else stop_price
                    exit_reason = "TRAILING_STOP" if use_trailing_stop else "STOP_LOSS"
                elif high >= target_price:
                    exit_price = target_price
                    exit_reason = "TAKE_PROFIT"
            else:
                if high >= (trailing_stop if use_trailing_stop else stop_price):
                    exit_price = trailing_stop if use_trailing_stop else stop_price
                    exit_reason = "TRAILING_STOP" if use_trailing_stop else "STOP_LOSS"
                elif low <= target_price:
                    exit_price = target_price
                    exit_reason = "TAKE_PROFIT"
            
            if exit_price is not None:
                # Calculate P&L
                if direction == 1:
                    trade_pnl = (exit_price - entry_price) * shares
                    trade_pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    d.at[idx, "action"] = "SELL"
                else:
                    trade_pnl = (entry_price - exit_price) * shares
                    trade_pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                    d.at[idx, "action"] = "BUY"
                
                capital += trade_pnl
                d.at[idx, "pnl"] = trade_pnl
                d.at[idx, "pnl_pct"] = trade_pnl_pct
                d.at[idx, "equity"] = capital
                d.at[idx, "position"] = 0
                d.at[idx, "shares"] = 0
                
                trades.append({
                    "side": "LONG" if direction == 1 else "SHORT",
                    "entry_time": entry_time,
                    "exit_time": idx,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": shares,
                    "pnl": trade_pnl,
                    "pnl_pct": trade_pnl_pct,
                    "exit_reason": exit_reason,
                    "duration": (idx - entry_time).total_seconds() / 60 if isinstance(idx, pd.Timestamp) else 0,
                })
                
                in_trade = False
                direction = 0
                shares = 0
    
    # Calculate cumulative equity
    d["equity"] = initial_capital + d["pnl"].cumsum()
    
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()
    
    return d, trades_df


# ------------------------------------------
# 8. COMPREHENSIVE METRICS
# ------------------------------------------

def compute_comprehensive_metrics(trades: pd.DataFrame, results: pd.DataFrame, initial_capital: float = 100000):
    """Calculate institutional-grade performance metrics"""
    if trades is None or trades.empty:
        return {}
    
    # Basic metrics
    num_trades = len(trades)
    winners = trades[trades["pnl"] > 0]
    losers = trades[trades["pnl"] < 0]
    
    win_rate = len(winners) / num_trades if num_trades > 0 else 0
    avg_win = winners["pnl"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl"].mean() if len(losers) > 0 else 0
    
    # Return metrics
    total_return = trades["pnl"].sum()
    total_return_pct = (total_return / initial_capital) * 100
    
    # Risk metrics
    equity_curve = results["equity"].dropna()
    returns = equity_curve.pct_change().dropna()
    
    volatility = returns.std() * np.sqrt(252 * 26)  # Annualized for 15-min bars
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 26)) if returns.std() > 0 else 0
    
    # Drawdown metrics
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak * 100
    max_dd = drawdown.min()
    
    # Calculate max consecutive losses
    loss_streak = 0
    max_loss_streak = 0
    for pnl in trades["pnl"]:
        if pnl < 0:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0
    
    # Profit factor
    gross_profit = winners["pnl"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl"].sum()) if len(losers) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Calmar ratio
    calmar = (total_return_pct / abs(max_dd)) if max_dd != 0 else 0
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252 * 26)
    sortino = (returns.mean() / downside_std * np.sqrt(252 * 26)) if downside_std > 0 else 0
    
    return {
        "num_trades": num_trades,
        "num_winners": len(winners),
        "num_losers": len(losers),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_return": total_return,
        "total_return_pct": total_return_pct,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_loss_streak": max_loss_streak,
        "volatility": volatility,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


# ------------------------------------------
# 9. PROFESSIONAL VISUALIZATIONS
# ------------------------------------------

def plot_comprehensive_dashboard(results: pd.DataFrame, trades: pd.DataFrame, metrics: dict):
    """Create comprehensive performance dashboard"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Equity curve with drawdown
    ax1 = fig.add_subplot(gs[0, :])
    equity = results["equity"].dropna()
    ax1.plot(equity.index, equity.values, linewidth=2, label="Equity", color='#2E86AB')
    ax1.fill_between(equity.index, equity.values, equity.values.min(), alpha=0.3, color='#2E86AB')
    ax1.set_title("Equity Curve", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Capital ($)", fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color='#E63946', alpha=0.6)
    ax2.plot(drawdown.index, drawdown.values, color='#E63946', linewidth=2)
    ax2.set_title("Drawdown (%)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Returns distribution
    ax3 = fig.add_subplot(gs[2, 0])
    if not trades.empty:
        ax3.hist(trades["pnl_pct"], bins=30, color='#06A77D', alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_title("Returns Distribution", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Return (%)", fontsize=10)
        ax3.set_ylabel("Frequency", fontsize=10)
        ax3.grid(True, alpha=0.3)
    
    # 4. Win/Loss analysis
    ax4 = fig.add_subplot(gs[2, 1])
    if not trades.empty:
        win_loss_data = [metrics["num_winners"], metrics["num_losers"]]
        colors_pie = ['#06A77D', '#E63946']
        ax4.pie(win_loss_data, labels=['Winners', 'Losers'], autopct='%1.1f%%',
                colors=colors_pie, startangle=90, textprops={'fontsize': 10})
        ax4.set_title(f"Win Rate: {metrics['win_rate']:.1%}", fontsize=12, fontweight='bold')
    
    # 5. Cumulative P&L
    ax5 = fig.add_subplot(gs[2, 2])
    if not trades.empty:
        cum_pnl = trades["cum_pnl"]
        ax5.plot(range(len(cum_pnl)), cum_pnl, linewidth=2, color='#F77F00', marker='o', markersize=3)
        ax5.fill_between(range(len(cum_pnl)), cum_pnl, 0, alpha=0.3, color='#F77F00')
        ax5.set_title("Cumulative P&L by Trade", fontsize=12, fontweight='bold')
        ax5.set_xlabel("Trade Number", fontsize=10)
        ax5.set_ylabel("Cumulative P&L ($)", fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 6. Monthly returns heatmap
    ax6 = fig.add_subplot(gs[3, :2])
    if not trades.empty and hasattr(trades["exit_time"].iloc[0], 'month'):
        trades_copy = trades.copy()
        trades_copy["year"] = pd.to_datetime(trades_copy["exit_time"]).dt.year
        trades_copy["month"] = pd.to_datetime(trades_copy["exit_time"]).dt.month
        monthly_returns = trades_copy.groupby(["year", "month"])["pnl"].sum().unstack(fill_value=0)
        
        if not monthly_returns.empty:
            sns.heatmap(monthly_returns, annot=True, fmt='.0f', cmap='RdYlGn', 
                       center=0, ax=ax6, cbar_kws={'label': 'P&L ($)'})
            ax6.set_title("Monthly P&L Heatmap", fontsize=12, fontweight='bold')
            ax6.set_xlabel("Month", fontsize=10)
            ax6.set_ylabel("Year", fontsize=10)
    
    # 7. Performance metrics table
    ax7 = fig.add_subplot(gs[3, 2])
    ax7.axis('off')
    
    metrics_text = f"""
    PERFORMANCE METRICS
    {'='*35}
    Total Trades: {metrics['num_trades']}
    Win Rate: {metrics['win_rate']:.1%}
    
    Total Return: ${metrics['total_return']:,.2f}
    Return %: {metrics['total_return_pct']:.2f}%
    
    Sharpe Ratio: {metrics['sharpe']:.2f}
    Sortino Ratio: {metrics['sortino']:.2f}
    Calmar Ratio: {metrics['calmar']:.2f}
    
    Max Drawdown: {metrics['max_dd']:.2f}%
    Profit Factor: {metrics['profit_factor']:.2f}
    
    Avg Win: ${metrics['avg_win']:,.2f}
    Avg Loss: ${metrics['avg_loss']:,.2f}
    Expectancy: ${metrics['expectancy']:,.2f}
    
    Max Loss Streak: {metrics['max_loss_streak']}
    """
    
    ax7.text(0.1, 0.95, metrics_text, transform=ax7.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle("ML Trading Strategy - Comprehensive Performance Dashboard", 
                 fontsize=16, fontweight='bold', y=0.995)
    
    return fig


def plot_signal_analysis(results: pd.DataFrame):
    """Plot signal analysis and ML predictions"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # Plot recent data
    recent = results.iloc[-500:] if len(results) > 500 else results
    
    # 1. Price with signals
    ax1 = axes[0]
    ax1.plot(recent.index, recent["close"], label="Close", linewidth=1.5, color='black')
    ax1.plot(recent.index, recent["ema20"], label="EMA20", linewidth=1, alpha=0.7)
    ax1.plot(recent.index, recent["ema50"], label="EMA50", linewidth=1, alpha=0.7)
    ax1.plot(recent.index, recent["vwap"], label="VWAP", linewidth=1, linestyle='--', alpha=0.7)
    
    # Mark buy/sell signals
    buys = recent[recent["action"] == "BUY"]
    sells = recent[recent["action"] == "SELL"]
    
    ax1.scatter(buys.index, buys["close"], color='green', marker='^', 
                s=150, label='Buy', zorder=5, edgecolors='black', linewidths=1.5)
    ax1.scatter(sells.index, sells["close"], color='red', marker='v', 
                s=150, label='Sell', zorder=5, edgecolors='black', linewidths=1.5)
    
    ax1.set_title("Price Action with Trading Signals", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Price", fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. ML Confidence
    ax2 = axes[1]
    ax2.plot(recent.index, recent["prob_up"], label="Prob Up", color='green', linewidth=1.5)
    ax2.plot(recent.index, recent["prob_down"], label="Prob Down", color='red', linewidth=1.5)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(recent.index, recent["prob_up"], recent["prob_down"], 
                     where=recent["prob_up"] >= recent["prob_down"],
                     color='green', alpha=0.2)
    ax2.fill_between(recent.index, recent["prob_up"], recent["prob_down"], 
                     where=recent["prob_up"] < recent["prob_down"],
                     color='red', alpha=0.2)
    ax2.set_title("ML Model Prediction Confidence", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Probability", fontsize=11)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Volume with signals
    ax3 = axes[2]
    colors = ['green' if row["signal"] == 1 else 'red' if row["signal"] == -1 else 'gray' 
              for _, row in recent.iterrows()]
    ax3.bar(recent.index, recent["volume"], color=colors, alpha=0.6, width=0.8)
    ax3.plot(recent.index, recent["vol_ma20"], label="Vol MA20", color='blue', linewidth=2)
    ax3.set_title("Volume Analysis", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Volume", fontsize=11)
    ax3.set_xlabel("Time", fontsize=11)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_cols):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:20]  # Top 20
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(indices)), importance[indices], color='#2E86AB', alpha=0.8)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_cols[i] for i in indices])
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title("Top 20 Feature Importances", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        return fig
    return None


# ------------------------------------------
# 10. STREAMLIT APPLICATION
# ------------------------------------------

def main():
    st.set_page_config(page_title="ML Trading Strategy", layout="wide", 
                       initial_sidebar_state="expanded")
    
    st.title("🚀 Professional ML-Based Intraday Trading Strategy")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV Data", type=["csv"])
    
    st.sidebar.markdown("### ML Model Settings")
    model_choice = st.sidebar.selectbox("Select Model", 
                                        ["GaussianNB", "LogisticRegression", 
                                         "RandomForest", "GradientBoosting"])
    
    var_smoothing = st.sidebar.number_input("Var Smoothing (NB only)", 
                                            min_value=1e-12, max_value=1e-6,
                                            value=1e-9, format="%.2e")
    
    st.sidebar.markdown("### Labeling Settings")
    label_method = st.sidebar.selectbox("Label Method", 
                                        ["threshold", "tercile", "quintile"])
    return_threshold = st.sidebar.slider("Return Threshold (%)", 
                                         min_value=0.1, max_value=2.0, 
                                         value=0.2, step=0.1)
    horizon = st.sidebar.slider("Prediction Horizon (bars)", 
                                min_value=1, max_value=10, value=1)
    
    st.sidebar.markdown("### Signal Generation")
    ml_confidence = st.sidebar.slider("ML Confidence Threshold", 
                                      min_value=0.5, max_value=0.95, 
                                      value=0.65, step=0.05)
    vol_threshold = st.sidebar.slider("Volume Threshold", 
                                      min_value=0.5, max_value=2.0, 
                                      value=1.0, step=0.1)
    use_regime = st.sidebar.checkbox("Use Regime Filter", value=True)
    
    st.sidebar.markdown("### Backtest Settings")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", 
                                              min_value=10000, max_value=1000000,
                                              value=100000, step=10000)
    sl_mult = st.sidebar.slider("Stop Loss (ATR Multiple)", 
                                min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    tp_mult = st.sidebar.slider("Take Profit (ATR Multiple)", 
                                min_value=1.0, max_value=5.0, value=1.5, step=0.1)
    risk_pct = st.sidebar.slider("Risk Per Trade (%)", 
                                 min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    use_trailing = st.sidebar.checkbox("Use Trailing Stop", value=False)
    trailing_mult = st.sidebar.slider("Trailing Stop (ATR Multiple)", 
                                      min_value=0.3, max_value=2.0, 
                                      value=0.5, step=0.1)
    
    run_button = st.sidebar.button("▶️ RUN STRATEGY", type="primary")
    
    # Main content
    if uploaded_file is not None:
        try:
            with st.spinner("📊 Loading data..."):
                df = load_data_from_csv(uploaded_file)
            
            st.success(f"✅ Data loaded: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
            
            # Display data sample
            with st.expander("🔍 View Data Sample"):
                st.dataframe(df.head(10), use_container_width=True)
            
            if run_button:
                # Step 1: Add indicators
                with st.spinner("🔧 Computing technical indicators..."):
                    df = add_indicators(df)
                st.success("✅ Technical indicators computed")
                
                # Step 2: Create labels
                with st.spinner("🏷️ Creating labels..."):
                    df = create_labels(df, horizon=horizon, 
                                      return_threshold_pct=return_threshold,
                                      method=label_method)
                st.success("✅ Labels created")
                
                # Step 3: Build ML dataset
                with st.spinner("🧮 Building ML dataset..."):
                    df_ml, X, y, feature_cols = build_ml_dataset(df)
                
                st.success(f"✅ ML dataset built: {len(X)} samples, {len(feature_cols)} features")
                
                # Display class distribution
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Samples", len(y))
                    st.metric("Features", len(feature_cols))
                with col2:
                    st.metric("Up Samples", int(y.sum()))
                    st.metric("Down Samples", int(len(y) - y.sum()))
                
                # Step 4: Train models
                with st.spinner("🤖 Training ML models..."):
                    results_dict, scaler = train_models(X, y, var_smoothing=var_smoothing)
                
                st.success("✅ Models trained")
                
                # Display model comparison
                st.markdown("### 📊 Model Comparison")
                model_metrics = []
                for name, res in results_dict.items():
                    model_metrics.append({
                        "Model": name,
                        "Accuracy": f"{res['accuracy']:.3f}",
                        "AUC": f"{res['auc']:.3f}" if not np.isnan(res['auc']) else "N/A",
                        "Precision (Up)": f"{res['report'].get('1', {}).get('precision', 0):.3f}",
                        "Recall (Up)": f"{res['report'].get('1', {}).get('recall', 0):.3f}",
                        "F1 (Up)": f"{res['report'].get('1', {}).get('f1-score', 0):.3f}"
                    })
                
                st.dataframe(pd.DataFrame(model_metrics), use_container_width=True)
                
                # Select best model
                selected_model = results_dict[model_choice]["model"]
                
                # Step 5: Generate signals
                with st.spinner("📡 Generating trading signals..."):
                    df_signals = generate_signals_advanced(
                        df_ml, selected_model, scaler, feature_cols,
                        ml_conf_threshold=ml_confidence,
                        vol_threshold=vol_threshold,
                        use_regime_filter=use_regime
                    )
                
                signal_counts = df_signals["signal"].value_counts()
                st.success(f"✅ Signals generated: {signal_counts.get(1, 0)} Longs, {signal_counts.get(-1, 0)} Shorts")
                
                # Step 6: Backtest
                with st.spinner("⚡ Running backtest..."):
                    results, trades = backtest_advanced(
                        df_signals,
                        initial_capital=initial_capital,
                        sl_mult=sl_mult,
                        tp_mult=tp_mult,
                        risk_per_trade_pct=risk_pct,
                        use_trailing_stop=use_trailing,
                        trailing_mult=trailing_mult
                    )
                
                # Step 7: Compute metrics
                metrics = compute_comprehensive_metrics(trades, results, initial_capital)
                
                st.success("✅ Backtest completed")
                
                # Display key metrics
                st.markdown("### 📈 Performance Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"${metrics['total_return']:,.2f}",
                             delta=f"{metrics['total_return_pct']:.2f}%")
                    st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
                    st.metric("Sortino Ratio", f"{metrics['sortino']:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{metrics['max_dd']:.2f}%")
                    st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                with col4:
                    st.metric("Total Trades", metrics['num_trades'])
                    st.metric("Expectancy", f"${metrics['expectancy']:,.2f}")
                
                # Visualizations
                st.markdown("### 📊 Performance Dashboard")
                with st.spinner("Creating visualizations..."):
                    fig_dashboard = plot_comprehensive_dashboard(results, trades, metrics)
                    st.pyplot(fig_dashboard)
                
                st.markdown("### 🎯 Signal Analysis")
                fig_signals = plot_signal_analysis(results)
                st.pyplot(fig_signals)
                
                # Feature importance
                if model_choice in ["RandomForest", "GradientBoosting"]:
                    st.markdown("### 🔍 Feature Importance")
                    fig_importance = plot_feature_importance(selected_model, feature_cols)
                    if fig_importance:
                        st.pyplot(fig_importance)
                
                # Trade log
                st.markdown("### 📋 Trade Log")
                if not trades.empty:
                    trades_display = trades.copy()
                    trades_display["pnl"] = trades_display["pnl"].apply(lambda x: f"${x:,.2f}")
                    trades_display["pnl_pct"] = trades_display["pnl_pct"].apply(lambda x: f"{x:.2f}%")
                    trades_display["entry_price"] = trades_display["entry_price"].apply(lambda x: f"${x:.2f}")
                    trades_display["exit_price"] = trades_display["exit_price"].apply(lambda x: f"${x:.2f}")
                    st.dataframe(trades_display, use_container_width=True)
                else:
                    st.warning("No trades executed during backtest period")
                
                # Download results
                st.markdown("### 💾 Download Results")
                col1, col2 = st.columns(2)
                with col1:
                    csv_results = results.to_csv()
                    st.download_button("📥 Download Results CSV", csv_results, 
                                      "trading_results.csv", "text/csv")
                with col2:
                    if not trades.empty:
                        csv_trades = trades.to_csv()
                        st.download_button("📥 Download Trades CSV", csv_trades,
                                          "trades_log.csv", "text/csv")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👈 Please upload a CSV file to begin")
        st.markdown("""
        ### 📖 Instructions
        
        1. **Upload Data**: Upload your intraday OHLCV CSV file
        2. **Configure**: Adjust ML model, labeling, and backtest parameters
        3. **Run**: Click 'RUN STRATEGY' to execute the complete pipeline
        4. **Analyze**: Review performance metrics, charts, and trade log
        5. **Download**: Export results for further analysis
        
        ### 📋 Required CSV Format
        Your CSV must contain these columns:
        - `datetime_ist` or `datetime` (timestamp)
        - `open`, `high`, `low`, `close` (price data)
        - `volume` (trading volume)
        
        ### ⚠️ Note
        This is a backtesting tool for educational purposes. Past performance does not guarantee future results.
        Always validate strategies on out-of-sample data before live trading.
        """)


if __name__ == "__main__":
    main()