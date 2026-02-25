"""
Prediction engine for Bubble Radar v2.
Computes technical indicators (RSI, MACD, Moving Averages), engineers
features, and trains a Prophet model for 60-day price forecasting.
Sentiment acts as a weighted external regressor.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import PREDICTION, THRESHOLDS

logger = logging.getLogger(__name__)

# Suppress Prophet / cmdstanpy noise
for _name in ("prophet", "cmdstanpy"):
    logging.getLogger(_name).setLevel(logging.WARNING)

try:
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except ImportError:
    _PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed.  pip install prophet")


# ═══════════════════════════════════════════════════════════════════════════
# Technical Indicators
# ═══════════════════════════════════════════════════════════════════════════

def compute_rsi(prices: pd.Series, period: int = PREDICTION.RSI_PERIOD) -> pd.Series:
    """Relative Strength Index (0-100)."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    return 100 - (100 / (1 + rs))


def compute_macd(
    prices: pd.Series,
    fast: int = PREDICTION.MACD_FAST,
    slow: int = PREDICTION.MACD_SLOW,
    signal: int = PREDICTION.MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_moving_averages(
    prices: pd.Series,
    short: int = PREDICTION.MA_SHORT,
    long: int = PREDICTION.MA_LONG,
) -> tuple[pd.Series, pd.Series]:
    """Short-term and long-term simple moving averages."""
    ma_short = prices.rolling(window=short, min_periods=1).mean()
    ma_long = prices.rolling(window=long, min_periods=1).mean()
    return ma_short, ma_long


def compute_technical_indicators(price_df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Compute all technical indicators for display purposes.
    Returns a dict keyed by indicator name.
    """
    close = price_df["Close"]
    rsi = compute_rsi(close)
    macd_line, signal_line, histogram = compute_macd(close)
    ma_short, ma_long = compute_moving_averages(close)

    return {
        "rsi": rsi,
        "macd_line": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
        "ma_short": ma_short,
        "ma_long": ma_long,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════

def build_features(
    price_df: pd.DataFrame,
    sentiment_ts: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the feature DataFrame expected by Prophet.

    Columns produced:
        ds, y, rsi_norm, macd_norm, ma_ratio, sentiment
    """
    df = pd.DataFrame()
    df["ds"] = pd.to_datetime(price_df["Date"])
    df["y"] = price_df["Close"].values

    # --- Normalised technical indicators ---
    rsi = compute_rsi(price_df["Close"])
    df["rsi_norm"] = (rsi.values - 50) / 50                   # [-1, +1]

    macd_line, _, _ = compute_macd(price_df["Close"])
    price_range = price_df["Close"].max() - price_df["Close"].min()
    df["macd_norm"] = (
        (macd_line.values / price_range) if price_range > 0 else 0.0
    )

    ma_short, ma_long = compute_moving_averages(price_df["Close"])
    df["ma_ratio"] = (ma_short.values / ma_long.values) - 1   # deviation from long MA

    # --- Sentiment regressor ---
    if sentiment_ts is not None and not sentiment_ts.empty:
        sent = sentiment_ts.copy()
        sent["date"] = pd.to_datetime(sent["date"])
        df = df.merge(
            sent[["date", "avg_sentiment"]],
            left_on="ds",
            right_on="date",
            how="left",
        )
        df["sentiment"] = df["avg_sentiment"].ffill().fillna(0)
        df.drop(columns=["date", "avg_sentiment"], inplace=True, errors="ignore")
    else:
        df["sentiment"] = 0.0

    df = df.fillna(0)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Prophet Forecasting
# ═══════════════════════════════════════════════════════════════════════════

_REGRESSOR_COLS = ["rsi_norm", "macd_norm", "ma_ratio", "sentiment"]


def forecast_price(
    features_df: pd.DataFrame,
    forecast_days: int = PREDICTION.FORECAST_DAYS,
) -> pd.DataFrame:
    """
    Train a Prophet model on historical features and generate a
    *forecast_days*-day-ahead prediction with confidence bands.

    Returns:
        DataFrame with columns:  ds, y (NaN for future), yhat,
        yhat_lower, yhat_upper.
    """
    if not _PROPHET_AVAILABLE:
        raise ImportError(
            "Prophet is required for forecasting.  pip install prophet"
        )

    if len(features_df) < PREDICTION.MIN_TRAINING_POINTS:
        raise ValueError(
            f"Insufficient data for Prophet: {len(features_df)} rows "
            f"(need ≥ {PREDICTION.MIN_TRAINING_POINTS})."
        )

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=PREDICTION.PROPHET_CHANGEPOINT_SCALE,
        interval_width=PREDICTION.PROPHET_INTERVAL_WIDTH,
    )

    for col in _REGRESSOR_COLS:
        if col in features_df.columns:
            model.add_regressor(col)

    train = features_df[["ds", "y"] + _REGRESSOR_COLS].copy()
    model.fit(train)

    future = model.make_future_dataframe(periods=forecast_days)

    # Project regressors forward using recent rolling averages
    window = PREDICTION.SENTIMENT_ROLLING_WINDOW
    recent_vals = {
        col: train[col].rolling(window, min_periods=1).mean().iloc[-1]
        for col in _REGRESSOR_COLS
    }

    future = future.merge(
        train[["ds"] + _REGRESSOR_COLS], on="ds", how="left",
    )
    for col in _REGRESSOR_COLS:
        future[col] = future[col].fillna(recent_vals[col])

    forecast = model.predict(future)
    forecast = forecast.merge(train[["ds", "y"]], on="ds", how="left")

    return forecast[["ds", "y", "yhat", "yhat_lower", "yhat_upper"]]


# ═══════════════════════════════════════════════════════════════════════════
# Trend Classification
# ═══════════════════════════════════════════════════════════════════════════

def classify_trend(forecast_df: pd.DataFrame) -> str:
    """
    Compare the last historical price to the final predicted price
    and return 'Bullish', 'Bearish', or 'Neutral'.
    """
    historical = forecast_df.dropna(subset=["y"])
    if historical.empty:
        return "Neutral"

    last_price = historical["y"].iloc[-1]
    future = forecast_df[forecast_df["y"].isna()]
    if future.empty:
        return "Neutral"

    predicted_end = future["yhat"].iloc[-1]
    change_pct = ((predicted_end - last_price) / last_price) * 100

    if change_pct > THRESHOLDS.BULLISH_PCT:
        return "Bullish"
    if change_pct < THRESHOLDS.BEARISH_PCT:
        return "Bearish"
    return "Neutral"
