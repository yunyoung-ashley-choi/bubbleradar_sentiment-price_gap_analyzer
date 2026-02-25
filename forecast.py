"""
Prophet-based 60-day stock price forecasting with Plotly visualisation.

Self-contained module — fetches historical data from yfinance,
trains a Prophet model, and returns both the forecast DataFrame
and an interactive Plotly chart.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

logger = logging.getLogger(__name__)

for _n in ("prophet", "cmdstanpy"):
    logging.getLogger(_n).setLevel(logging.WARNING)

try:
    from prophet import Prophet
    _PROPHET_OK = True
except ImportError:
    _PROPHET_OK = False
    logger.warning("Prophet not installed — run:  pip install prophet")


# ═══════════════════════════════════════════════════════════════════════════
# Data fetching
# ═══════════════════════════════════════════════════════════════════════════

def fetch_price_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch OHLCV history from yfinance.

    Returns:
        DataFrame with columns Date, Open, High, Low, Close, Volume.
    Raises:
        ValueError: if yfinance returns no rows.
    """
    end = datetime.now()
    start = end - timedelta(days=days)

    df = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    if df.empty:
        raise ValueError(f"No price data for '{ticker}'.")

    df = df.reset_index()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


# ═══════════════════════════════════════════════════════════════════════════
# Prophet forecast
# ═══════════════════════════════════════════════════════════════════════════

def run_forecast(
    price_df: pd.DataFrame,
    forecast_days: int = 60,
) -> pd.DataFrame:
    """
    Train Prophet on *price_df* and return a combined DataFrame
    with columns: ds, y, yhat, yhat_lower, yhat_upper.
    Historical rows have real *y*; future rows have *y = NaN*.
    """
    if not _PROPHET_OK:
        raise ImportError("Prophet is required.  pip install prophet")

    prophet_df = price_df[["Date", "Close"]].copy()
    prophet_df.columns = ["ds", "y"]

    if len(prophet_df) < 30:
        raise ValueError(
            f"Prophet needs ≥ 30 data points, got {len(prophet_df)}."
        )

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.80,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    forecast = forecast.merge(prophet_df, on="ds", how="left")

    return forecast[["ds", "y", "yhat", "yhat_lower", "yhat_upper"]]


# ═══════════════════════════════════════════════════════════════════════════
# Plotly chart
# ═══════════════════════════════════════════════════════════════════════════

def create_forecast_chart(
    forecast_df: pd.DataFrame,
    ticker: str,
    name: str,
    currency: str = "$",
) -> go.Figure:
    """
    Historical line + dashed forecast line + 80 % confidence band.
    """
    hist = forecast_df.dropna(subset=["y"])
    future = forecast_df[forecast_df["y"].isna()]

    hover = (
        "₩%{y:,.0f}<extra>%{fullData.name}</extra>"
        if currency == "₩"
        else "$%{y:,.2f}<extra>%{fullData.name}</extra>"
    )

    fig = go.Figure()

    # Historical
    fig.add_trace(
        go.Scatter(
            x=hist["ds"],
            y=hist["y"],
            name="Historical",
            line=dict(color="#636EFA", width=2),
            hovertemplate=hover,
        )
    )

    if not future.empty and not hist.empty:
        # Bridge segment
        bx = pd.concat([hist["ds"].tail(1), future["ds"].head(1)], ignore_index=True)
        by = pd.concat([hist["y"].tail(1), future["yhat"].head(1)], ignore_index=True)
        fig.add_trace(
            go.Scatter(
                x=bx, y=by,
                line=dict(color="#FFA15A", width=2, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if not future.empty:
        fig.add_trace(
            go.Scatter(
                x=future["ds"],
                y=future["yhat"],
                name="60-Day Forecast",
                line=dict(color="#FFA15A", width=2.5, dash="dash"),
                hovertemplate=hover,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future["ds"],
                y=future["yhat_upper"],
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future["ds"],
                y=future["yhat_lower"],
                name="80 % Confidence",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255,161,90,0.15)",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{name} ({ticker}) — 60-Day Price Forecast",
            font=dict(size=18),
        ),
        template="plotly_dark",
        height=520,
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
    )
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig
