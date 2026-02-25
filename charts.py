"""
Chart generation module using Plotly.
Interactive charts for price-sentiment overlay, bubble gauge,
and 60-day forecast with confidence bands.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ═══════════════════════════════════════════════════════════════════════════
# Price vs Sentiment dual-axis chart
# ═══════════════════════════════════════════════════════════════════════════

def create_price_sentiment_chart(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    ticker: str,
    sector: str,
) -> go.Figure:
    """
    Dual-axis line chart:
      left  → Stock/ETF closing price
      right → Daily average sentiment score (bars + 3-day MA)
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=price_df["Date"],
            y=price_df["Close"],
            name=f"{ticker} Price",
            line=dict(color="#636EFA", width=2.5),
            mode="lines",
            hovertemplate="$%{y:.2f}<extra>Price</extra>",
        ),
        secondary_y=False,
    )

    if not sentiment_df.empty:
        colors = [
            "#00CC96" if v >= 0 else "#EF553B"
            for v in sentiment_df["avg_sentiment"]
        ]
        fig.add_trace(
            go.Bar(
                x=sentiment_df["date"],
                y=sentiment_df["avg_sentiment"],
                name="Daily Sentiment",
                marker_color=colors,
                opacity=0.5,
                hovertemplate="%{y:.3f}<extra>Sentiment</extra>",
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=sentiment_df["date"],
                y=sentiment_df["avg_sentiment"]
                .rolling(3, min_periods=1)
                .mean(),
                name="Sentiment Trend (3-day MA)",
                line=dict(color="#FFA15A", width=2, dash="dot"),
                mode="lines",
                hovertemplate="%{y:.3f}<extra>3-day MA</extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=dict(
            text=f"{sector} — Price vs. Sentiment",
            font=dict(size=18),
        ),
        template="plotly_dark",
        height=520,
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(
        title_text=f"{ticker} Close Price",
        secondary_y=False,
        gridcolor="rgba(128,128,128,0.2)",
    )
    fig.update_yaxes(
        title_text="Sentiment Score",
        secondary_y=True,
        range=[-1.1, 1.1],
        gridcolor="rgba(128,128,128,0.1)",
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Bubble Index gauge
# ═══════════════════════════════════════════════════════════════════════════

def create_bubble_gauge(bubble_index: float) -> go.Figure:
    """Gauge chart for the Bubble Index value (-1.5 to +1.5)."""
    clamped = max(-1.5, min(1.5, bubble_index))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=clamped,
            number=dict(suffix="", font=dict(size=36)),
            title=dict(text="Bubble Index", font=dict(size=16)),
            gauge=dict(
                axis=dict(range=[-1.5, 1.5], tickwidth=1),
                bar=dict(color="#636EFA", thickness=0.3),
                bgcolor="rgba(0,0,0,0)",
                steps=[
                    dict(range=[-1.5, -0.4], color="rgba(0, 204, 150, 0.3)"),
                    dict(range=[-0.4, 0.4], color="rgba(99, 110, 250, 0.15)"),
                    dict(range=[0.4, 1.5], color="rgba(239, 85, 59, 0.3)"),
                ],
                threshold=dict(
                    line=dict(color="white", width=3),
                    thickness=0.8,
                    value=clamped,
                ),
            ),
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=50, b=10),
        template="plotly_dark",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 60-Day Forecast chart
# ═══════════════════════════════════════════════════════════════════════════

def _hover_template(currency: str) -> str:
    if currency == "₩":
        return "₩%{y:,.0f}<extra>%{fullData.name}</extra>"
    return "$%{y:,.2f}<extra>%{fullData.name}</extra>"


def create_forecast_chart(
    forecast_df: pd.DataFrame,
    ticker: str,
    name: str,
    currency: str = "$",
) -> go.Figure:
    """
    Historical prices + 60-day forecast line + 80 % confidence band.

    Args:
        forecast_df: Output of engine.forecast_price() with columns
                     ds, y, yhat, yhat_lower, yhat_upper.
        ticker: Ticker string for axis label.
        name: Human-readable asset name.
        currency: '$' or '₩' for formatting.
    """
    historical = forecast_df.dropna(subset=["y"]).copy()
    future = forecast_df[forecast_df["y"].isna()].copy()
    h_tmpl = _hover_template(currency)

    fig = go.Figure()

    # --- Historical price line ---
    fig.add_trace(
        go.Scatter(
            x=historical["ds"],
            y=historical["y"],
            name="Historical",
            line=dict(color="#636EFA", width=2),
            mode="lines",
            hovertemplate=h_tmpl,
        )
    )

    if not future.empty and not historical.empty:
        # Bridge segment connecting last real point to first forecast
        bridge_ds = pd.concat(
            [historical["ds"].tail(1), future["ds"].head(1)], ignore_index=True
        )
        bridge_y = pd.concat(
            [historical["y"].tail(1), future["yhat"].head(1)], ignore_index=True
        )
        fig.add_trace(
            go.Scatter(
                x=bridge_ds,
                y=bridge_y,
                line=dict(color="#FFA15A", width=2, dash="dash"),
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if not future.empty:
        # Forecast line
        fig.add_trace(
            go.Scatter(
                x=future["ds"],
                y=future["yhat"],
                name="60-Day Forecast",
                line=dict(color="#FFA15A", width=2.5, dash="dash"),
                mode="lines",
                hovertemplate=h_tmpl,
            )
        )

        # Confidence band (upper boundary — invisible, just for fill anchor)
        fig.add_trace(
            go.Scatter(
                x=future["ds"],
                y=future["yhat_upper"],
                line=dict(width=0),
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Confidence band (lower boundary — fills to previous trace)
        fig.add_trace(
            go.Scatter(
                x=future["ds"],
                y=future["yhat_lower"],
                name="80 % Confidence",
                line=dict(width=0),
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(255, 161, 90, 0.15)",
                hoverinfo="skip",
            )
        )

    y_title = f"Price ({currency})"
    fig.update_layout(
        title=dict(
            text=f"{name} ({ticker}) — 60-Day Price Forecast",
            font=dict(size=18),
        ),
        template="plotly_dark",
        height=520,
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=y_title,
    )
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig
