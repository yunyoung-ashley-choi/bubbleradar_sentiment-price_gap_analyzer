"""
Chart generation module using Plotly.
Creates interactive dual-axis charts for price and sentiment visualization.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_price_sentiment_chart(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    ticker: str,
    sector: str,
) -> go.Figure:
    """
    Create an interactive dual-axis line chart:
      - Primary Y-axis (left): Stock/ETF closing price
      - Secondary Y-axis (right): Daily average sentiment score

    Args:
        price_df: DataFrame with 'Date' and 'Close' columns.
        sentiment_df: DataFrame with 'date' and 'avg_sentiment' columns.
        ticker: Ticker symbol for labeling.
        sector: Sector name for the title.

    Returns:
        Plotly Figure object.
    """
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
    )

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
                y=sentiment_df["avg_sentiment"].rolling(3, min_periods=1).mean(),
                name="Sentiment Trend (3-day MA)",
                line=dict(color="#FFA15A", width=2, dash="dot"),
                mode="lines",
                hovertemplate="%{y:.3f}<extra>3-day MA</extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=dict(
            text=f"{sector} — Price vs. Sentiment (Last 30 Days)",
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
        title_text=f"{ticker} Close Price ($)",
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


def create_bubble_gauge(bubble_index: float) -> go.Figure:
    """
    Create a gauge chart representing the Bubble Index value.

    Args:
        bubble_index: Computed bubble index value (typically -2 to +2).

    Returns:
        Plotly Figure with a gauge indicator.
    """
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
