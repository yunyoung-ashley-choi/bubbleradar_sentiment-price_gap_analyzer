"""
Bubble Index calculation module.
Quantifies the gap between price momentum and market sentiment
to identify potential bubbles or undervalued conditions.
"""

import math
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from config import THRESHOLDS


class MarketStatus(Enum):
    OVERHEATED = "Overheated"
    STABLE = "Stable"
    UNDERVALUED = "Undervalued"


@dataclass
class BubbleResult:
    """Container for bubble analysis output."""
    bubble_index: float
    status: MarketStatus
    price_change_pct: float
    price_momentum_score: float
    sentiment_score: float
    explanation: str


def _normalize_price_momentum(price_change_pct: float) -> float:
    """
    Normalize price change percentage to [-1, 1] using tanh.
    This compresses extreme moves while preserving direction.
    The PRICE_NORM_FACTOR controls sensitivity (lower = more sensitive).
    """
    return math.tanh(price_change_pct / THRESHOLDS.PRICE_NORM_FACTOR)


def calculate_price_change(price_df: pd.DataFrame) -> float:
    """
    Calculate percentage price change over the period.

    Args:
        price_df: DataFrame with 'Close' column, sorted by date ascending.

    Returns:
        Percentage change as a float (e.g., 5.2 for +5.2%).
    """
    if price_df.empty or len(price_df) < 2:
        return 0.0

    first_close = price_df["Close"].iloc[0]
    last_close = price_df["Close"].iloc[-1]

    if first_close == 0:
        return 0.0

    return ((last_close - first_close) / first_close) * 100


def compute_bubble_index(
    price_df: pd.DataFrame,
    overall_sentiment: float,
) -> BubbleResult:
    """
    Compute the Bubble Index by comparing price momentum vs. sentiment.

    Bubble Index = price_momentum_score - sentiment_score
    - Positive gap (price >> sentiment): Overheated / potential bubble
    - Negative gap (sentiment >> price): Undervalued / potential opportunity
    - Near zero: Stable / price reflects sentiment

    Args:
        price_df: Price DataFrame with 'Close' column.
        overall_sentiment: Aggregated sentiment score [-1, +1].

    Returns:
        BubbleResult with index value, status, and breakdown.
    """
    price_change_pct = calculate_price_change(price_df)
    price_momentum = _normalize_price_momentum(price_change_pct)
    sentiment = max(-1.0, min(1.0, overall_sentiment))

    bubble_index = round(price_momentum - sentiment, 4)

    if bubble_index > THRESHOLDS.OVERHEATED:
        status = MarketStatus.OVERHEATED
        explanation = (
            f"Price momentum ({price_change_pct:+.1f}%) significantly exceeds "
            f"sentiment ({sentiment:+.2f}). The market may be pricing in more "
            f"optimism than news justifies — potential bubble risk."
        )
    elif bubble_index < THRESHOLDS.UNDERVALUED:
        status = MarketStatus.UNDERVALUED
        explanation = (
            f"Sentiment ({sentiment:+.2f}) is considerably more positive than "
            f"price action ({price_change_pct:+.1f}%). The sector may be "
            f"undervalued relative to the prevailing narrative."
        )
    else:
        status = MarketStatus.STABLE
        explanation = (
            f"Price trend ({price_change_pct:+.1f}%) and sentiment ({sentiment:+.2f}) "
            f"are reasonably aligned. No significant divergence detected."
        )

    return BubbleResult(
        bubble_index=bubble_index,
        status=status,
        price_change_pct=round(price_change_pct, 2),
        price_momentum_score=round(price_momentum, 4),
        sentiment_score=round(sentiment, 4),
        explanation=explanation,
    )
