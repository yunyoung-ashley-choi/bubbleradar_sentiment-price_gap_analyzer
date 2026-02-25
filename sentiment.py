"""
Sentiment analysis module using FinBERT (ProsusAI/finbert).
Fetches news headlines via GNews and scores them with a pre-trained
financial NLP model.  Sentiment scores range from -1 (negative) to +1 (positive).
"""

import logging
from functools import lru_cache
from typing import Optional, Callable

import numpy as np
import pandas as pd

from config import THRESHOLDS
from data_fetcher import fetch_news_headlines, build_news_dataframe

logger = logging.getLogger(__name__)

_FINBERT_MODEL_ID = "ProsusAI/finbert"

# ---------------------------------------------------------------------------
# Lazy import — torch / transformers are heavy; fail gracefully
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _FINBERT_AVAILABLE = True
except ImportError:
    _FINBERT_AVAILABLE = False
    logger.warning(
        "FinBERT deps missing. Install with:  pip install transformers torch"
    )


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_finbert() -> tuple:
    """
    Load and cache the FinBERT model + tokenizer.
    First invocation downloads ~420 MB from Hugging Face Hub.
    """
    if not _FINBERT_AVAILABLE:
        raise ImportError(
            "FinBERT requires 'transformers' and 'torch'. "
            "Install them with:  pip install transformers torch"
        )
    logger.info("Loading FinBERT model (first run may download ~420 MB)...")
    tokenizer = AutoTokenizer.from_pretrained(_FINBERT_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL_ID)
    model.eval()
    logger.info("FinBERT loaded successfully.")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_headlines(
    headlines: list[str],
    batch_size: int = 16,
) -> list[float]:
    """
    Score a list of headlines using FinBERT.

    FinBERT label order:  0 = positive, 1 = negative, 2 = neutral.
    Score = P(positive) − P(negative)  →  range [-1, +1].
    """
    if not headlines:
        return []

    tokenizer, model = _load_finbert()
    all_scores: list[float] = []

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        for j in range(len(batch)):
            p_pos = probs[j][0].item()
            p_neg = probs[j][1].item()
            all_scores.append(round(p_pos - p_neg, 4))

    return all_scores


def compute_overall_sentiment(scores: list[float]) -> float:
    """Aggregate per-headline scores into a single number."""
    if not scores:
        return 0.0
    return round(float(np.mean(scores)), 4)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def analyze_asset_sentiment(
    asset_name: str,
    days: int = THRESHOLDS.LOOKBACK_DAYS,
    max_headlines: int = THRESHOLDS.MAX_HEADLINES,
    status_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Full pipeline: fetch news → FinBERT scoring → aggregated results.

    Args:
        asset_name: Key in SECTOR_SEARCH_KEYWORDS (e.g. "Samsung Electronics").
        days: How many days of news to look back.
        max_headlines: Cap on headlines to score.
        status_callback: Optional UI progress updater.

    Returns:
        Dict with keys:
          headlines, scores, overall_sentiment, news_df, sentiment_ts
    """
    empty_result = {
        "headlines": [],
        "scores": [],
        "overall_sentiment": 0.0,
        "news_df": pd.DataFrame(columns=["date", "title", "source", "url", "description"]),
        "sentiment_ts": pd.DataFrame(columns=["date", "avg_sentiment"]),
    }

    # --- Fetch news ---
    if status_callback:
        status_callback(f"Fetching news for {asset_name}...")

    try:
        raw_articles = fetch_news_headlines(
            sector=asset_name,
            days=days,
            max_results=max_headlines,
        )
    except ConnectionError:
        logger.warning(f"News fetch failed for {asset_name}")
        return empty_result

    news_df = build_news_dataframe(raw_articles)
    headlines = news_df["title"].tolist() if not news_df.empty else []

    if not headlines:
        return empty_result

    # --- Score with FinBERT ---
    if status_callback:
        status_callback(f"Scoring {len(headlines)} headlines with FinBERT...")

    trimmed = headlines[: max_headlines]
    scores = score_headlines(trimmed)
    overall = compute_overall_sentiment(scores)

    # --- Build daily time-series ---
    sentiment_ts = _build_sentiment_timeseries(news_df, scores)

    return {
        "headlines": trimmed,
        "scores": scores,
        "overall_sentiment": overall,
        "news_df": news_df,
        "sentiment_ts": sentiment_ts,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_sentiment_timeseries(
    news_df: pd.DataFrame,
    scores: list[float],
) -> pd.DataFrame:
    """Merge per-article scores with dates → daily average series."""
    if news_df.empty or not scores:
        return pd.DataFrame(columns=["date", "avg_sentiment"])

    scored = news_df.head(len(scores)).copy()
    scored["sentiment_score"] = scores[: len(scored)]

    daily = (
        scored.groupby("date")["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_score": "avg_sentiment"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily
