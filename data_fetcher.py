"""
Data fetching module for Bubble Radar.
Handles stock price retrieval via yfinance and news headlines via GNews (Google News).
GNews is free, requires no API key, and works from any server including Streamlit Cloud.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from gnews import GNews

from config import SECTOR_SEARCH_KEYWORDS, THRESHOLDS

logger = logging.getLogger(__name__)


def fetch_stock_data(ticker: str, days: int = THRESHOLDS.LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetch historical stock price data for a given ticker.

    Args:
        ticker: ETF/stock ticker symbol.
        days: Number of days of historical data to fetch.

    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume columns.

    Raises:
        ValueError: If no data is returned for the ticker.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

        if df.empty:
            raise ValueError(f"No price data returned for ticker '{ticker}'.")

        df = df.reset_index()

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

        logger.info(f"Fetched {len(df)} rows of price data for {ticker}.")
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch stock data for {ticker}: {e}")
        raise ValueError(f"Could not retrieve data for '{ticker}'. Error: {e}") from e


def fetch_news_headlines(
    sector: str,
    days: int = THRESHOLDS.LOOKBACK_DAYS,
    max_results: int = THRESHOLDS.MAX_HEADLINES,
) -> list[dict]:
    """
    Fetch news headlines from Google News via the gnews package.
    Free, no API key required, works from any server.

    Args:
        sector: Sector name to search for.
        days: How many days back to search.
        max_results: Max number of articles to fetch.

    Returns:
        List of dicts with keys: title, source, published_at, url, description.

    Raises:
        ConnectionError: If the request fails.
    """
    keywords = SECTOR_SEARCH_KEYWORDS.get(sector, sector)
    period_str = f"{days}d"

    try:
        google_news = GNews(
            language="en",
            country="US",
            period=period_str,
            max_results=max_results,
        )

        raw_articles = google_news.get_news(keywords)

        if not raw_articles:
            logger.warning(f"No news articles found for sector: {sector}")
            return []

        parsed = []
        for article in raw_articles:
            title = (article.get("title") or "").strip()
            if not title:
                continue

            publisher = article.get("publisher", {})
            source_name = publisher.get("title", "Unknown") if isinstance(publisher, dict) else str(publisher)

            parsed.append({
                "title": title,
                "source": source_name,
                "published_at": article.get("published date", ""),
                "url": article.get("url", ""),
                "description": article.get("description", "") or "",
            })

        logger.info(f"Fetched {len(parsed)} news articles for sector: {sector}")
        return parsed

    except Exception as e:
        logger.error(f"Failed to fetch news for {sector}: {e}")
        raise ConnectionError(f"Google News fetch failed: {e}") from e


def build_news_dataframe(articles: list[dict]) -> pd.DataFrame:
    """
    Convert raw article list into a date-indexed DataFrame.

    Args:
        articles: List of article dicts from fetch_news_headlines.

    Returns:
        DataFrame with parsed dates and article metadata.
    """
    if not articles:
        return pd.DataFrame(columns=["date", "title", "source", "url", "description"])

    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["date"] = df["date"].dt.date
    df = df.dropna(subset=["date"])
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    return df[["date", "title", "source", "url", "description"]]
