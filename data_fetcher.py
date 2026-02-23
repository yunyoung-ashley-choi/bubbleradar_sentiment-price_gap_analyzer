"""
Data fetching module for Bubble Radar.
Handles stock price retrieval via yfinance and news headlines via NewsAPI.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

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
    api_key: str,
    days: int = THRESHOLDS.LOOKBACK_DAYS,
    page_size: int = 100,
) -> list[dict]:
    """
    Fetch news headlines from NewsAPI for a given sector.

    Args:
        sector: Sector name to search for.
        api_key: NewsAPI API key.
        days: How many days back to search.
        page_size: Max number of articles to fetch.

    Returns:
        List of dicts with keys: title, source, published_at, url, description.

    Raises:
        ConnectionError: If NewsAPI request fails.
    """
    keywords = SECTOR_SEARCH_KEYWORDS.get(sector, sector)
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keywords,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            error_msg = data.get("message", "Unknown NewsAPI error")
            raise ConnectionError(f"NewsAPI error: {error_msg}")

        articles = data.get("articles", [])
        if not articles:
            logger.warning(f"No news articles found for sector: {sector}")
            return []

        parsed = []
        for article in articles:
            title = article.get("title", "").strip()
            if not title or title == "[Removed]":
                continue
            parsed.append({
                "title": title,
                "source": article.get("source", {}).get("name", "Unknown"),
                "published_at": article.get("publishedAt", ""),
                "url": article.get("url", ""),
                "description": article.get("description", "") or "",
            })

        logger.info(f"Fetched {len(parsed)} news articles for sector: {sector}")
        return parsed

    except requests.exceptions.Timeout:
        raise ConnectionError("NewsAPI request timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        raise ConnectionError(f"NewsAPI HTTP error: {e}") from e
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to NewsAPI: {e}") from e


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
    df["date"] = pd.to_datetime(df["published_at"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    return df[["date", "title", "source", "url", "description"]]
