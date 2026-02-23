"""
Sentiment analysis module using Google Gemini API (google-genai SDK).
Analyzes financial news headlines and returns structured sentiment scores.
Includes automatic model fallback for free-tier quota limits.
"""

import json
import logging
import re
import time

from google import genai
import pandas as pd

from config import GEMINI_MODELS, THRESHOLDS

logger = logging.getLogger(__name__)

_client: genai.Client | None = None

_SENTIMENT_PROMPT_TEMPLATE = """You are an expert financial sentiment analyst.
Analyze the following news headlines related to "{sector}".

For EACH headline, assign a sentiment score on a continuous scale:
  -1.0 = Extremely Negative (catastrophic news, crashes, fraud)
  -0.5 = Moderately Negative (declines, downgrades, concerns)
   0.0 = Neutral (factual, no clear positive/negative implication)
  +0.5 = Moderately Positive (growth, upgrades, partnerships)
  +1.0 = Extremely Positive (breakthroughs, record earnings, major deals)

Also provide a single overall_sentiment score that represents the aggregate
sentiment across ALL headlines.

CRITICAL: Respond ONLY with valid JSON. No markdown, no explanation, no extra text.
Use this exact JSON structure:

{{
  "overall_sentiment": <float between -1.0 and 1.0>,
  "articles": [
    {{"index": 1, "score": <float between -1.0 and 1.0>}},
    {{"index": 2, "score": <float between -1.0 and 1.0>}}
  ]
}}

Headlines to analyze:
{headlines_block}
"""


def configure_gemini(api_key: str) -> None:
    """Initialize the Gemini client with the provided API key."""
    global _client
    _client = genai.Client(api_key=api_key)


def _get_client() -> genai.Client:
    """Return the configured client or raise."""
    if _client is None:
        raise RuntimeError("Gemini client not configured. Call configure_gemini() first.")
    return _client


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _is_retryable_error(error: Exception) -> bool:
    """Check if an error is a rate-limit/quota issue (worth retrying or falling back)."""
    err = str(error)
    err_lower = err.lower()
    return (
        "429" in err
        or "quota" in err_lower
        or "rate limit" in err_lower
        or "resource_exhausted" in err_lower
        or "resource exhausted" in err_lower
    )


def _is_model_not_found(error: Exception) -> bool:
    """Check if the model name is invalid / sunset."""
    err = str(error)
    return "404" in err or "not_found" in err.lower() or "not found" in err.lower()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _build_headlines_block(headlines: list[str]) -> str:
    """Format headlines into a numbered list for the prompt."""
    return "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))


def _parse_gemini_response(raw_text: str, num_headlines: int) -> dict:
    """
    Parse the JSON response from Gemini, with fallback extraction.

    Returns dict with 'overall_sentiment' (float) and 'articles' (list of dicts).
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini JSON response: {e}")
                raise ValueError("Gemini returned an unparseable response.") from e
        else:
            raise ValueError("Gemini response contained no valid JSON.")

    overall = float(parsed.get("overall_sentiment", 0.0))
    overall = max(-1.0, min(1.0, overall))

    articles = []
    raw_articles = parsed.get("articles", [])
    for entry in raw_articles:
        score = float(entry.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        articles.append({"index": entry.get("index", 0), "score": score})

    if len(articles) < num_headlines:
        for i in range(len(articles), num_headlines):
            articles.append({"index": i + 1, "score": 0.0})

    return {"overall_sentiment": overall, "articles": articles}


# ---------------------------------------------------------------------------
# Core sentiment analysis with automatic model fallback
# ---------------------------------------------------------------------------

def analyze_sentiment(
    headlines: list[str],
    sector: str,
    model_name: str = "gemini-2.0-flash-lite",
    status_callback=None,
) -> dict:
    """
    Send headlines to Gemini for sentiment scoring.
    Automatically falls back to other models if the selected one is rate-limited
    or unavailable — no separate probe call needed (zero wasted quota).

    Args:
        headlines: List of headline strings.
        sector: Sector name for context.
        model_name: Preferred Gemini model.
        status_callback: Optional callable(message) for UI progress updates.

    Returns:
        Dict with keys:
          - 'overall_sentiment' (float)
          - 'articles' (list of score dicts)
          - 'model_used' (str) — the model that actually succeeded

    Raises:
        ValueError: If Gemini returns an unparseable response.
        ConnectionError: If all models fail.
    """
    if not headlines:
        return {"overall_sentiment": 0.0, "articles": [], "model_used": model_name}

    max_headlines = THRESHOLDS.MAX_HEADLINES
    if len(headlines) > max_headlines:
        headlines = headlines[:max_headlines]
        logger.info(f"Trimmed headlines to {max_headlines} for API efficiency.")

    prompt = _SENTIMENT_PROMPT_TEMPLATE.format(
        sector=sector,
        headlines_block=_build_headlines_block(headlines),
    )

    models_to_try = [model_name] + [m for m in GEMINI_MODELS if m != model_name]
    client = _get_client()
    all_errors: list[str] = []

    for model in models_to_try:
        for attempt in range(1, 3):
            try:
                if status_callback:
                    status_callback(f"Calling {model}...")

                response = client.models.generate_content(
                    model=model, contents=prompt,
                )
                raw_text = response.text
                logger.info(f"Gemini analysis succeeded with model {model}.")

                result = _parse_gemini_response(raw_text, len(headlines))
                result["model_used"] = model
                return result

            except Exception as e:
                error_str = str(e)

                if _is_model_not_found(e):
                    logger.warning(f"Model {model} not found, skipping.")
                    all_errors.append(f"{model}: not found / unavailable")
                    break

                if _is_retryable_error(e):
                    if attempt < 2:
                        wait_match = re.search(r"retry in ([\d.]+)s", error_str, re.IGNORECASE)
                        wait_secs = float(wait_match.group(1)) + 1.0 if wait_match else 15.0
                        wait_secs = min(wait_secs, 30.0)

                        if status_callback:
                            status_callback(
                                f"{model} rate-limited. Waiting {wait_secs:.0f}s before retry..."
                            )
                        time.sleep(wait_secs)
                        continue
                    else:
                        logger.warning(f"Model {model} quota exhausted after retry, trying next model.")
                        all_errors.append(f"{model}: quota / rate-limit exhausted")
                        break

                logger.error(f"Model {model} non-retryable error: {e}")
                all_errors.append(f"{model}: {error_str}")
                break

    error_summary = "\n".join(f"  - {e}" for e in all_errors)
    raise ConnectionError(
        f"All Gemini models failed:\n{error_summary}\n\n"
        f"If you just created your API key, wait 2-3 minutes for quotas to activate. "
        f"Your Google Cloud project may also need billing enabled (even for free tier)."
    )


# ---------------------------------------------------------------------------
# Timeseries builder
# ---------------------------------------------------------------------------

def build_sentiment_timeseries(
    news_df: pd.DataFrame,
    sentiment_result: dict,
) -> pd.DataFrame:
    """
    Merge per-article sentiment scores with dates to create a daily timeseries.

    Args:
        news_df: DataFrame with 'date' and 'title' columns.
        sentiment_result: Output from analyze_sentiment().

    Returns:
        DataFrame with columns: date, avg_sentiment (daily average).
    """
    if news_df.empty or not sentiment_result.get("articles"):
        return pd.DataFrame(columns=["date", "avg_sentiment"])

    articles = sentiment_result["articles"]
    scores = [a["score"] for a in articles]

    scored_df = news_df.head(len(scores)).copy()
    scored_df["sentiment_score"] = scores[: len(scored_df)]

    daily = (
        scored_df.groupby("date")["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_score": "avg_sentiment"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    return daily
