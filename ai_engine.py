"""
Cost-optimised AI engine using Google Gemini 2.5 Flash Lite.

* Fetches the **top 3** news headlines per ticker via GNews (free).
* Calls the cheapest available Gemini model once per ticker.
* Automatic fallback chain if the primary model is rate-limited.
* Results cached for 1 hour (`@st.cache_data`) to prevent repeat charges.

Verified models (2025-02):
  gemini-2.5-flash-lite  ← cheapest, fastest  (CONFIRMED WORKING)
  gemini-2.5-flash       ← balanced fallback   (CONFIRMED WORKING)
"""

import json
import logging
import os
import re
import time

import streamlit as st
from gnews import GNews
from google import genai

logger = logging.getLogger(__name__)

_MODELS: list[str] = [
    "gemini-2.5-flash-lite",   # cheapest — confirmed working
    "gemini-2.5-flash",        # fallback — confirmed working
]

_SEARCH_KEYWORDS: dict[str, str] = {
    "005930.KS": "Samsung Electronics semiconductor memory",
    "015760.KS": "KEPCO Korea Electric Power utility",
    "000720.KS": "Hyundai Engineering Construction infrastructure",
    "294870.KS": "HDC Hyundai Development real estate",
    "^GSPC":     "S&P 500 stock market Wall Street economy",
    "^NDX":      "NASDAQ 100 tech stocks index growth",
    "TSLA":      "Tesla Elon Musk EV electric vehicle",
    "NVDA":      "NVIDIA GPU AI chips Jensen Huang",
}


# ═══════════════════════════════════════════════════════════════════════════
# Gemini client
# ═══════════════════════════════════════════════════════════════════════════

def _get_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Create a .env file — see .env.example for the template."
        )
    return genai.Client(api_key=key)


# ═══════════════════════════════════════════════════════════════════════════
# News fetching (top 3 only — keeps Gemini costs minimal)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_top_headlines(ticker: str, max_results: int = 3) -> list[str]:
    """Return up to *max_results* recent headlines from Google News."""
    keywords = _SEARCH_KEYWORDS.get(ticker, ticker)
    try:
        gn = GNews(language="en", country="US", period="7d", max_results=max_results)
        articles = gn.get_news(keywords) or []
        return [
            a["title"].strip()
            for a in articles
            if a.get("title", "").strip()
        ][:max_results]
    except Exception as exc:
        logger.warning("GNews fetch failed for %s: %s", ticker, exc)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Gemini sentiment analysis (cached 1 h)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_sentiment(ticker: str, headlines_tuple: tuple[str, ...]) -> dict:
    """
    Score headlines with Gemini and produce a 1-sentence summary.

    Tries each model in ``_MODELS`` in order.  If a model is
    rate-limited (429) or not found (404), the next one is tried
    automatically — zero wasted quota on probe calls.

    Args:
        ticker: Stock ticker string (also part of the cache key).
        headlines_tuple: Tuple of headline strings (hashable for cache).

    Returns:
        {"sentiment": float, "summary": str, "scores": list[float],
         "model_used": str}
    """
    headlines = list(headlines_tuple)
    if not headlines:
        return {
            "sentiment": 0.0,
            "summary": "No recent news available.",
            "scores": [],
            "model_used": "",
        }

    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    prompt = (
        f"You are a senior financial analyst.  "
        f"Analyse these news headlines related to **{ticker}**.\n\n"
        f"Headlines:\n{numbered}\n\n"
        "For each headline assign a sentiment score from "
        "-1.0 (very negative) to +1.0 (very positive).\n"
        "Then compute an overall_sentiment (weighted average) and write "
        "a concise 1-sentence market outlook.\n\n"
        "Respond ONLY with valid JSON — no markdown fences:\n"
        '{"overall_sentiment": <float>, '
        '"summary": "<1-sentence outlook>", '
        '"scores": [<float>, ...]}'
    )

    client = _get_client()
    errors: list[str] = []

    for model in _MODELS:
        for attempt in range(1, 3):
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt,
                )
                result = _parse_response(response.text, len(headlines))
                result["model_used"] = model
                logger.info("Gemini OK: %s for %s", model, ticker)
                return result

            except Exception as exc:
                err = str(exc).lower()

                # Model retired / not found → skip immediately
                if "404" in str(exc) or "not_found" in err or "not found" in err:
                    logger.warning("Model %s not available, skipping.", model)
                    errors.append(f"{model}: not found")
                    break

                # Rate-limited → wait once then retry, else next model
                if "429" in str(exc) or "quota" in err or "rate" in err:
                    if attempt < 2:
                        wait = 15.0
                        m = re.search(r"retry in ([\d.]+)s", str(exc), re.I)
                        if m:
                            wait = min(float(m.group(1)) + 1, 30.0)
                        logger.info("%s rate-limited, waiting %.0fs…", model, wait)
                        time.sleep(wait)
                        continue
                    errors.append(f"{model}: rate-limited")
                    break

                # Other error → give up on this model
                logger.error("Gemini error (%s) for %s: %s", model, ticker, exc)
                errors.append(f"{model}: {exc}")
                break

    summary = "; ".join(errors) if errors else "Unknown failure"
    return {
        "sentiment": 0.0,
        "summary": f"AI analysis unavailable — {summary}",
        "scores": [],
        "model_used": "",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Response parser
# ═══════════════════════════════════════════════════════════════════════════

def _parse_response(raw: str, expected: int) -> dict:
    """Extract JSON from Gemini output with fallback regex extraction."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return {
                    "sentiment": 0.0,
                    "summary": "Could not parse AI response.",
                    "scores": [],
                }
        else:
            return {
                "sentiment": 0.0,
                "summary": "Could not parse AI response.",
                "scores": [],
            }

    sentiment = max(-1.0, min(1.0, float(data.get("overall_sentiment", 0.0))))
    summary = str(data.get("summary", "No summary."))
    scores = [
        max(-1.0, min(1.0, float(s)))
        for s in data.get("scores", [])
    ]

    return {"sentiment": sentiment, "summary": summary, "scores": scores}


# ═══════════════════════════════════════════════════════════════════════════
# AI Trading Strategy (cached 1 h)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def generate_strategy(
    ticker: str,
    sentiment: float,
    predicted_change_pct: float,
    headlines_tuple: tuple[str, ...],
) -> dict:
    """
    Generate a concise AI trading strategy based on the 60-day prediction
    and recent news sentiment.

    Returns:
        {"action": str, "risk": str, "reasoning": str}
    """
    hl_text = "; ".join(headlines_tuple[:3]) if headlines_tuple else "No recent headlines"

    prompt = (
        f"Provide a low-risk trading strategy for the next 2 months for "
        f"{ticker} considering the recent news sentiment.\n\n"
        f"Data:\n"
        f"- Sentiment score: {sentiment:+.2f} (scale -1 to +1)\n"
        f"- 60-day predicted price change: {predicted_change_pct:+.2f}%\n"
        f"- Recent headlines: {hl_text}\n\n"
        "Based on this data, respond ONLY with valid JSON:\n"
        "{\n"
        '  "action": "<Accumulate | Hold | Wait>",\n'
        '  "risk": "<High | Medium | Low>",\n'
        '  "reasoning": "<exactly 2 sentences explaining why>"\n'
        "}"
    )

    client = _get_client()
    errors: list[str] = []

    for model in _MODELS:
        try:
            response = client.models.generate_content(
                model=model, contents=prompt,
            )
            return _parse_strategy_response(response.text)
        except Exception as exc:
            logger.warning("Strategy gen failed with %s: %s", model, exc)
            errors.append(f"{model}: {exc}")
            continue

    return {
        "action": "N/A",
        "risk": "N/A",
        "reasoning": "Strategy generation failed. " + "; ".join(errors),
    }


def _parse_strategy_response(raw: str) -> dict:
    """Parse the JSON strategy output from Gemini."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return {
                    "action": "N/A",
                    "risk": "N/A",
                    "reasoning": "Could not parse strategy response.",
                }
        else:
            return {
                "action": "N/A",
                "risk": "N/A",
                "reasoning": "Could not parse strategy response.",
            }

    return {
        "action": str(data.get("action", "N/A")),
        "risk": str(data.get("risk", "N/A")),
        "reasoning": str(data.get("reasoning", "No reasoning provided.")),
    }
