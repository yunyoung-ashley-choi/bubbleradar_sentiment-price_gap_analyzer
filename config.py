"""
Configuration for Bubble Radar v2 — AI Stock Intelligence Dashboard.
Priority assets, sector mappings, prediction settings, and UI config.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Priority Assets — displayed as metric cards at the top of the dashboard
# ---------------------------------------------------------------------------
PRIORITY_ASSETS: dict[str, dict[str, str]] = {
    "Samsung Electronics": {"ticker": "005930.KS", "currency": "₩"},
    "KEPCO":               {"ticker": "015760.KS", "currency": "₩"},
    "Hyundai E&C":         {"ticker": "000720.KS", "currency": "₩"},
    "HDC":                 {"ticker": "294870.KS", "currency": "₩"},
    "S&P 500":             {"ticker": "^GSPC",     "currency": "$"},
    "KOSPI 200":           {"ticker": "069500.KS", "currency": "₩"},
}

# ---------------------------------------------------------------------------
# Sector / ETF mappings — includes priority assets + existing universe
# ---------------------------------------------------------------------------
SECTOR_ETF_MAP: dict[str, str] = {
    # --- Priority Assets ---
    "Samsung Electronics": "005930.KS",
    "KEPCO": "015760.KS",
    "Hyundai E&C": "000720.KS",
    "HDC": "294870.KS",
    "S&P 500": "^GSPC",
    "KOSPI 200": "069500.KS",
    # --- Individual US Stocks ---
    "NVIDIA": "NVDA",
    "Tesla": "TSLA",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google (Alphabet)": "GOOGL",
    "Meta (Facebook)": "META",
    "AMD": "AMD",
    "Intel": "INTC",
    "Palantir": "PLTR",
    # --- Crypto ---
    "Bitcoin": "BITO",
    "Ethereum": "ETHA",
    # --- Sector ETFs ---
    "S&P 500 (SPY)": "SPY",
    "NASDAQ 100 (QQQ)": "QQQ",
    "Semiconductors (SOXX)": "SOXX",
    "AI & Robotics (BOTZ)": "BOTZ",
    "Cybersecurity (CIBR)": "CIBR",
    "Cloud Computing (SKYY)": "SKYY",
    "Healthcare (XLV)": "XLV",
    "Biotech (XBI)": "XBI",
    "Energy (XLE)": "XLE",
    "Clean Energy (ICLN)": "ICLN",
    "Gold (GLD)": "GLD",
    "Real Estate (VNQ)": "VNQ",
    "Financials (XLF)": "XLF",
    "Aerospace & Defense (ITA)": "ITA",
}

# ---------------------------------------------------------------------------
# News search keywords per asset / sector
# ---------------------------------------------------------------------------
SECTOR_SEARCH_KEYWORDS: dict[str, str] = {
    "Samsung Electronics": "Samsung Electronics semiconductor memory chip DRAM",
    "KEPCO": "KEPCO Korea Electric Power utility energy grid",
    "Hyundai E&C": "Hyundai Engineering Construction infrastructure Korea",
    "HDC": "HDC Hyundai Development real estate construction Korea",
    "S&P 500": "S&P 500 stock market Wall Street index economy",
    "KOSPI 200": "KOSPI Korea stock market index economy",
    "NVIDIA": "NVIDIA GPU AI chips Jensen Huang",
    "Tesla": "Tesla Elon Musk EV electric vehicle",
    "Apple": "Apple iPhone Mac Vision Pro",
    "Microsoft": "Microsoft Azure Copilot Windows",
    "Amazon": "Amazon AWS e-commerce cloud",
    "Google (Alphabet)": "Google Alphabet Gemini search AI",
    "Meta (Facebook)": "Meta Facebook Instagram Threads Zuckerberg",
    "AMD": "AMD Ryzen EPYC Lisa Su chips",
    "Intel": "Intel foundry chips processor",
    "Palantir": "Palantir AI data analytics government",
    "Bitcoin": "Bitcoin BTC crypto cryptocurrency halving",
    "Ethereum": "Ethereum ETH crypto blockchain DeFi",
    "S&P 500 (SPY)": "S&P 500 stock market Wall Street index",
    "NASDAQ 100 (QQQ)": "NASDAQ tech stocks growth",
    "Semiconductors (SOXX)": "semiconductor chips TSMC foundry wafer",
    "AI & Robotics (BOTZ)": "artificial intelligence robotics machine learning",
    "Cybersecurity (CIBR)": "cybersecurity data breach hacking security",
    "Cloud Computing (SKYY)": "cloud computing SaaS AWS Azure",
    "Healthcare (XLV)": "healthcare pharma medical FDA",
    "Biotech (XBI)": "biotech biotechnology drug clinical trials",
    "Energy (XLE)": "oil gas energy crude petroleum OPEC",
    "Clean Energy (ICLN)": "solar wind renewable energy EV charging",
    "Gold (GLD)": "gold bullion precious metals safe haven",
    "Real Estate (VNQ)": "real estate housing REIT mortgage",
    "Financials (XLF)": "banks financial interest rates Fed",
    "Aerospace & Defense (ITA)": "aerospace defense military space contracts",
}

# Kept for backward compatibility with app.py (Gemini-based flow)
GEMINI_MODELS: list[str] = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]


# ---------------------------------------------------------------------------
# Prediction / ML settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictionConfig:
    """Prophet model and feature-engineering settings."""
    FORECAST_DAYS: int = 60
    TRAINING_LOOKBACK_DAYS: int = 365
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    MA_SHORT: int = 20
    MA_LONG: int = 60
    SENTIMENT_ROLLING_WINDOW: int = 7
    PROPHET_CHANGEPOINT_SCALE: float = 0.05
    PROPHET_INTERVAL_WIDTH: float = 0.80
    MIN_TRAINING_POINTS: int = 60


@dataclass(frozen=True)
class AnalysisThresholds:
    """Thresholds for bubble-index and trend classification."""
    OVERHEATED: float = 0.40
    UNDERVALUED: float = -0.40
    PRICE_NORM_FACTOR: float = 15.0
    LOOKBACK_DAYS: int = 30
    MAX_HEADLINES: int = 25
    BULLISH_PCT: float = 2.0
    BEARISH_PCT: float = -2.0


@dataclass(frozen=True)
class UIConfig:
    """UI display settings."""
    APP_TITLE: str = "Bubble Radar"
    APP_SUBTITLE: str = "AI Stock Intelligence Dashboard"
    PAGE_ICON: str = "🫧"
    LAYOUT: str = "wide"


PREDICTION = PredictionConfig()
THRESHOLDS = AnalysisThresholds()
UI = UIConfig()
