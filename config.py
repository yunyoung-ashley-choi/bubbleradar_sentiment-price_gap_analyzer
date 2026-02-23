"""
Configuration constants for Bubble Radar application.
Sector-to-ticker mappings, UI settings, and analysis thresholds.
"""

from dataclasses import dataclass


SECTOR_ETF_MAP: dict[str, str] = {
    # --- Individual Stocks ---
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

SECTOR_SEARCH_KEYWORDS: dict[str, str] = {
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

GEMINI_MODELS: list[str] = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]


@dataclass(frozen=True)
class AnalysisThresholds:
    """Thresholds for bubble index classification."""
    OVERHEATED: float = 0.40
    UNDERVALUED: float = -0.40
    PRICE_NORM_FACTOR: float = 15.0
    LOOKBACK_DAYS: int = 30
    MAX_HEADLINES: int = 25


@dataclass(frozen=True)
class UIConfig:
    """UI display settings."""
    APP_TITLE: str = "Bubble Radar"
    APP_SUBTITLE: str = "Sentiment-Price Gap Analyzer"
    PAGE_ICON: str = "🫧"
    LAYOUT: str = "wide"


THRESHOLDS = AnalysisThresholds()
UI = UIConfig()
