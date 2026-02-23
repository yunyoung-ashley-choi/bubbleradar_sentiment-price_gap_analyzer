# Bubble Radar: Sentiment-Price Gap Analyzer

A professional Streamlit dashboard that compares stock price trends against news sentiment to identify potential market bubbles or undervalued opportunities across multiple sectors.

## Architecture

```
Bubble Radar/
├── app.py                  # Main Streamlit application
├── config.py               # Sector/ETF mappings, thresholds, UI constants
├── data_fetcher.py         # yfinance price data + NewsAPI headlines
├── sentiment_analyzer.py   # Google Gemini AI sentiment scoring
├── bubble_index.py         # Bubble index calculation engine
├── charts.py               # Plotly interactive chart generators
├── requirements.txt        # Python dependencies
└── .streamlit/
    └── secrets.toml.example  # API key template
```

## Features

- **20 sector categories** with mapped ETF tickers (Technology, Semiconductors, AI, Healthcare, Crypto, etc.)
- **Real-time price data** via yfinance (30-day lookback)
- **News sentiment analysis** powered by Google Gemini AI
- **Interactive dual-axis chart** (price + sentiment overlay) built with Plotly
- **Bubble Index gauge** with Overheated / Stable / Undervalued classification
- **Professional dark-themed UI** with KPI metric cards

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

**Option A — `.streamlit/secrets.toml` (recommended):**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and add your actual keys:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
NEWSAPI_KEY = "your-newsapi-key"
```

**Option B — Sidebar input:** Enter keys directly in the app's sidebar at runtime.

### 3. Run the app

```bash
streamlit run app.py
```

## API Keys

| Service | Free Tier | Get Key |
|---------|-----------|---------|
| Google Gemini | 60 req/min | [Google AI Studio](https://aistudio.google.com/apikey) |
| NewsAPI | 100 req/day | [newsapi.org/register](https://newsapi.org/register) |

> **Note:** NewsAPI's free tier only works from `localhost`. For cloud deployment, a paid plan is required.

## How the Bubble Index Works

```
Bubble Index = Price Momentum Score − Sentiment Score
```

- **Price Momentum Score**: 30-day price change normalized to [-1, +1] via `tanh(change% / 15)`
- **Sentiment Score**: AI-aggregated headline sentiment from -1 (bearish) to +1 (bullish)
- **> +0.40**: Overheated — price running ahead of fundamentals
- **< −0.40**: Undervalued — positive sentiment not yet reflected in price
- **Between**: Stable — price and sentiment reasonably aligned
