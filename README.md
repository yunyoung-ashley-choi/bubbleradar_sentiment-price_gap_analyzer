# Bubble Radar — AI Stock Intelligence Dashboard

A secure, cost-optimised Streamlit dashboard that combines **Gemini 2.5 Flash Lite** sentiment analysis with **Prophet** 60-day price forecasting and **AI-generated trading strategies**.

## Architecture

```
Bubble Radar/
├── app.py              # Auth gatekeeper + dashboard UI + strategy panel
├── ai_engine.py        # Gemini 2.5 Flash Lite sentiment & strategy (cached 1 h)
├── forecast.py         # Prophet 60-day forecast + Plotly chart
├── requirements.txt    # Python dependencies
├── .env                # Local secrets  (NEVER commit — gitignored)
├── .env.example        # Template with placeholder values
└── .gitignore          # Protects .env, secrets.toml, config.toml
```

## Features

| Feature | Detail |
|---------|--------|
| **Login** | Session-based auth — only authorised users access the dashboard |
| **8 tickers** | Samsung (삼성전자), KEPCO (한국전력), Hyundai E&C (현대건설), HDC, S&P 500, NASDAQ 100, Tesla, NVIDIA |
| **AI sentiment** | Gemini 2.5 Flash Lite scores top-3 headlines per ticker (-1 to +1) |
| **AI strategy** | Action (Accumulate / Hold / Wait) + Risk level + 2-sentence reasoning |
| **60-day forecast** | Prophet model with 80% confidence bands |
| **Cost control** | Only 3 headlines per ticker; all AI calls cached 1 hour |
| **Dark-themed UI** | Professional Plotly charts + custom CSS |

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/yunyoung-ashley-choi/bubbleradar_sentiment-price_gap_analyzer.git
cd "Bubble Radar"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows note for Prophet:** if `pip install prophet` fails, run these first:
> ```bash
> pip install cmdstanpy
> python -m cmdstanpy.install_cmdstan
> pip install prophet
> ```

### 4. Create the `.env` file

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and fill in your values:

```dotenv
GEMINI_API_KEY=your-gemini-api-key-here
ADMIN_ID=choi
ADMIN_PW=700912
```

Get a free Gemini API key at [Google AI Studio](https://aistudio.google.com/apikey).

### 5. Run the app

```bash
streamlit run app.py
```

---

## Streamlit Cloud Deployment

When deploying on [Streamlit Cloud](https://share.streamlit.io), there is no `.env` file. Instead, secrets are configured through the dashboard.

### Step-by-step

1. Push your code to GitHub (`.env` is gitignored and will **not** be uploaded).
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app.
3. Point it to your repository → branch `main` → file `app.py`.
4. Open **Advanced settings → Secrets** and paste the following:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
ADMIN_ID = "choi"
ADMIN_PW = "700912"
```

5. Click **Deploy**. The app will read these values via `st.secrets` automatically.

### How secrets resolution works

The app resolves each secret using this priority order:

```
Streamlit Cloud secrets (st.secrets)  →  .env file (python-dotenv)  →  hardcoded default
```

This means:
- **Locally**, the app reads from your `.env` file.
- **On Streamlit Cloud**, it reads from the Secrets dashboard.
- No code changes are needed between local and cloud environments.

---

## Security

| Secret | Local | Streamlit Cloud | Protection |
|--------|-------|-----------------|------------|
| `GEMINI_API_KEY` | `.env` | Secrets dashboard | `.env` gitignored |
| `ADMIN_ID` | `.env` | Secrets dashboard | `.env` gitignored |
| `ADMIN_PW` | `.env` | Secrets dashboard | `.env` gitignored |
| `.streamlit/secrets.toml` | File | N/A | gitignored |
| `.streamlit/config.toml` | File | N/A | gitignored |

---

## How It Works

1. **Login** → credentials checked against `ADMIN_ID` / `ADMIN_PW`.
2. **Priority cards** → 30-day prices (yfinance) + top-3 GNews headlines → Gemini 2.5 Flash Lite sentiment.
3. **60-day forecast** → 1-year historical data → Prophet → Plotly chart with confidence bands.
4. **AI strategy** → Gemini generates Action / Risk / Reasoning based on forecast + sentiment.
5. **Caching** → `@st.cache_data(ttl=3600)` prevents duplicate API calls.

## API Costs

| Service | Free Tier | Notes |
|---------|-----------|-------|
| Gemini 2.5 Flash Lite | 30 RPM / 1M TPM | Cheapest Gemini model |
| GNews | Unlimited | Free Google News scraping |
| yfinance | Unlimited | Free Yahoo Finance data |

With top-3 headlines and 1-hour caching, a full day of active use costs effectively **$0**.
