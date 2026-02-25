"""
Bubble Radar — Secure AI Stock Intelligence Dashboard.

Authentication gatekeeper → cost-optimised Gemini 2.5 Flash Lite sentiment
→ Prophet 60-day price forecast → interactive Plotly visualisation.

Entry point:  streamlit run app.py
"""

import logging
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ai_engine import analyze_sentiment, fetch_top_headlines, generate_strategy
from forecast import create_forecast_chart, fetch_price_data, run_forecast

# ── Load environment variables ──────────────────────────────────────────
load_dotenv()


def _resolve_secret(key: str, default: str = "") -> str:
    """Streamlit Cloud secrets take priority over .env variables."""
    try:
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except FileNotFoundError:
        pass
    return os.environ.get(key, default)


# Resolve all secrets once at startup so ai_engine.py can read them
# from os.environ without needing its own st.secrets logic.
os.environ["GEMINI_API_KEY"] = _resolve_secret("GEMINI_API_KEY")
os.environ["ADMIN_ID"] = _resolve_secret("ADMIN_ID", "choi")
os.environ["ADMIN_PW"] = _resolve_secret("ADMIN_PW", "700912")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Bubble Radar",
    page_icon="🫧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Tickers with Korean labels ──────────────────────────────────────────
TICKERS: dict[str, dict[str, str]] = {
    "Samsung Electronics (삼성전자)": {"ticker": "005930.KS", "currency": "₩"},
    "KEPCO (한국전력)":               {"ticker": "015760.KS", "currency": "₩"},
    "Hyundai E&C (현대건설)":         {"ticker": "000720.KS", "currency": "₩"},
    "HDC (HDC현대산업개발)":           {"ticker": "294870.KS", "currency": "₩"},
    "S&P 500 (S&P 500 지수)":        {"ticker": "^GSPC",     "currency": "$"},
    "NASDAQ 100 (나스닥 100)":        {"ticker": "^NDX",      "currency": "$"},
    "Tesla (테슬라)":                 {"ticker": "TSLA",      "currency": "$"},
    "NVIDIA (엔비디아)":              {"ticker": "NVDA",      "currency": "$"},
}


def _fmt(price: float, cur: str) -> str:
    return f"₩{price:,.0f}" if cur == "₩" else f"${price:,.2f}"


# ═══════════════════════════════════════════════════════════════════════════
#  1.  LOGIN PAGE
# ═══════════════════════════════════════════════════════════════════════════

def _login_page() -> None:
    """Full-screen login form — blocks all dashboard access until success."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none; }
        .login-wrap {
            max-width: 420px;
            margin: 6rem auto 0 auto;
            padding: 2.5rem 2rem;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-radius: 16px;
            border: 1px solid rgba(99,110,250,0.3);
            text-align: center;
        }
        .login-wrap h1 { color: #e0e0ff; font-size: 2.2rem; margin-bottom: 0.2rem; }
        .login-wrap p  { color: #a0a0c0; font-size: 0.95rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-wrap">'
        "<h1>🫧 Bubble Radar</h1>"
        "<p>AI Stock Intelligence Dashboard</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    _spacer, form_col, _spacer2 = st.columns([1, 2, 1])
    with form_col:
        with st.form("login_form"):
            uid = st.text_input("User ID", placeholder="Enter your ID")
            pwd = st.text_input(
                "Password", type="password", placeholder="Enter password",
            )
            submitted = st.form_submit_button(
                "🔐 Sign In", use_container_width=True,
            )

        if submitted:
            expected_id = os.environ.get("ADMIN_ID", "choi")
            expected_pw = os.environ.get("ADMIN_PW", "700912")

            if uid == expected_id and pwd == expected_pw:
                st.session_state["authenticated"] = True
                st.session_state["user_id"] = uid
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.", icon="🚫")


# ═══════════════════════════════════════════════════════════════════════════
#  2.  DASHBOARD — CSS
# ═══════════════════════════════════════════════════════════════════════════

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Header */
        .main-header {
            background: linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
            padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
            border: 1px solid rgba(99,110,250,0.3);
        }
        .main-header h1 { color:#e0e0ff; margin:0; font-size:2rem; }
        .main-header p  { color:#a0a0c0; margin:0.3rem 0 0 0; font-size:1rem; }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: rgba(30,30,60,0.5);
            border: 1px solid rgba(99,110,250,0.2);
            border-radius: 10px; padding: 1rem;
        }

        /* Asset cards */
        .card {
            background: linear-gradient(135deg,rgba(30,30,60,0.7),rgba(20,20,50,0.9));
            border: 1px solid rgba(99,110,250,0.25);
            border-radius: 12px; padding: 1.2rem;
            margin-bottom: 0.8rem; min-height: 230px;
            transition: border-color 0.2s;
        }
        .card:hover { border-color: rgba(99,110,250,0.6); }
        .card h4   { color:#e0e0ff; margin:0 0 0.5rem 0; font-size:1rem; }
        .card .price { color:#fff; font-size:1.4rem; font-weight:700; }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg,#111827,#1f2937);
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color:#f9fafb !important; font-weight:700 !important;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p {
            color:#d1d5db !important;
        }
        section[data-testid="stSidebar"] .stCaption p {
            color:#9ca3af !important; font-size:0.85rem !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] span,
        section[data-testid="stSidebar"] [data-baseweb="select"] > div > div {
            color:#111827 !important;
            -webkit-text-fill-color:#111827 !important;
        }
        [data-baseweb="popover"] li,
        [data-baseweb="popover"] li span {
            color:#111827 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  3.  DASHBOARD — SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> str:
    """Build sidebar; return the name of the ticker selected for forecasting."""
    with st.sidebar:
        st.markdown(
            f"### Welcome, **{st.session_state.get('user_id', 'User')}**"
        )

        st.markdown("---")
        st.markdown("### Settings")
        st.caption("AI Model: **Gemini 2.5 Flash Lite**")
        st.caption("Headlines per ticker: **3**")
        st.caption("Sentiment cache: **1 hour**")
        st.caption("Forecast horizon: **60 days**")

        st.markdown("---")
        selected = st.selectbox(
            "Select ticker for 60-day forecast",
            options=list(TICKERS.keys()),
            index=0,
        )
        ticker = TICKERS[selected]["ticker"]
        st.info(f"Tracking: **{ticker}**", icon="📈")

        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            for k in [
                k for k in st.session_state if k.startswith("forecast_")
            ]:
                del st.session_state[k]
            st.rerun()

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state.pop("user_id", None)
            st.rerun()

        st.markdown("---")
        st.markdown(
            "<p style='color:#9ca3af;font-size:0.8rem;text-align:center;'>"
            "Powered by Prophet · Gemini 2.5 Flash Lite · yfinance"
            "</p>",
            unsafe_allow_html=True,
        )

    return selected


# ═══════════════════════════════════════════════════════════════════════════
#  4.  PRIORITY TICKER CARDS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _load_all_ticker_data() -> dict:
    """Fetch 30-day prices + top-3 sentiment for every priority ticker."""
    data: dict = {}
    for name, info in TICKERS.items():
        ticker = info["ticker"]
        currency = info["currency"]
        try:
            pdf = fetch_price_data(ticker, days=30)
            cur = pdf["Close"].iloc[-1]
            prev = pdf["Close"].iloc[-2] if len(pdf) > 1 else cur
            pct = ((cur - prev) / prev) * 100

            hl = fetch_top_headlines(ticker, max_results=3)
            ai = analyze_sentiment(ticker, tuple(hl))

            data[name] = {
                "ticker": ticker,
                "currency": currency,
                "price": cur,
                "change": pct,
                "sentiment": ai["sentiment"],
                "summary": ai["summary"],
                "error": None,
            }
        except Exception as exc:
            logger.error("Load failed for %s: %s", name, exc)
            data[name] = {
                "ticker": ticker,
                "currency": currency,
                "error": str(exc),
            }
    return data


def _render_cards(data: dict) -> None:
    """Draw ticker cards in rows of 4."""
    names = list(data.keys())
    for row_start in range(0, len(names), 4):
        row_names = names[row_start : row_start + 4]
        cols = st.columns(4)
        for col, name in zip(cols, row_names):
            with col:
                info = data[name]
                if info.get("error"):
                    st.error(f"**{name}**\n\n{info['error']}")
                    continue

                chg = info["change"]
                chg_c = "#00CC96" if chg >= 0 else "#EF553B"
                s = info["sentiment"]
                s_c = "#00CC96" if s >= 0 else "#EF553B"

                if s > 0.1 and chg > 0:
                    trend = '<span style="color:#00CC96;font-weight:700">📈 Bullish</span>'
                elif s < -0.1 and chg < 0:
                    trend = '<span style="color:#EF553B;font-weight:700">📉 Bearish</span>'
                else:
                    trend = '<span style="color:#636EFA;font-weight:700">➡️ Neutral</span>'

                summary = info["summary"]
                if len(summary) > 90:
                    summary = summary[:87] + "…"

                # Extract short English name for the card title
                short_name = name.split("(")[0].strip()

                st.markdown(
                    f"""
                    <div class="card">
                        <h4>{short_name}
                            <span style="color:#888;font-weight:400;font-size:0.85rem;">
                            ({info['ticker']})</span></h4>
                        <div class="price">{_fmt(info['price'], info['currency'])}</div>
                        <div style="color:{chg_c};font-size:0.9rem">
                            Daily: {chg:+.2f}%</div>
                        <div style="color:{s_c};font-size:0.85rem;margin-top:0.3rem">
                            Sentiment: {s:+.2f}</div>
                        <div style="font-size:0.8rem;color:#a0a0c0;margin-top:0.3rem">
                            {summary}</div>
                        <div style="margin-top:0.5rem">60-Day: {trend}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════
#  5.  AI TRADING STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

def _render_strategy(
    name: str,
    ticker: str,
    predicted_change_pct: float,
    card_data: dict,
) -> None:
    """Show the AI-generated trading strategy below the forecast chart."""
    st.markdown("---")
    st.markdown("#### AI Trading Strategy")

    cd = card_data.get(name, {})
    sentiment = cd.get("sentiment", 0.0)

    headlines = fetch_top_headlines(ticker, max_results=3)

    with st.spinner("Generating AI strategy…"):
        strategy = generate_strategy(
            ticker,
            sentiment,
            predicted_change_pct,
            tuple(headlines),
        )

    action = strategy["action"]
    risk = strategy["risk"]
    reasoning = strategy["reasoning"]

    action_colors = {
        "Accumulate": ("#00CC96", "📈"),
        "Hold":       ("#636EFA", "⚖️"),
        "Wait":       ("#FFA15A", "⏳"),
    }
    risk_colors = {
        "Low":    "#00CC96",
        "Medium": "#FFA15A",
        "High":   "#EF553B",
    }

    a_color, a_icon = action_colors.get(action, ("#888", "📊"))
    r_color = risk_colors.get(risk, "#888")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(30,30,60,0.7), rgba(20,20,50,0.9));
            border-left: 5px solid {a_color};
            border-radius: 10px;
            padding: 1.5rem 2rem;
            margin: 0.5rem 0 1rem 0;
        ">
            <div style="display:flex; gap:2rem; flex-wrap:wrap; margin-bottom:1rem;">
                <div>
                    <span style="color:#9ca3af;font-size:0.8rem;">ACTION</span><br/>
                    <span style="color:{a_color};font-size:1.3rem;font-weight:700;">
                        {a_icon} {action}</span>
                </div>
                <div>
                    <span style="color:#9ca3af;font-size:0.8rem;">RISK LEVEL</span><br/>
                    <span style="color:{r_color};font-size:1.3rem;font-weight:700;">
                        {risk}</span>
                </div>
                <div>
                    <span style="color:#9ca3af;font-size:0.8rem;">SENTIMENT</span><br/>
                    <span style="color:{'#00CC96' if sentiment >= 0 else '#EF553B'};
                           font-size:1.3rem;font-weight:700;">
                        {sentiment:+.2f}</span>
                </div>
            </div>
            <div style="color:#c0c0d0;font-size:0.95rem;line-height:1.6;">
                <strong style="color:#e0e0ff;">Reasoning:</strong> {reasoning}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "※ 이 전략은 AI 분석에 기반한 참고 자료이며, 투자 조언이 아닙니다. "
        "투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다."
    )


# ═══════════════════════════════════════════════════════════════════════════
#  6.  60-DAY FORECAST SECTION
# ═══════════════════════════════════════════════════════════════════════════

def _render_forecast(name: str, card_data: dict) -> None:
    """Prophet forecast panel with session-cached results."""
    info = TICKERS[name]
    ticker = info["ticker"]
    currency = info["currency"]
    cache_key = f"forecast_{name}"

    run = st.button(
        "🔍 Run 60-Day Forecast",
        use_container_width=True,
        type="primary",
    )

    if run:
        with st.spinner("Fetching 1-year history & training Prophet…"):
            try:
                pdf = fetch_price_data(ticker, days=365)
                fdf = run_forecast(pdf, forecast_days=60)
                st.session_state[cache_key] = fdf
            except Exception as exc:
                st.error(f"Forecast failed: {exc}")
                return

    fdf = st.session_state.get(cache_key)

    if fdf is not None:
        chart = create_forecast_chart(fdf, ticker, name, currency)
        st.plotly_chart(chart, use_container_width=True)

        hist = fdf.dropna(subset=["y"])
        future = fdf[fdf["y"].isna()]

        if not future.empty and not hist.empty:
            last = hist["y"].iloc[-1]
            pred = future["yhat"].iloc[-1]
            delta = ((pred - last) / last) * 100
            trend = (
                "Bullish" if delta > 2
                else ("Bearish" if delta < -2 else "Neutral")
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Current Price", _fmt(last, currency))
            with c2:
                st.metric(
                    "60-Day Predicted",
                    _fmt(pred, currency),
                    f"{delta:+.2f}%",
                )
            with c3:
                arrow = (
                    "📈" if trend == "Bullish"
                    else ("📉" if trend == "Bearish" else "➡️")
                )
                st.metric("Trend", trend, arrow)

            # ── AI Trading Strategy ─────────────────────────────
            _render_strategy(name, ticker, delta, card_data)

    else:
        cd = card_data.get(name)
        if cd and not cd.get("error"):
            st.info(
                f"**{name}** — {cd.get('summary', 'No summary.')}",
                icon="📊",
            )
        else:
            st.info(
                "Click **Run 60-Day Forecast** to train a Prophet model "
                "and visualise the predicted path.",
                icon="🔮",
            )


# ═══════════════════════════════════════════════════════════════════════════
#  7.  MAIN DASHBOARD PAGE
# ═══════════════════════════════════════════════════════════════════════════

def _dashboard() -> None:
    _inject_css()

    st.markdown(
        """
        <div class="main-header">
            <h1>🫧 Bubble Radar</h1>
            <p>AI Stock Intelligence Dashboard — Gemini 2.5 Flash Lite · 60-Day Prophet Forecast</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = _render_sidebar()

    # Priority cards
    with st.spinner("Loading priority tickers…"):
        card_data = _load_all_ticker_data()
    _render_cards(card_data)

    st.markdown("---")

    # Forecast section
    st.markdown(f"### 60-Day Forecast: {selected}")
    _render_forecast(selected, card_data)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 2rem 0;">
            <p style="color:#6b7280;font-size:0.85rem;margin:0;">
                &copy; 2026 Bubble Radar.
                해당 웹사이트는 <strong>최윤영</strong>과 <strong>AI</strong>의 합작품입니다.
            </p>
            <p style="color:#4b5563;font-size:0.75rem;margin:0.3rem 0 0 0;">
                Powered by Prophet · Gemini 2.5 Flash Lite · yfinance · Google News
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    if st.session_state.get("authenticated"):
        _dashboard()
    else:
        _login_page()


if __name__ == "__main__":
    main()
