"""
Bubble Radar: Sentiment-Price Gap Analyzer
Main Streamlit application entry point.
"""

import logging

import pandas as pd
import streamlit as st

from bubble_index import MarketStatus, compute_bubble_index
from charts import create_bubble_gauge, create_price_sentiment_chart
from config import GEMINI_MODELS, SECTOR_ETF_MAP, THRESHOLDS, UI
from data_fetcher import build_news_dataframe, fetch_news_headlines, fetch_stock_data
from sentiment_analyzer import (
    analyze_sentiment,
    build_sentiment_timeseries,
    configure_gemini,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{UI.APP_TITLE}: {UI.APP_SUBTITLE}",
    page_icon=UI.PAGE_ICON,
    layout=UI.LAYOUT,
    initial_sidebar_state="expanded",
)


def _inject_custom_css() -> None:
    """Apply custom CSS for professional styling."""
    st.markdown(
        """
        <style>
        /* Main header */
        .main-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(99, 110, 250, 0.3);
        }
        .main-header h1 {
            color: #e0e0ff;
            margin: 0;
            font-size: 2rem;
        }
        .main-header p {
            color: #a0a0c0;
            margin: 0.3rem 0 0 0;
            font-size: 1rem;
        }

        /* Status badges */
        .status-overheated {
            background: linear-gradient(135deg, #ff4444, #cc0000);
            color: white;
            padding: 0.4rem 1.2rem;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1.1rem;
            display: inline-block;
            text-align: center;
        }
        .status-stable {
            background: linear-gradient(135deg, #636EFA, #4040cc);
            color: white;
            padding: 0.4rem 1.2rem;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1.1rem;
            display: inline-block;
            text-align: center;
        }
        .status-undervalued {
            background: linear-gradient(135deg, #00CC96, #009966);
            color: white;
            padding: 0.4rem 1.2rem;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1.1rem;
            display: inline-block;
            text-align: center;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: rgba(30, 30, 60, 0.5);
            border: 1px solid rgba(99, 110, 250, 0.2);
            border-radius: 10px;
            padding: 1rem;
        }

        /* ---- Sidebar readability ---- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #f9fafb !important;
            font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] label {
            color: #d1d5db !important;
            font-weight: 500 !important;
            font-size: 0.95rem !important;
        }
        section[data-testid="stSidebar"] p {
            color: #d1d5db !important;
        }
        section[data-testid="stSidebar"] .stCaption p {
            color: #9ca3af !important;
            font-size: 0.85rem !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(156, 163, 175, 0.3) !important;
        }
        section[data-testid="stSidebar"] .stAlert p {
            font-size: 0.9rem !important;
        }

        /* ---- Dropdown / selectbox fix ---- */
        /* Selected value text: must be dark on the white input bg */
        section[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] [data-baseweb="select"] span,
        section[data-testid="stSidebar"] [data-baseweb="select"] div[aria-selected],
        section[data-testid="stSidebar"] [data-baseweb="select"] > div > div {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }
        /* The SVG arrow icon inside dropdowns */
        section[data-testid="stSidebar"] [data-baseweb="select"] svg {
            fill: #6b7280 !important;
        }
        /* Text input fields (password keys) */
        section[data-testid="stSidebar"] .stTextInput input {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }
        /* Dropdown popup option list */
        [data-baseweb="popover"] li,
        [data-baseweb="popover"] li span,
        [data-baseweb="menu"] li,
        [data-baseweb="menu"] li span {
            color: #111827 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_api_keys() -> tuple[str, str]:
    """
    Retrieve API keys from st.secrets or sidebar input.
    Returns (gemini_key, newsapi_key).
    """
    gemini_key = ""
    newsapi_key = ""

    try:
        gemini_key = st.secrets.get("GEMINI_API_KEY", "")
        newsapi_key = st.secrets.get("NEWSAPI_KEY", "")
    except FileNotFoundError:
        pass

    with st.sidebar:
        st.markdown("### API Configuration")

        if not gemini_key:
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="Get your key at https://aistudio.google.com/apikey",
            )
        else:
            st.success("Gemini API Key loaded from secrets", icon="\u2705")

        if not newsapi_key:
            newsapi_key = st.text_input(
                "NewsAPI Key",
                type="password",
                help="Get your key at https://newsapi.org/register",
            )
        else:
            st.success("NewsAPI Key loaded from secrets", icon="\u2705")

    return gemini_key, newsapi_key


def _render_sidebar() -> tuple[str, str]:
    """Render sidebar controls and return (selected_sector, selected_model)."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Asset / Sector")
        sector = st.selectbox(
            "Choose what to analyze",
            options=list(SECTOR_ETF_MAP.keys()),
            index=0,
        )

        ticker = SECTOR_ETF_MAP[sector]
        st.info(f"Tracking: **{ticker}**", icon="\U0001F4C8")

        st.markdown("---")
        st.markdown("### Gemini Model")
        model = st.selectbox(
            "Preferred AI model",
            options=GEMINI_MODELS,
            index=0,
            help="Auto-fallback to other models if this one is rate-limited.",
        )

        st.markdown("---")
        st.markdown("### Analysis Settings")
        st.caption(f"Lookback: **{THRESHOLDS.LOOKBACK_DAYS} days**")
        st.caption(f"Max headlines: **{THRESHOLDS.MAX_HEADLINES}**")
        st.caption(f"Overheated: **> {THRESHOLDS.OVERHEATED}**")
        st.caption(f"Undervalued: **< {THRESHOLDS.UNDERVALUED}**")

        st.markdown("---")
        st.markdown(
            "<p style='color: #9ca3af !important; font-size: 0.8rem; text-align: center;'>"
            "Built with Streamlit, yfinance, NewsAPI &amp; Gemini"
            "</p>",
            unsafe_allow_html=True,
        )

    return sector, model


def _render_header() -> None:
    """Render the main dashboard header."""
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{UI.PAGE_ICON} {UI.APP_TITLE}</h1>
            <p>{UI.APP_SUBTITLE} — Identify market bubbles and hidden opportunities</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_status_badge(status: MarketStatus) -> None:
    """Render a colored status badge."""
    css_class = {
        MarketStatus.OVERHEATED: "status-overheated",
        MarketStatus.STABLE: "status-stable",
        MarketStatus.UNDERVALUED: "status-undervalued",
    }[status]

    icon = {
        MarketStatus.OVERHEATED: "\U0001F525",
        MarketStatus.STABLE: "\u2696\ufe0f",
        MarketStatus.UNDERVALUED: "\U0001F4A1",
    }[status]

    st.markdown(
        f'<span class="{css_class}">{icon} {status.value}</span>',
        unsafe_allow_html=True,
    )


def _render_metrics(
    price_df: pd.DataFrame,
    bubble_result,
    ticker: str,
) -> None:
    """Render the top-row KPI metric cards."""
    latest_price = price_df["Close"].iloc[-1]
    prev_price = price_df["Close"].iloc[0]
    price_delta = latest_price - prev_price

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label=f"{ticker} Current Price",
            value=f"${latest_price:.2f}",
            delta=f"{bubble_result.price_change_pct:+.2f}%",
        )
    with col2:
        st.metric(
            label="30D Price Change",
            value=f"${price_delta:+.2f}",
            delta=f"{bubble_result.price_change_pct:+.2f}%",
        )
    with col3:
        sentiment_val = bubble_result.sentiment_score
        st.metric(
            label="Sentiment Score",
            value=f"{sentiment_val:+.3f}",
            delta="Positive" if sentiment_val > 0 else ("Negative" if sentiment_val < 0 else "Neutral"),
            delta_color="normal" if sentiment_val >= 0 else "inverse",
        )
    with col4:
        st.metric(
            label="Bubble Index",
            value=f"{bubble_result.bubble_index:+.4f}",
        )


_NEXT_ACTION_MAP: dict[MarketStatus, dict[str, str]] = {
    MarketStatus.OVERHEATED: {
        "icon": "\U0001F525",
        "status_ko": "과열 (Overheated)",
        "color": "#ff4444",
        "meaning_ko": (
            "주가 상승 속도가 뉴스 심리보다 훨씬 빠릅니다. "
            "시장이 실제 펀더멘털보다 과도한 낙관을 반영하고 있을 가능성이 높습니다."
        ),
        "actions_ko": [
            "신규 매수 자제 — 현재 가격이 고평가 구간일 수 있습니다",
            "보유 중이라면 일부 이익 실현(차익 매도) 검토",
            "손절매 라인 설정 — 급격한 조정에 대비하세요",
            "거래량 및 변동성 지표를 주시하세요",
            "FOMO(놓칠까 봐 두려운 심리)에 휩쓸리지 마세요",
        ],
    },
    MarketStatus.STABLE: {
        "icon": "\u2696\ufe0f",
        "status_ko": "안정 (Stable)",
        "color": "#636EFA",
        "meaning_ko": (
            "주가 흐름과 뉴스 심리가 균형 잡혀 있습니다. "
            "시장이 현재 뉴스를 합리적으로 반영하고 있는 상태입니다."
        ),
        "actions_ko": [
            "현재 포지션 유지 — 급격한 조정 가능성 낮음",
            "정기적으로 뉴스와 가격을 모니터링하세요",
            "분할 매수(DCA) 전략 고려 가능",
            "갑작스러운 뉴스 변화에 대비한 알림 설정 권장",
            "장기 투자 관점에서 안정적인 구간입니다",
        ],
    },
    MarketStatus.UNDERVALUED: {
        "icon": "\U0001F4A1",
        "status_ko": "저평가 (Undervalued)",
        "color": "#00CC96",
        "meaning_ko": (
            "뉴스 심리가 긍정적인데 비해 주가가 아직 충분히 반영하지 못하고 있습니다. "
            "시장이 좋은 뉴스를 아직 '소화'하지 못한 상태 — 매수 기회일 수 있습니다."
        ),
        "actions_ko": [
            "매수 기회 탐색 — 현재 가격이 저평가 구간일 수 있습니다",
            "분할 매수(DCA)로 리스크를 분산하세요",
            "긍정적 심리의 원인(뉴스)을 직접 확인하세요",
            "단기 급등보다는 중장기 관점으로 접근 권장",
            "해당 섹터의 펀더멘털(실적, 성장성)도 함께 분석하세요",
        ],
    },
}


def _render_next_action(bubble_result) -> None:
    """Render the Next Action recommendation panel in Korean."""
    info = _NEXT_ACTION_MAP[bubble_result.status]

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(30,30,60,0.7), rgba(20,20,50,0.9));
            border-left: 5px solid {info['color']};
            border-radius: 10px;
            padding: 1.5rem 2rem;
            margin: 1rem 0;
        ">
            <h4 style="color: #f0f0ff; margin-top: 0;">
                {info['icon']} 다음 행동 가이드 (Next Action)
            </h4>
            <p style="color: {info['color']}; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">
                상태: {info['status_ko']}
            </p>
            <p style="color: #c0c0d0; font-size: 0.95rem; margin-bottom: 1rem;">
                {info['meaning_ko']}
            </p>
            <div style="color: #e0e0f0;">
                {''.join(
                    f'<div style="padding: 0.35rem 0; display: flex; align-items: flex-start;">'
                    f'<span style="color: {info["color"]}; margin-right: 0.5rem; font-weight: bold;">✓</span>'
                    f'<span>{action}</span></div>'
                    for action in info['actions_ko']
                )}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "※ 이 분석은 뉴스 심리와 가격 데이터를 기반으로 한 참고 자료이며, "
        "투자 조언이 아닙니다. 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다."
    )


def _render_news_table(news_df: pd.DataFrame) -> None:
    """Render the latest news headlines in a clean table."""
    st.markdown("#### Latest News Headlines")

    if news_df.empty:
        st.info("No news articles found for this sector and time range.")
        return

    display_df = news_df[["date", "title", "source"]].head(20).copy()
    display_df.columns = ["Date", "Headline", "Source"]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", width="small"),
            "Headline": st.column_config.TextColumn("Headline", width="large"),
            "Source": st.column_config.TextColumn("Source", width="small"),
        },
    )


def _cache_key(sector: str) -> str:
    """Build a unique session-state cache key per sector."""
    return f"results_{sector}"


# ---------------------------------------------------------------------------
# Main application flow
# ---------------------------------------------------------------------------
def main() -> None:
    """Main application entry point."""
    _inject_custom_css()
    _render_header()

    gemini_key, newsapi_key = _get_api_keys()
    sector, gemini_model = _render_sidebar()
    ticker = SECTOR_ETF_MAP[sector]

    if not gemini_key or not newsapi_key:
        st.warning(
            "Please provide both **Gemini** and **NewsAPI** keys in the sidebar "
            "(or via `.streamlit/secrets.toml`) to run the analysis.",
            icon="\U0001F511",
        )

        st.markdown("---")
        st.markdown("#### How to set up API keys")
        st.markdown(
            """
            **Option 1 — Sidebar input:** Paste your keys directly in the sidebar fields above.

            **Option 2 — `st.secrets` (recommended for deployment):**

            Create a file at `.streamlit/secrets.toml` in this project directory:

            ```toml
            GEMINI_API_KEY = "your-gemini-api-key-here"
            NEWSAPI_KEY = "your-newsapi-key-here"
            ```

            Streamlit will automatically load these on startup.
            """
        )
        st.stop()

    configure_gemini(gemini_key)

    # --- Run analysis ---
    analyze_btn = st.sidebar.button(
        "\U0001F50D Run Analysis",
        use_container_width=True,
        type="primary",
    )

    ck = _cache_key(sector)
    cached = st.session_state.get(ck)

    if analyze_btn or cached:

        if analyze_btn or not cached:
            status_placeholder = st.empty()

            with st.spinner("Fetching price data..."):
                try:
                    price_df = fetch_stock_data(ticker)
                except ValueError as e:
                    st.error(f"Price data error: {e}")
                    st.stop()

            with st.spinner("Fetching news headlines..."):
                try:
                    raw_articles = fetch_news_headlines(sector, newsapi_key)
                    news_df = build_news_dataframe(raw_articles)
                except ConnectionError as e:
                    st.error(f"News data error: {e}")
                    st.stop()

            headlines = news_df["title"].tolist() if not news_df.empty else []

            def _status_update(msg: str) -> None:
                status_placeholder.info(msg, icon="\u23f3")

            with st.spinner(f"Analyzing sentiment (trying {gemini_model} first)..."):
                try:
                    sentiment_result = analyze_sentiment(
                        headlines, sector, model_name=gemini_model,
                        status_callback=_status_update,
                    )
                except (ValueError, ConnectionError) as e:
                    status_placeholder.empty()
                    st.error(f"Sentiment analysis error: {e}")
                    st.stop()

            used_model = sentiment_result.get("model_used", gemini_model)

            if used_model != gemini_model:
                status_placeholder.success(
                    f"**{gemini_model}** was unavailable — succeeded with **{used_model}**",
                    icon="\U0001F504",
                )
            else:
                status_placeholder.empty()

            sentiment_ts = build_sentiment_timeseries(news_df, sentiment_result)
            bubble_result = compute_bubble_index(price_df, sentiment_result["overall_sentiment"])

            st.session_state[ck] = {
                "price_df": price_df,
                "news_df": news_df,
                "headlines": headlines,
                "sentiment_result": sentiment_result,
                "sentiment_ts": sentiment_ts,
                "bubble_result": bubble_result,
                "used_model": used_model,
            }
            cached = st.session_state[ck]

        price_df = cached["price_df"]
        news_df = cached["news_df"]
        headlines = cached["headlines"]
        sentiment_ts = cached["sentiment_ts"]
        bubble_result = cached["bubble_result"]
        used_model = cached.get("used_model", gemini_model)

        # --- Render dashboard ---
        _render_metrics(price_df, bubble_result, ticker)

        st.markdown("---")

        chart_col, gauge_col = st.columns([3, 1])

        with chart_col:
            fig = create_price_sentiment_chart(price_df, sentiment_ts, ticker, sector)
            st.plotly_chart(fig, use_container_width=True)

        with gauge_col:
            gauge_fig = create_bubble_gauge(bubble_result.bubble_index)
            st.plotly_chart(gauge_fig, use_container_width=True)

            st.markdown("**Market Status**")
            _render_status_badge(bubble_result.status)

        st.markdown("---")

        st.markdown("#### Analysis Breakdown")
        breakdown_col1, breakdown_col2 = st.columns(2)

        with breakdown_col1:
            st.markdown("**Price Momentum**")
            st.markdown(
                f"- 30-day change: **{bubble_result.price_change_pct:+.2f}%**\n"
                f"- Normalized momentum score: **{bubble_result.price_momentum_score:+.4f}**"
            )

        with breakdown_col2:
            st.markdown("**Sentiment Analysis**")
            st.markdown(
                f"- Overall sentiment: **{bubble_result.sentiment_score:+.4f}**\n"
                f"- Headlines analyzed: **{len(headlines)}**\n"
                f"- Model: **{used_model}**"
            )

        st.info(bubble_result.explanation, icon="\U0001F4CA")

        st.markdown("---")

        _render_next_action(bubble_result)

        st.markdown("---")
        _render_news_table(news_df)

    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 4rem 2rem; color: #888;">
                <h3>Select an asset and click "Run Analysis" to begin</h3>
                <p>This tool compares stock price trends against news sentiment to identify
                potential bubbles or undervalued opportunities.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
