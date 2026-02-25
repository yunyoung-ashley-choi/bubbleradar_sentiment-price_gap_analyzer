"""
Bubble Radar v2 — AI Stock Intelligence Dashboard
Main Streamlit application with priority asset cards and 60-day ML forecasting.

Entry point:  streamlit run main.py
"""

import logging

import pandas as pd
import streamlit as st

from bubble_index import MarketStatus, compute_bubble_index
from charts import create_bubble_gauge, create_forecast_chart, create_price_sentiment_chart
from config import (
    PREDICTION,
    PRIORITY_ASSETS,
    SECTOR_ETF_MAP,
    SECTOR_SEARCH_KEYWORDS,
    THRESHOLDS,
    UI,
)
from data_fetcher import fetch_stock_data
from engine import build_features, classify_trend, compute_technical_indicators, forecast_price
from sentiment import analyze_asset_sentiment

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Page configuration
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title=f"{UI.APP_TITLE}: {UI.APP_SUBTITLE}",
    page_icon=UI.PAGE_ICON,
    layout=UI.LAYOUT,
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════════════════════

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ---- Header ---- */
        .main-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(99, 110, 250, 0.3);
        }
        .main-header h1 { color: #e0e0ff; margin: 0; font-size: 2rem; }
        .main-header p  { color: #a0a0c0; margin: 0.3rem 0 0 0; font-size: 1rem; }

        /* ---- Metric cards ---- */
        div[data-testid="stMetric"] {
            background: rgba(30, 30, 60, 0.5);
            border: 1px solid rgba(99, 110, 250, 0.2);
            border-radius: 10px;
            padding: 1rem;
        }

        /* ---- Priority asset card ---- */
        .asset-card {
            background: linear-gradient(135deg, rgba(30,30,60,0.7), rgba(20,20,50,0.9));
            border: 1px solid rgba(99, 110, 250, 0.25);
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 0.8rem;
            min-height: 195px;
            transition: border-color 0.2s;
        }
        .asset-card:hover { border-color: rgba(99, 110, 250, 0.6); }
        .asset-card h4 { color: #e0e0ff; margin: 0 0 0.5rem 0; font-size: 1rem; }
        .asset-card .price { color: #ffffff; font-size: 1.4rem; font-weight: 700; }
        .change-pos { color: #00CC96; font-size: 0.9rem; }
        .change-neg { color: #EF553B; font-size: 0.9rem; }
        .trend-bullish { color: #00CC96; font-weight: 700; }
        .trend-bearish { color: #EF553B; font-weight: 700; }
        .trend-neutral { color: #636EFA; font-weight: 700; }

        /* ---- Status badges ---- */
        .status-overheated {
            background: linear-gradient(135deg, #ff4444, #cc0000);
            color: white; padding: 0.4rem 1.2rem; border-radius: 20px;
            font-weight: 700; font-size: 1.1rem; display: inline-block;
        }
        .status-stable {
            background: linear-gradient(135deg, #636EFA, #4040cc);
            color: white; padding: 0.4rem 1.2rem; border-radius: 20px;
            font-weight: 700; font-size: 1.1rem; display: inline-block;
        }
        .status-undervalued {
            background: linear-gradient(135deg, #00CC96, #009966);
            color: white; padding: 0.4rem 1.2rem; border-radius: 20px;
            font-weight: 700; font-size: 1.1rem; display: inline-block;
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #f9fafb !important; font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] label {
            color: #d1d5db !important; font-weight: 500 !important;
            font-size: 0.95rem !important;
        }
        section[data-testid="stSidebar"] p { color: #d1d5db !important; }
        section[data-testid="stSidebar"] .stCaption p {
            color: #9ca3af !important; font-size: 0.85rem !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(156, 163, 175, 0.3) !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] [data-baseweb="select"] span,
        section[data-testid="stSidebar"] [data-baseweb="select"] div[aria-selected],
        section[data-testid="stSidebar"] [data-baseweb="select"] > div > div {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] svg {
            fill: #6b7280 !important;
        }
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


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_price(price: float, currency: str) -> str:
    """Format a price with the correct currency symbol and precision."""
    if currency == "₩":
        return f"₩{price:,.0f}"
    return f"${price:,.2f}"


def _quick_trend(price_change_30d: float, sentiment: float) -> str:
    """
    Lightweight trend estimate used for the priority cards so we don't
    have to run Prophet for every asset on page load.
    """
    combined = price_change_30d * 0.6 + sentiment * 40 * 0.4
    if combined > THRESHOLDS.BULLISH_PCT:
        return "Bullish"
    if combined < THRESHOLDS.BEARISH_PCT:
        return "Bearish"
    return "Neutral"


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> str:
    """Build sidebar; return the asset name selected for deep analysis."""
    with st.sidebar:
        st.markdown("### Deep Analysis")

        all_assets = list(PRIORITY_ASSETS.keys()) + [
            k for k in SECTOR_ETF_MAP if k not in PRIORITY_ASSETS
        ]
        selected = st.selectbox(
            "Select asset for 60-day forecast",
            options=all_assets,
            index=0,
        )

        ticker = SECTOR_ETF_MAP.get(selected, "")
        if ticker:
            st.info(f"Tracking: **{ticker}**", icon="📈")

        st.markdown("---")
        st.markdown("### Forecast Settings")
        st.caption(f"Horizon: **{PREDICTION.FORECAST_DAYS} days**")
        st.caption(f"Training window: **{PREDICTION.TRAINING_LOOKBACK_DAYS} days**")
        st.caption(f"RSI period: **{PREDICTION.RSI_PERIOD}**")
        st.caption(f"Headlines cap: **{THRESHOLDS.MAX_HEADLINES}**")

        st.markdown("---")
        if st.button("🔄 Refresh All Data", use_container_width=True):
            keys_to_clear = [
                k for k in st.session_state
                if k.startswith(("priority_", "analysis_"))
            ]
            for k in keys_to_clear:
                del st.session_state[k]
            st.rerun()

        st.markdown("---")
        st.markdown(
            "<p style='color:#9ca3af;font-size:0.8rem;text-align:center;'>"
            "Powered by Prophet · FinBERT · yfinance · Google News"
            "</p>",
            unsafe_allow_html=True,
        )

    return selected


# ═══════════════════════════════════════════════════════════════════════════
# Priority Asset Cards (top section)
# ═══════════════════════════════════════════════════════════════════════════

def _load_priority_data() -> dict:
    """Fetch price + sentiment for every priority asset; cache in session."""
    if "priority_data" in st.session_state:
        return st.session_state["priority_data"]

    data: dict = {}
    total = len(PRIORITY_ASSETS)
    progress = st.progress(0, text="Loading priority assets…")

    for idx, (name, info) in enumerate(PRIORITY_ASSETS.items()):
        progress.progress(
            (idx + 1) / total,
            text=f"Loading {name} ({idx+1}/{total})…",
        )
        ticker = info["ticker"]
        currency = info["currency"]

        try:
            price_df = fetch_stock_data(ticker, days=THRESHOLDS.LOOKBACK_DAYS)
            current = price_df["Close"].iloc[-1]
            prev = price_df["Close"].iloc[-2] if len(price_df) > 1 else current
            daily_pct = ((current - prev) / prev) * 100

            try:
                sent = analyze_asset_sentiment(name, days=14, max_headlines=15)
                score = sent["overall_sentiment"]
            except Exception as exc:
                logger.warning("Sentiment failed for %s: %s", name, exc)
                score = 0.0
                sent = None

            pct30 = (
                ((current - price_df["Close"].iloc[0])
                 / price_df["Close"].iloc[0]) * 100
            )
            trend = _quick_trend(pct30, score)

            data[name] = {
                "price_df": price_df,
                "current_price": current,
                "daily_change_pct": daily_pct,
                "sentiment_score": score,
                "sentiment_data": sent,
                "trend": trend,
                "currency": currency,
                "ticker": ticker,
                "error": None,
            }
        except Exception as exc:
            logger.error("Failed to load %s: %s", name, exc)
            data[name] = {
                "error": str(exc),
                "ticker": ticker,
                "currency": currency,
            }

    progress.empty()
    st.session_state["priority_data"] = data
    return data


def _render_priority_cards(data: dict) -> None:
    """Render 6 priority-asset metric cards (3 per row)."""
    st.markdown("### Priority Assets")

    names = list(data.keys())
    for row_start in range(0, len(names), 3):
        row_names = names[row_start : row_start + 3]
        cols = st.columns(3)
        for col, name in zip(cols, row_names):
            with col:
                info = data[name]
                if info.get("error"):
                    st.error(f"**{name}**\n\n{info['error']}")
                    continue

                price_str = _fmt_price(info["current_price"], info["currency"])
                change = info["daily_change_pct"]
                chg_cls = "change-pos" if change >= 0 else "change-neg"

                sentiment = info["sentiment_score"]
                s_color = "#00CC96" if sentiment >= 0 else "#EF553B"

                trend = info["trend"]
                if trend == "Bullish":
                    trend_html = '<span class="trend-bullish">📈 Bullish</span>'
                elif trend == "Bearish":
                    trend_html = '<span class="trend-bearish">📉 Bearish</span>'
                else:
                    trend_html = '<span class="trend-neutral">➡️ Neutral</span>'

                st.markdown(
                    f"""
                    <div class="asset-card">
                        <h4>{name} <span style="color:#888;font-weight:400;">
                            ({info['ticker']})</span></h4>
                        <div class="price">{price_str}</div>
                        <div class="{chg_cls}">Daily: {change:+.2f}%</div>
                        <div style="color:{s_color};font-size:0.85rem;margin-top:0.3rem;">
                            Sentiment: {sentiment:+.3f}</div>
                        <div style="margin-top:0.4rem;">60-Day: {trend_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════
# Detailed 60-Day Analysis (on-demand per asset)
# ═══════════════════════════════════════════════════════════════════════════

def _run_detailed_analysis(name: str) -> None:
    """Show the button, run Prophet if clicked, then render results."""
    cache_key = f"analysis_{name}"
    cached = st.session_state.get(cache_key)

    analyze_btn = st.button(
        "🔍 Run 60-Day Prediction",
        use_container_width=True,
        type="primary",
    )

    if not analyze_btn and not cached:
        st.info(
            "Click **Run 60-Day Prediction** to train a Prophet model "
            "and generate a 60-day forecast for this asset.",
            icon="🔮",
        )
        return

    if analyze_btn or not cached:
        _compute_analysis(name, cache_key)
        cached = st.session_state.get(cache_key)

    if cached is None:
        return
    if cached.get("error"):
        st.error(f"Analysis failed: {cached['error']}")
        return

    _display_results(name, cached)


def _compute_analysis(name: str, cache_key: str) -> None:
    """Execute the full pipeline: prices → sentiment → features → Prophet."""
    ticker = SECTOR_ETF_MAP.get(
        name, PRIORITY_ASSETS.get(name, {}).get("ticker", "")
    )
    if not ticker:
        st.session_state[cache_key] = {"error": f"Unknown asset: {name}"}
        return

    currency = PRIORITY_ASSETS.get(name, {}).get("currency", "$")

    # 1 — Historical prices (1 year)
    with st.spinner("Fetching 1-year price history…"):
        try:
            price_df = fetch_stock_data(
                ticker, days=PREDICTION.TRAINING_LOOKBACK_DAYS,
            )
        except ValueError as exc:
            st.session_state[cache_key] = {"error": str(exc)}
            return

    # 2 — FinBERT sentiment
    with st.spinner("Running FinBERT sentiment analysis…"):
        try:
            sentiment_data = analyze_asset_sentiment(
                name, days=THRESHOLDS.LOOKBACK_DAYS,
            )
        except Exception as exc:
            logger.warning("Sentiment failed for %s: %s", name, exc)
            sentiment_data = {
                "headlines": [],
                "scores": [],
                "overall_sentiment": 0.0,
                "news_df": pd.DataFrame(),
                "sentiment_ts": pd.DataFrame(
                    columns=["date", "avg_sentiment"]
                ),
            }

    # 3 — Technical indicators + features
    with st.spinner("Computing technical indicators…"):
        tech = compute_technical_indicators(price_df)
        features_df = build_features(
            price_df, sentiment_data.get("sentiment_ts"),
        )

    # 4 — Prophet forecast
    with st.spinner("Training Prophet model (this may take 15-30 s)…"):
        try:
            forecast_df = forecast_price(features_df)
            trend = classify_trend(forecast_df)
        except Exception as exc:
            logger.error("Prophet error for %s: %s", name, exc)
            st.session_state[cache_key] = {
                "error": f"Prediction failed: {exc}",
            }
            return

    # 5 — Bubble index
    bubble = compute_bubble_index(
        price_df, sentiment_data["overall_sentiment"],
    )

    st.session_state[cache_key] = {
        "price_df": price_df,
        "sentiment_data": sentiment_data,
        "tech": tech,
        "features_df": features_df,
        "forecast_df": forecast_df,
        "trend": trend,
        "bubble": bubble,
        "ticker": ticker,
        "currency": currency,
        "error": None,
    }


def _display_results(name: str, data: dict) -> None:
    """Render charts, metrics, and tables for a completed analysis."""
    ticker = data["ticker"]
    currency = data["currency"]
    price_df = data["price_df"]
    forecast_df = data["forecast_df"]
    sentiment_data = data["sentiment_data"]
    bubble = data["bubble"]
    tech = data["tech"]
    trend = data["trend"]

    current_price = price_df["Close"].iloc[-1]

    # ── Top metric cards ────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            f"{ticker} Price",
            _fmt_price(current_price, currency),
            f"{bubble.price_change_pct:+.2f}%",
        )
    with c2:
        s = sentiment_data["overall_sentiment"]
        st.metric(
            "Sentiment",
            f"{s:+.3f}",
            "Positive" if s > 0 else ("Negative" if s < 0 else "Neutral"),
            delta_color="normal" if s >= 0 else "inverse",
        )
    with c3:
        st.metric("Bubble Index", f"{bubble.bubble_index:+.4f}")
    with c4:
        arrow = "📈" if trend == "Bullish" else ("📉" if trend == "Bearish" else "➡️")
        st.metric("60-Day Trend", trend, arrow)

    st.markdown("---")

    # ── Forecast chart ──────────────────────────────────────────────────
    st.markdown("#### 60-Day Price Forecast")
    st.plotly_chart(
        create_forecast_chart(forecast_df, ticker, name, currency),
        use_container_width=True,
    )

    # ── Price vs Sentiment + Bubble gauge ───────────────────────────────
    chart_col, gauge_col = st.columns([3, 1])
    with chart_col:
        recent = price_df.tail(THRESHOLDS.LOOKBACK_DAYS)
        st.plotly_chart(
            create_price_sentiment_chart(
                recent,
                sentiment_data.get(
                    "sentiment_ts",
                    pd.DataFrame(columns=["date", "avg_sentiment"]),
                ),
                ticker,
                name,
            ),
            use_container_width=True,
        )
    with gauge_col:
        st.plotly_chart(
            create_bubble_gauge(bubble.bubble_index),
            use_container_width=True,
        )
        badge_map = {
            MarketStatus.OVERHEATED: ("status-overheated", "🔥"),
            MarketStatus.STABLE: ("status-stable", "⚖️"),
            MarketStatus.UNDERVALUED: ("status-undervalued", "💡"),
        }
        css_cls, icon = badge_map[bubble.status]
        st.markdown(
            f'<span class="{css_cls}">{icon} {bubble.status.value}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Technical indicators ────────────────────────────────────────────
    st.markdown("#### Technical Indicators (Current)")
    t1, t2, t3, t4 = st.columns(4)

    rsi_now = tech["rsi"].iloc[-1]
    rsi_lbl = (
        "Overbought" if rsi_now > 70
        else ("Oversold" if rsi_now < 30 else "Normal")
    )
    with t1:
        st.metric("RSI (14)", f"{rsi_now:.1f}", rsi_lbl)
    with t2:
        st.metric("MACD", f"{tech['macd_line'].iloc[-1]:.4f}")
    with t3:
        st.metric(
            f"MA {PREDICTION.MA_SHORT}",
            _fmt_price(tech["ma_short"].iloc[-1], currency),
        )
    with t4:
        st.metric(
            f"MA {PREDICTION.MA_LONG}",
            _fmt_price(tech["ma_long"].iloc[-1], currency),
        )

    st.info(bubble.explanation, icon="📊")

    # ── Breakdown ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Analysis Breakdown")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Price Momentum**")
        st.markdown(
            f"- 30-day change: **{bubble.price_change_pct:+.2f}%**\n"
            f"- Momentum score: **{bubble.price_momentum_score:+.4f}**\n"
            f"- 60-day trend: **{trend}**"
        )
    with b2:
        st.markdown("**Sentiment (FinBERT)**")
        st.markdown(
            f"- Overall sentiment: **{sentiment_data['overall_sentiment']:+.4f}**\n"
            f"- Headlines analysed: **{len(sentiment_data['headlines'])}**\n"
            f"- Model: **ProsusAI/finbert**"
        )

    # ── Next-action panel (Korean + English) ────────────────────────────
    st.markdown("---")
    _render_next_action(bubble)

    # ── News table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Recent News Headlines")
    news_df = sentiment_data.get("news_df", pd.DataFrame())
    if news_df.empty:
        st.info("No news articles found for this asset.")
    else:
        disp = news_df[["date", "title", "source"]].head(20).copy()
        disp.columns = ["Date", "Headline", "Source"]
        scores = sentiment_data.get("scores", [])
        if scores:
            disp["Sentiment"] = (scores + [None] * len(disp))[: len(disp)]
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# Next-action recommendation panel
# ═══════════════════════════════════════════════════════════════════════════

_NEXT_ACTION = {
    MarketStatus.OVERHEATED: {
        "icon": "🔥",
        "label_ko": "과열 (Overheated)",
        "color": "#ff4444",
        "desc_ko": (
            "주가 상승 속도가 뉴스 심리보다 훨씬 빠릅니다. "
            "시장이 실제 펀더멘털보다 과도한 낙관을 반영하고 있을 가능성이 높습니다."
        ),
        "actions": [
            "신규 매수 자제 — 현재 가격이 고평가 구간일 수 있습니다",
            "보유 중이라면 일부 이익 실현(차익 매도) 검토",
            "손절매 라인 설정 — 급격한 조정에 대비하세요",
            "거래량 및 변동성 지표를 주시하세요",
            "FOMO(놓칠까 봐 두려운 심리)에 휩쓸리지 마세요",
        ],
    },
    MarketStatus.STABLE: {
        "icon": "⚖️",
        "label_ko": "안정 (Stable)",
        "color": "#636EFA",
        "desc_ko": (
            "주가 흐름과 뉴스 심리가 균형 잡혀 있습니다. "
            "시장이 현재 뉴스를 합리적으로 반영하고 있는 상태입니다."
        ),
        "actions": [
            "현재 포지션 유지 — 급격한 조정 가능성 낮음",
            "정기적으로 뉴스와 가격을 모니터링하세요",
            "분할 매수(DCA) 전략 고려 가능",
            "갑작스러운 뉴스 변화에 대비한 알림 설정 권장",
            "장기 투자 관점에서 안정적인 구간입니다",
        ],
    },
    MarketStatus.UNDERVALUED: {
        "icon": "💡",
        "label_ko": "저평가 (Undervalued)",
        "color": "#00CC96",
        "desc_ko": (
            "뉴스 심리가 긍정적인데 비해 주가가 아직 충분히 반영하지 못하고 있습니다. "
            "시장이 좋은 뉴스를 아직 '소화'하지 못한 상태 — 매수 기회일 수 있습니다."
        ),
        "actions": [
            "매수 기회 탐색 — 현재 가격이 저평가 구간일 수 있습니다",
            "분할 매수(DCA)로 리스크를 분산하세요",
            "긍정적 심리의 원인(뉴스)을 직접 확인하세요",
            "단기 급등보다는 중장기 관점으로 접근 권장",
            "해당 섹터의 펀더멘털(실적, 성장성)도 함께 분석하세요",
        ],
    },
}


def _render_next_action(bubble) -> None:
    info = _NEXT_ACTION[bubble.status]
    checks = "".join(
        f'<div style="padding:0.35rem 0;display:flex;align-items:flex-start;">'
        f'<span style="color:{info["color"]};margin-right:0.5rem;font-weight:bold;">✓</span>'
        f"<span>{a}</span></div>"
        for a in info["actions"]
    )
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(30,30,60,0.7), rgba(20,20,50,0.9));
            border-left: 5px solid {info['color']};
            border-radius: 10px;
            padding: 1.5rem 2rem;
            margin: 1rem 0;
        ">
            <h4 style="color:#f0f0ff;margin-top:0;">
                {info['icon']} 다음 행동 가이드 (Next Action)
            </h4>
            <p style="color:{info['color']};font-weight:700;font-size:1.1rem;margin-bottom:0.5rem;">
                상태: {info['label_ko']}
            </p>
            <p style="color:#c0c0d0;font-size:0.95rem;margin-bottom:1rem;">
                {info['desc_ko']}
            </p>
            <div style="color:#e0e0f0;">{checks}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "※ 이 분석은 뉴스 심리와 가격 데이터를 기반으로 한 참고 자료이며, "
        "투자 조언이 아닙니다. 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    _inject_css()

    # Header
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{UI.PAGE_ICON} {UI.APP_TITLE}</h1>
            <p>{UI.APP_SUBTITLE} — 60-Day ML Forecasting · Sentiment-Price Gap Analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = _render_sidebar()

    # Priority asset cards
    priority_data = _load_priority_data()
    _render_priority_cards(priority_data)

    st.markdown("---")

    # Detailed analysis for the selected asset
    st.markdown(f"### Deep Analysis: {selected}")
    _run_detailed_analysis(selected)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 2rem 0;">
            <p style="color:#6b7280;font-size:0.85rem;margin:0;">
                &copy; 2026 Bubble Radar v2.
                해당 웹사이트는 <strong>최윤영</strong>과 <strong>AI</strong>의 합작품입니다.
            </p>
            <p style="color:#4b5563;font-size:0.75rem;margin:0.3rem 0 0 0;">
                Powered by Prophet · FinBERT · yfinance · Google News
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
