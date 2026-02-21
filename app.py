import os
from datetime import date
from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.data_fetcher import (
    fetch_historical_data,
    fetch_news_for_stock,
    NewsItem,
)
from src.openai_service import (
    analyze_news_sentiment,
    chat_about_stock,
    generate_price_outlook,
)
from src.utils import POPULAR_STOCKS, get_date_range


load_dotenv()

st.set_page_config(
    page_title="India Stock AI Insights",
    layout="wide",
)


def summarize_history(symbol: str, hist: pd.DataFrame) -> str:
    hist = hist.dropna()
    if hist.empty:
        return f"No recent data available for {symbol}."

    start_date = hist.index.min().date()
    end_date = hist.index.max().date()
    start_price = float(hist["Close"].iloc[0])
    end_price = float(hist["Close"].iloc[-1])
    pct_change = (end_price - start_price) / start_price * 100 if start_price else 0.0

    period_high = float(hist["Close"].max())
    period_low = float(hist["Close"].min())
    n = len(hist)
    closes = hist["Close"]
    recent_5 = closes.iloc[-5:] if n >= 5 else closes
    recent_change = (
        (recent_5.iloc[-1] - recent_5.iloc[0]) / recent_5.iloc[0] * 100
        if len(recent_5) > 1 and recent_5.iloc[0]
        else 0.0
    )

    parts = [
        f"From {start_date} to {end_date}, closing price moved from "
        f"{start_price:.2f} to {end_price:.2f} ({pct_change:+.2f}% change).",
        f"Period high: {period_high:.2f}, low: {period_low:.2f}.",
    ]
    if n >= 5:
        parts.append(f"Last 5 sessions: {recent_change:+.2f}%.")

    # Short-term trend: last 5 sessions vs previous 5 sessions
    if n >= 10:
        prev_5 = closes.iloc[-10:-5]
        prev_change = (
            (prev_5.iloc[-1] - prev_5.iloc[0]) / prev_5.iloc[0] * 100
            if prev_5.iloc[0]
            else 0.0
        )
        if recent_change > 0.3 and recent_change > prev_change + 0.2:
            trend = "up"
        elif recent_change < -0.3 and recent_change < prev_change - 0.2:
            trend = "down"
        else:
            trend = "flat"
        parts.append(f"Short-term trend: {trend}.")
    elif n >= 5:
        trend = "up" if recent_change > 0.3 else ("down" if recent_change < -0.3 else "flat")
        parts.append(f"Short-term trend: {trend}.")

    # Volatility: daily return std
    daily_returns = closes.pct_change().dropna()
    if not daily_returns.empty:
        vol_pct = float(daily_returns.std() * 100)
        parts.append(f"Volatility (daily return std): {vol_pct:.2f}%.")

    # Volume: last 5 days avg vs period average (if Volume present)
    if "Volume" in hist.columns and hist["Volume"].notna().any():
        vol = hist["Volume"].replace(0, float("nan")).dropna()
        if len(vol) >= 5:
            period_avg = float(vol.mean())
            last_5_avg = float(vol.iloc[-5:].mean())
            if period_avg and period_avg > 0:
                ratio = last_5_avg / period_avg
                if ratio > 1.15:
                    vol_desc = "above"
                elif ratio < 0.85:
                    vol_desc = "below"
                else:
                    vol_desc = "similar"
                parts.append(f"Recent volume vs period average: {vol_desc}.")

    return " ".join(parts)


def summarize_sentiment(sentiment: dict) -> str:
    overall = sentiment.get("overall", "neutral")
    score = float(sentiment.get("score", 0.0))
    return f"Overall news sentiment is {overall} with an average score of {score:+.2f} on [-1, 1]."


def build_sentiment_detail(sentiment: dict, top_n: int = 3) -> str:
    """Build a short string of top positive and negative headlines with scores for the outlook prompt."""
    details = sentiment.get("details") or []
    if not details:
        return ""

    with_score = [(d, float(d.get("score", 0.0))) for d in details if isinstance(d, dict)]
    positive = sorted([x for x in with_score if x[1] > 0], key=lambda x: -x[1])[:top_n]
    negative = sorted([x for x in with_score if x[1] < 0], key=lambda x: x[1])[:top_n]

    lines = []
    if positive:
        lines.append("Positive headlines:")
        for d, s in positive:
            headline = (d.get("headline") or str(d))[:120]
            lines.append(f"  • {headline} (score: {s:+.2f})")
    if negative:
        lines.append("Negative headlines:")
        for d, s in negative:
            headline = (d.get("headline") or str(d))[:120]
            lines.append(f"  • {headline} (score: {s:+.2f})")
    return "\n".join(lines) if lines else ""


@st.cache_data(show_spinner="Loading historical data...")
def cached_hist(symbol: str, start: date, end: date) -> pd.DataFrame:
    return fetch_historical_data(symbol, start, end)


@st.cache_data(show_spinner="Fetching recent news...")
def cached_news(company_name: str, max_articles: int = 10, symbol: Optional[str] = None):
    return fetch_news_for_stock(company_name, max_articles=max_articles, symbol=symbol)


def render_sidebar():
    st.sidebar.title("India Stock AI Tool")
    company_name = st.sidebar.selectbox(
        "Select stock (NSE):",
        list(POPULAR_STOCKS.keys()),
    )
    symbol = POPULAR_STOCKS[company_name]

    preset = st.sidebar.selectbox(
        "Date range:",
        ["3M", "6M", "1Y", "2Y"],
        index=1,
    )

    start_date, end_date = get_date_range(preset)
    st.sidebar.markdown(
        f"**Date window:** {start_date.isoformat()} → {end_date.isoformat()}"
    )

    return company_name, symbol, start_date, end_date


def render_price_chart(hist: pd.DataFrame, symbol: str):
    st.subheader(f"Historical Price – {symbol}")
    hist = hist.reset_index()
    fig = px.line(hist, x="Date", y="Close", title="Closing Price")
    st.plotly_chart(fig, use_container_width=True)


def render_news(news_items: List[NewsItem], warning: Optional[str], sentiment: dict):
    st.subheader("Recent News & Sentiment")
    if warning:
        st.info(warning)

    if not news_items:
        st.write("No news available.")
        return

    st.markdown(f"**{summarize_sentiment(sentiment)}**")

    for item in news_items:
        st.markdown(f"### [{item.title}]({item.url})")
        if item.description:
            st.write(item.description)
        meta = f"{item.source} — {item.published_at}"
        st.caption(meta)
        st.markdown("---")


def render_prediction_panel(
    symbol: str,
    company_name: str,
    hist: pd.DataFrame,
    sentiment: dict,
):
    st.subheader("AI Outlook (Experimental)")
    st.caption("For education only; not financial advice.")
    if "OPENAI_API_KEY" not in os.environ or not os.getenv("OPENAI_API_KEY"):
        st.warning(
            "OPENAI_API_KEY not configured. Set it in your environment or `.env` file "
            "to enable AI predictions and chatbot."
        )
        return

    with st.spinner("Generating AI outlook..."):
        hist_summary = summarize_history(symbol, hist)
        sentiment_summary = summarize_sentiment(sentiment)
        sentiment_details = build_sentiment_detail(sentiment)
        outlook = generate_price_outlook(
            symbol=symbol,
            company_name=company_name,
            hist_summary=hist_summary,
            sentiment_summary=sentiment_summary,
            sentiment_details=sentiment_details or None,
        )

    direction = outlook.get("direction", "sideways")
    confidence = outlook.get("confidence", "medium")
    reasons = outlook.get("reasons", [])
    risks = outlook.get("risks", [])
    disclaimer = outlook.get("disclaimer", "This is not financial advice. Do your own research.")

    dir_label = direction.capitalize()
    conf_label = confidence.capitalize()
    st.markdown(f"**Direction:** {dir_label}   \n**Confidence:** {conf_label}")

    if reasons:
        st.markdown("**Reasons**")
        for r in reasons:
            st.markdown(f"- {r}")
    if risks:
        st.markdown("**Risks**")
        for r in risks:
            st.markdown(f"- {r}")
    st.info(disclaimer)


def render_chatbot_tab(
    symbol: str,
    company_name: str,
    hist: pd.DataFrame,
    sentiment: dict,
):
    st.subheader("Chat about this stock")

    if "OPENAI_API_KEY" not in os.environ or not os.getenv("OPENAI_API_KEY"):
        st.warning(
            "OPENAI_API_KEY not configured. Set it in your environment or `.env` file "
            "to enable the chatbot."
        )
        return

    user_q = st.text_input("Ask a question (e.g., risks, recent moves, etc.)")
    if not user_q:
        st.info("Type a question above to start the conversation.")
        return

    if st.button("Ask", type="primary"):
        with st.spinner("Thinking..."):
            hist_summary = summarize_history(symbol, hist)
            sentiment_summary = summarize_sentiment(sentiment)
            answer = chat_about_stock(
                symbol=symbol,
                company_name=company_name,
                hist_summary=hist_summary,
                sentiment_summary=sentiment_summary,
                user_question=user_q,
            )
        st.markdown(answer)


def main():
    st.title("India Stock Market AI Insights (Demo)")
    st.caption(
        "Fetch Indian stock data, analyze news sentiment, and get AI-generated insights. "
        "This is for educational purposes only, not financial advice."
    )

    company_name, symbol, start_date, end_date = render_sidebar()

    try:
        with st.spinner("Loading historical data..."):
            hist = cached_hist(symbol, start_date, end_date)
    except Exception as exc:
        st.error(f"Failed to load historical data: {exc}")
        return

    cols = st.columns([2, 1])
    with cols[0]:
        render_price_chart(hist, symbol)
    with cols[1]:
        st.metric(
            "Last Close",
            f"{hist['Close'].iloc[-1]:.2f}",
        )
        st.metric(
            "Daily Change",
            f"{(hist['Close'].iloc[-1] - hist['Close'].iloc[-2]):+.2f}",
        )

    st.markdown("---")

    news_items: List[NewsItem]
    warning: Optional[str]
    with st.spinner("Fetching recent news..."):
        news_items, warning = cached_news(company_name, symbol=symbol)

    if news_items:
        with st.spinner("Analyzing news sentiment..."):
            sentiment = analyze_news_sentiment(news_items, company_name=company_name)
    else:
        sentiment = {
            "overall": "neutral",
            "score": 0.0,
            "details": [],
        }

    tabs = st.tabs(["News & Sentiment", "AI Outlook", "Chatbot"])

    with tabs[0]:
        render_news(news_items, warning, sentiment)
    with tabs[1]:
        render_prediction_panel(symbol, company_name, hist, sentiment)
    with tabs[2]:
        render_chatbot_tab(symbol, company_name, hist, sentiment)


if __name__ == "__main__":
    main()

