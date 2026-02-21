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

    return (
        f"From {start_date} to {end_date}, closing price moved from "
        f"{start_price:.2f} to {end_price:.2f} "
        f"({pct_change:+.2f}% change)."
    )


def summarize_sentiment(sentiment: dict) -> str:
    overall = sentiment.get("overall", "neutral")
    score = float(sentiment.get("score", 0.0))
    return f"Overall news sentiment is {overall} with an average score of {score:+.2f} on [-1, 1]."


@st.cache_data(show_spinner=False)
def cached_hist(symbol: str, start: date, end: date) -> pd.DataFrame:
    return fetch_historical_data(symbol, start, end)


@st.cache_data(show_spinner=False)
def cached_news(company_name: str, max_articles: int = 10):
    return fetch_news_for_stock(company_name, max_articles=max_articles)


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
    if "OPENAI_API_KEY" not in os.environ or not os.getenv("OPENAI_API_KEY"):
        st.warning(
            "OPENAI_API_KEY not configured. Set it in your environment or `.env` file "
            "to enable AI predictions and chatbot."
        )
        return

    with st.spinner("Generating AI outlook..."):
        hist_summary = summarize_history(symbol, hist)
        sentiment_summary = summarize_sentiment(sentiment)
        outlook = generate_price_outlook(
            symbol=symbol,
            company_name=company_name,
            hist_summary=hist_summary,
            sentiment_summary=sentiment_summary,
        )

    st.markdown(outlook)


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
        news_items, warning = cached_news(company_name)

    sentiment = analyze_news_sentiment(news_items) if news_items else {
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

