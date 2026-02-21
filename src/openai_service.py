from __future__ import annotations

import os
from typing import List, Dict, Any

from openai import OpenAI

from .data_fetcher import NewsItem
from .utils import chunk_list


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Please configure it in your environment."
        )
    return OpenAI(api_key=api_key)


def analyze_news_sentiment(news: List[NewsItem]) -> Dict[str, Any]:
    """Use OpenAI to classify sentiment for each news item and aggregate."""
    if not news:
        return {
            "overall": "neutral",
            "score": 0.0,
            "details": [],
            "note": "No news available.",
        }

    client = _get_client()

    headlines = [n.title for n in news]
    chunks = chunk_list(headlines, max_len=10)

    sentiment_details: List[Dict[str, Any]] = []

    for chunk in chunks:
        prompt = (
            "You are a financial news sentiment analyst. "
            "Classify each headline as 'positive', 'negative', or 'neutral' "
            "for the underlying stock, and assign a sentiment score between "
            "-1 (very negative) and 1 (very positive).\n\n"
        )

        for i, h in enumerate(chunk, start=1):
            prompt += f"{i}. {h}\n"

        prompt += (
            "\nRespond in JSON with a list named 'sentiments', where each item has:\n"
            "{'headline': str, 'label': 'positive'|'negative'|'neutral', 'score': float}."
        )

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        try:
            raw = completion.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(f"Unexpected OpenAI response format: {exc}")

        try:
            import json

            parsed = json.loads(raw)
            sentiments = parsed.get("sentiments", [])
            sentiment_details.extend(sentiments)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse sentiment JSON: {exc}")

    if not sentiment_details:
        return {
            "overall": "neutral",
            "score": 0.0,
            "details": [],
            "note": "No sentiment details parsed.",
        }

    avg_score = sum(float(s.get("score", 0.0)) for s in sentiment_details) / len(
        sentiment_details
    )
    if avg_score > 0.15:
        overall = "positive"
    elif avg_score < -0.15:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "overall": overall,
        "score": avg_score,
        "details": sentiment_details,
    }


def generate_price_outlook(
    symbol: str,
    company_name: str,
    hist_summary: str,
    sentiment_summary: str,
) -> str:
    """Generate a short-term outlook and explanation using OpenAI."""
    client = _get_client()

    system_prompt = (
        "You are a financial analysis assistant focused on Indian equities. "
        "Given recent price action and news sentiment, provide a short-term "
        "OUTLOOK for the stock (up / down / sideways) with a confidence level, "
        "plus 3–5 bullet points on reasons and risks.\n\n"
        "IMPORTANT: You must always include the disclaimer:\n"
        "\"This is not financial advice. Do your own research.\""
    )

    user_prompt = f"""
Stock symbol: {symbol}
Company name: {company_name}

Recent price action summary:
{hist_summary}

News sentiment summary:
{sentiment_summary}

Provide a concise outlook for the next few weeks for a retail investor.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    try:
        return completion.choices[0].message.content
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenAI response format: {exc}")


def chat_about_stock(
    symbol: str,
    company_name: str,
    hist_summary: str,
    sentiment_summary: str,
    user_question: str,
) -> str:
    """Stock-specific chatbot using OpenAI."""
    client = _get_client()

    system_prompt = (
        "You are a helpful assistant for Indian stock market investors. "
        "Answer questions about a specific stock using ONLY the provided "
        "context: recent price action and news sentiment. If you don't know "
        "something, say you don't know and suggest how the user could research it. "
        "Do not provide financial advice; instead, provide educational insights."
    )

    context = f"""
Stock symbol: {symbol}
Company name: {company_name}

Recent price action summary:
{hist_summary}

News sentiment summary:
{sentiment_summary}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
            {"role": "user", "content": user_question},
        ],
    )

    try:
        return completion.choices[0].message.content
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenAI response format: {exc}")

