from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

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


def analyze_news_sentiment(
    news: List[NewsItem],
    company_name: str = "",
) -> Dict[str, Any]:
    """Use OpenAI to classify sentiment for each news item and aggregate."""
    if not news:
        return {
            "overall": "neutral",
            "score": 0.0,
            "details": [],
            "note": "No news available.",
        }

    client = _get_client()

    # Build text: headline first (for model to echo back), then snippet for context
    items_text = [
        (n.title, (n.description or "")[:200].strip()) for n in news
    ]
    chunk_strings = [
        f"{t}\n  Snippet: {d}" if d else t for t, d in items_text
    ]
    chunks = chunk_list(chunk_strings, max_len=8)

    sentiment_details: List[Dict[str, Any]] = []
    company_context = f" (all headlines are about this company/stock: {company_name})" if company_name else ""

    for chunk in chunks:
        prompt = (
            "You are a financial news sentiment analyst for Indian equities. "
            "Classify each headline as 'positive', 'negative', or 'neutral' "
            "for the underlying stock impact, and assign a sentiment score strictly between "
            "-1 (very negative) and 1 (very positive). Use decimals (e.g. 0.3, -0.5). "
            "Base your judgment on how the news would typically affect the stock price."
            f"{company_context}\n\n"
        )

        for i, h in enumerate(chunk, start=1):
            prompt += f"{i}. {h}\n"

        prompt += (
            "\nRespond in JSON with a single key 'sentiments': a list of objects, "
            "each with keys: 'headline' (the exact headline only, first line of each item above), "
            "'label' ('positive'|'negative'|'neutral'), 'score' (float in [-1, 1]). "
            "One item per numbered item, same order."
        )

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        try:
            raw = completion.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(f"Unexpected OpenAI response format: {exc}")

        try:
            import json

            parsed = json.loads(raw)
            sentiments = parsed.get("sentiments", [])
            for s in sentiments:
                score = float(s.get("score", 0.0))
                score = max(-1.0, min(1.0, score))  # clamp to [-1, 1]
                s["score"] = score
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
    sentiment_details: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a short-term outlook as structured JSON (direction, confidence, reasons, risks, disclaimer)."""
    client = _get_client()

    system_prompt = (
        "You are a financial analysis assistant focused on Indian equities. "
        "Base your answer ONLY on the provided price action and news sentiment. "
        "Use the headline-level sentiment below where provided to support your outlook. "
        "When price action and sentiment align, give a clear directional outlook (up or down) with medium or high confidence. "
        "Reserve sideways or low confidence for when signals conflict, volatility is high, or the picture is genuinely mixed. "
        "Do not speculate beyond the given data. "
        "Respond with a JSON object only, no other text. Use exactly these keys: "
        "'direction' (one of: up, down, sideways), "
        "'confidence' (one of: low, medium, high), "
        "'reasons' (array of 3-5 short strings citing specific numbers from the data), "
        "'risks' (array of 0-3 short strings), "
        "'disclaimer' (exactly: \"This is not financial advice. Do your own research.\")."
    )

    user_prompt = f"""
Stock symbol: {symbol}
Company name: {company_name}

Recent price action summary:
{hist_summary}

News sentiment summary:
{sentiment_summary}
"""
    if sentiment_details:
        user_prompt += f"""

Relevant headlines and scores:
{sentiment_details}
"""
    user_prompt += """

Provide a concise outlook for the next few weeks for a retail investor as a single JSON object with keys: direction, confidence, reasons, risks, disclaimer.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.25,
        response_format={"type": "json_object"},
    )

    try:
        raw = completion.choices[0].message.content
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenAI response format: {exc}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse outlook JSON: {exc}")

    direction = (parsed.get("direction") or "sideways").lower()
    if direction not in ("up", "down", "sideways"):
        direction = "sideways"
    confidence = (parsed.get("confidence") or "medium").lower()
    if confidence not in ("low", "medium", "high"):
        confidence = "medium"
    reasons = parsed.get("reasons")
    if not isinstance(reasons, list):
        reasons = []
    reasons = [str(r) for r in reasons if r]
    risks = parsed.get("risks")
    if not isinstance(risks, list):
        risks = []
    risks = [str(r) for r in risks if r]
    disclaimer = parsed.get("disclaimer") or "This is not financial advice. Do your own research."

    return {
        "direction": direction,
        "confidence": confidence,
        "reasons": reasons,
        "risks": risks,
        "disclaimer": disclaimer,
    }


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
        temperature=0.4,
    )

    try:
        return completion.choices[0].message.content or ""
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenAI response format: {exc}")

