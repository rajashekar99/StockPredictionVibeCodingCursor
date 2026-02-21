from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
import feedparser


logger = logging.getLogger(__name__)

# RSS feeds for Indian financial news (no API key required)
RSS_FEED_URLS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://www.livemint.com/rss/markets",
]


@dataclass
class NewsItem:
    title: str
    description: str
    url: str
    published_at: str
    source: str


def _extract_keywords(company_name: str, symbol: Optional[str] = None) -> List[str]:
    """Extract searchable keywords from company display name and optional symbol.

    E.g. 'Reliance Industries (NSE)' -> ['Reliance', 'Industries'];
    with symbol 'RELIANCE.NS' adds 'RELIANCE', 'RIL' (common ticker aliases).
    """
    # Strip parenthetical suffix like (NSE)
    cleaned = re.sub(r"\s*\([^)]+\)\s*$", "", company_name).strip()
    tokens = re.findall(r"[a-zA-Z0-9]{2,}", cleaned)
    keywords = [t for t in tokens if t.lower() not in ("ns", "bse", "nse")]

    if symbol:
        # Add ticker base (e.g. RELIANCE from RELIANCE.NS, TCS from TCS.NS)
        base = re.sub(r"\.(NS|BO|NSE|BSE)$", "", symbol, flags=re.IGNORECASE).strip()
        if base and base not in keywords:
            keywords.append(base)
        # Common ticker aliases for better headline matching
        ticker_aliases: Dict[str, List[str]] = {
            "RELIANCE": ["RIL", "Reliance"],
            "HDFCBANK": ["HDFC"],
            "ICICIBANK": ["ICICI"],
            "AXISBANK": ["Axis"],
            "KOTAKBANK": ["Kotak"],
            "BHARTIARTL": ["Airtel", "Bharti"],
        }
        base_upper = base.upper() if base else ""
        for alias in ticker_aliases.get(base_upper, []):
            if alias not in keywords:
                keywords.append(alias)
    return keywords


def _fetch_rss_feed(url: str) -> List[dict]:
    """Fetch and parse an RSS feed, return list of {title, summary, link, published, source}."""
    raw_items: List[dict] = []
    try:
        parsed = feedparser.parse(url)
        feed_title = parsed.feed.get("title", "") if hasattr(parsed, "feed") else ""
        for entry in getattr(parsed, "entries", []):
            title = entry.get("title") or ""
            summary = entry.get("summary") or entry.get("description") or ""
            link = entry.get("link") or ""
            published = ""
            if hasattr(entry, "published"):
                published = entry.get("published", "")
            elif hasattr(entry, "updated"):
                published = entry.get("updated", "")
            raw_items.append({
                "title": title,
                "summary": summary,
                "link": link,
                "published": published,
                "source": feed_title or url,
            })
    except Exception as exc:
        logger.warning("Failed to fetch RSS feed %s: %s", url, exc)
    return raw_items


def _matches_keywords(text: str, keywords: List[str]) -> bool:
    """Return True if any keyword appears (case-insensitive) in text."""
    if not keywords:
        return True
    lower_text = text.lower()
    return any(kw.lower() in lower_text for kw in keywords)


def fetch_historical_data(
    symbol: str, start: date, end: date
) -> pd.DataFrame:
    """Fetch OHLCV historical data for an NSE/BSE symbol using yfinance."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end)
    if hist.empty:
        raise ValueError(f"No historical data returned for {symbol}.")
    return hist


def fetch_news_for_stock(
    company_name: str,
    max_articles: int = 10,
    symbol: Optional[str] = None,
) -> Tuple[List[NewsItem], Optional[str]]:
    """Fetch recent news for a company from Indian financial RSS feeds.

    Returns (news_items, warning_message). No API key required.
    Pass symbol (e.g. RELIANCE.NS) to improve relevance via ticker keywords.
    """
    keywords = _extract_keywords(company_name, symbol=symbol)
    seen_urls: set = set()
    matched_items: List[NewsItem] = []
    unfiltered_items: List[NewsItem] = []
    failed_count = 0
    MIN_MATCHES = 3

    for url in RSS_FEED_URLS:
        raw = _fetch_rss_feed(url)
        if not raw:
            failed_count += 1
            continue

        for r in raw:
            link = (r.get("link") or "").strip()
            if not link or link in seen_urls:
                continue

            title = r.get("title") or ""
            summary = r.get("summary") or ""
            combined = f"{title} {summary}"

            news_item = NewsItem(
                title=title,
                description=summary[:500] if summary else "",
                url=link,
                published_at=r.get("published") or "",
                source=r.get("source") or "RSS",
            )

            if keywords and _matches_keywords(combined, keywords):
                seen_urls.add(link)
                matched_items.append(news_item)
            else:
                unfiltered_items.append(news_item)

    # If too few keyword matches, include generic market news
    items = matched_items
    if len(items) < MIN_MATCHES and unfiltered_items:
        for n in unfiltered_items:
            if n.url not in seen_urls and len(items) < max_articles:
                seen_urls.add(n.url)
                items.append(n)
                if len(items) >= max_articles:
                    break

    # Sort by published date (best-effort; RSS dates vary in format)
    def _sort_key(n: NewsItem) -> str:
        return n.published_at or ""

    items.sort(key=_sort_key, reverse=True)
    items = items[:max_articles]

    warning = None
    if failed_count == len(RSS_FEED_URLS):
        warning = "All RSS feeds failed; no news available."

    return items, warning


def simple_healthcheck() -> bool:
    """Lightweight check to ensure we can hit an external endpoint."""
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except Exception:
        return False
