"""RSS/Atom feed adapter — free, no auth, no rate limits.

Curated list of major financial, macro, and political news feeds.
"""

from __future__ import annotations

import logging
from datetime import datetime
from email.utils import parsedate_to_datetime

import httpx

from news_agent.models import Article
from news_agent.sources.base import NewsSource

logger = logging.getLogger(__name__)

# Curated feeds grouped by category. Each feed is (name, url).
FEEDS: dict[str, list[tuple[str, str]]] = {
    "macro_economy": [
        ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
        ("Reuters Markets", "https://feeds.reuters.com/reuters/marketsNews"),
        ("AP Business", "https://rsshub.app/apnews/topics/business"),
        ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("CNBC Markets", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069"),
    ],
    "politics_elections": [
        ("Reuters World", "https://feeds.reuters.com/Reuters/worldNews"),
        ("AP Politics", "https://rsshub.app/apnews/topics/politics"),
        ("NPR Politics", "https://feeds.npr.org/1014/rss.xml"),
        ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
        ("The Guardian US", "https://www.theguardian.com/us-news/rss"),
    ],
    "central_banks_inflation": [
        ("Reuters Fed", "https://feeds.reuters.com/reuters/bankruptcyNews"),
        ("CNBC Fed", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20409666"),
    ],
    "commodities_energy": [
        ("Reuters Commodities", "https://feeds.reuters.com/reuters/commoditiesNews"),
        ("CNBC Energy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19836768"),
    ],
    "tech": [
        ("Reuters Tech", "https://feeds.reuters.com/reuters/technologyNews"),
        ("CNBC Tech", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19854910"),
    ],
}

# Map user-facing topic keywords → feed categories
_TOPIC_TO_CATEGORIES: dict[str, str] = {
    "macro": "macro_economy",
    "economy": "macro_economy",
    "market": "macro_economy",
    "stock": "macro_economy",
    "sp500": "macro_economy",
    "s&p": "macro_economy",
    "inflation": "central_banks_inflation",
    "fed": "central_banks_inflation",
    "central bank": "central_banks_inflation",
    "interest rate": "central_banks_inflation",
    "politics": "politics_elections",
    "election": "politics_elections",
    "geopolit": "politics_elections",
    "oil": "commodities_energy",
    "commodit": "commodities_energy",
    "energy": "commodities_energy",
    "gold": "commodities_energy",
    "tech": "tech",
    "crypto": "tech",
}


class RSSSource(NewsSource):
    """Fetches news from curated RSS/Atom feeds. No API key required."""

    name = "RSS"

    def __init__(self, categories: list[str] | None = None) -> None:
        """Initialize with specific categories, or None for all."""
        self._categories = categories

    async def fetch(self, query: str, max_results: int = 30) -> list[Article]:
        categories = self._resolve_categories(query)
        feeds = []
        for cat in categories:
            feeds.extend(FEEDS.get(cat, []))

        # Deduplicate feeds by URL
        seen: set[str] = set()
        unique_feeds = []
        for name, url in feeds:
            if url not in seen:
                seen.add(url)
                unique_feeds.append((name, url))

        articles: list[Article] = []
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            for feed_name, feed_url in unique_feeds:
                try:
                    fetched = await self._fetch_feed(client, feed_name, feed_url, query)
                    articles.extend(fetched)
                except Exception as exc:
                    logger.debug("RSS feed %s failed: %s", feed_name, exc)

        # Sort newest first, limit
        articles.sort(
            key=lambda a: a.published_at or datetime.min,
            reverse=True,
        )
        return articles[:max_results]

    def _resolve_categories(self, query: str) -> list[str]:
        """Map query keywords to feed categories."""
        if self._categories:
            return self._categories

        query_lower = query.lower()
        matched: set[str] = set()
        for keyword, category in _TOPIC_TO_CATEGORIES.items():
            if keyword in query_lower:
                matched.add(category)

        # Default: macro + politics (the core use case)
        if not matched:
            matched = {"macro_economy", "politics_elections"}
        return list(matched)

    async def _fetch_feed(
        self,
        client: httpx.AsyncClient,
        feed_name: str,
        feed_url: str,
        query: str,
    ) -> list[Article]:
        """Fetch and parse a single RSS/Atom feed."""
        resp = await client.get(feed_url, headers={"User-Agent": "llm-news-agent/0.1"})
        resp.raise_for_status()
        xml_text = resp.text

        return _parse_feed_xml(xml_text, feed_name, query)


def _parse_feed_xml(xml_text: str, source_name: str, query: str) -> list[Article]:
    """Minimal RSS/Atom parser using only stdlib xml.etree.

    We avoid feedparser dependency to keep the install lightweight.
    Handles both RSS 2.0 (<item>) and Atom (<entry>) feeds.
    """
    import xml.etree.ElementTree as ET

    articles: list[Article] = []
    query_keywords = {kw.lower() for kw in query.split() if len(kw) > 2}

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    # Handle namespaces common in Atom feeds
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    # Try RSS 2.0 items
    items = root.findall(".//item")
    # Try Atom entries
    if not items:
        items = root.findall(".//atom:entry", ns)
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    for item in items:
        title = _get_text(item, "title", ns) or ""
        summary = (
            _get_text(item, "description", ns)
            or _get_text(item, "summary", ns)
            or _get_text(item, "atom:summary", ns)
            or ""
        )
        link = (
            _get_text(item, "link", ns)
            or _get_attr(item, "link", "href", ns)
            or _get_attr(item, "atom:link", "href", ns)
            or ""
        )
        pub_date_str = (
            _get_text(item, "pubDate", ns)
            or _get_text(item, "published", ns)
            or _get_text(item, "atom:published", ns)
            or _get_text(item, "updated", ns)
            or _get_text(item, "atom:updated", ns)
            or ""
        )

        published = _parse_date(pub_date_str)

        # Light keyword filtering — if query has keywords, check title+summary
        if query_keywords:
            text_lower = (title + " " + summary).lower()
            if not any(kw in text_lower for kw in query_keywords):
                continue

        # Strip HTML tags from summary
        summary = _strip_html(summary)

        articles.append(
            Article(
                title=title.strip(),
                summary=summary.strip()[:500],
                url=link.strip(),
                source_name=source_name,
                published_at=published,
            )
        )

    return articles


def _get_text(element: "ET.Element", tag: str, ns: dict) -> str | None:
    """Find a child element and return its text."""
    import xml.etree.ElementTree as ET

    # Try without namespace
    child = element.find(tag)
    if child is not None and child.text:
        return child.text
    # Try with atom namespace
    for prefix, uri in ns.items():
        child = element.find(f"{{{uri}}}{tag.replace(f'{prefix}:', '')}")
        if child is not None and child.text:
            return child.text
    return None


def _get_attr(element: "ET.Element", tag: str, attr: str, ns: dict) -> str | None:
    """Find a child element and return an attribute."""
    import xml.etree.ElementTree as ET

    child = element.find(tag)
    if child is not None:
        return child.get(attr)
    for prefix, uri in ns.items():
        child = element.find(f"{{{uri}}}{tag.replace(f'{prefix}:', '')}")
        if child is not None:
            return child.get(attr)
    return None


def _parse_date(date_str: str) -> datetime | None:
    """Best-effort date parsing for RSS/Atom dates."""
    if not date_str:
        return None
    # Try RFC 2822 (RSS pubDate format)
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        pass
    # Try ISO 8601 (Atom format)
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        pass
    return None


def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    import re
    clean = re.sub(r"<[^>]+>", "", text)
    clean = clean.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    clean = clean.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return clean
