"""Core agent — fetches news, deduplicates, builds knowledge graph, and produces analysis via LLM."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from news_agent.config import Settings
from news_agent.extraction import extract_from_article
from news_agent.graph import GraphStore
from news_agent.llm import LLM, extract_json
from news_agent.models import (
    Article,
    BreakingAlert,
    DailyOverview,
    MarketImpact,
    NewsDigest,
    Urgency,
    UserInterests,
)
from news_agent.qa import answer_question
from news_agent.sources.alphavantage import AlphaVantageSource
from news_agent.sources.base import NewsSource
from news_agent.sources.finnhub import FinnhubSource
from news_agent.sources.newsapi import NewsAPISource
from news_agent.sources.reddit import RedditSource
from news_agent.sources.rss import RSSSource

logger = logging.getLogger(__name__)


class MarketNewsAgent:
    """Orchestrates news fetching, graph building, and LLM-based analysis."""

    def __init__(self, settings: Settings, interests: UserInterests) -> None:
        self.settings = settings
        self.interests = interests
        self.llm = LLM(settings)
        self.sources = self._build_sources()
        self.graph = GraphStore(Path("news_graph.db"))
        # Track already-seen article URLs to avoid re-alerting in watch mode
        self._seen_urls: set[str] = set()

    def _build_sources(self) -> list[NewsSource]:
        sources: list[NewsSource] = []
        # Free sources (no API key needed)
        if self.settings.enable_rss:
            sources.append(RSSSource())
        if self.settings.enable_reddit:
            sources.append(RedditSource())
        # API-key sources
        if self.settings.newsapi_key:
            sources.append(NewsAPISource(self.settings.newsapi_key))
        if self.settings.finnhub_key:
            sources.append(FinnhubSource(self.settings.finnhub_key))
        if self.settings.alphavantage_key:
            sources.append(AlphaVantageSource(self.settings.alphavantage_key))
        return sources

    # ------------------------------------------------------------------
    # Fetch & deduplicate
    # ------------------------------------------------------------------
    async def fetch_all(self) -> list[Article]:
        """Fetch from all configured sources in parallel, deduplicate by URL."""
        query = " OR ".join(self.interests.topics[:5])  # combine top topics

        tasks = [source.fetch(query) for source in self.sources]
        # Also fetch ticker-specific news from Finnhub if tickers configured
        for source in self.sources:
            if isinstance(source, FinnhubSource) and self.interests.tickers:
                for ticker in self.interests.tickers[:5]:
                    tasks.append(source.fetch_for_ticker(ticker))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        articles: list[Article] = []
        seen_urls: set[str] = set()
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Source fetch failed: %s", result)
                continue
            for article in result:
                url = article.url.strip()
                if url and url in seen_urls:
                    continue
                seen_urls.add(url)
                articles.append(article)

        # Sort newest first
        articles.sort(
            key=lambda a: a.published_at or datetime.min,
            reverse=True,
        )
        return articles

    async def ingest_to_graph(self, articles: list[Article]) -> int:
        """Run entity extraction on articles and ingest into the knowledge graph.

        Returns the number of articles successfully ingested.
        """
        # Prune stale data first
        self.graph.prune_older_than_days(7)

        ingested = 0
        for article in articles:
            try:
                result = await extract_from_article(self.llm, article)
                self.graph.ingest_extraction(
                    article, result.entities, result.events, result.relationships
                )
                ingested += 1
            except Exception as exc:
                logger.warning("Graph ingest failed for '%s': %s", article.title[:60], exc)
        return ingested

    # ------------------------------------------------------------------
    # Q&A (three-phase: data → agentic AI → report)
    # ------------------------------------------------------------------
    async def ask(self, question: str) -> str:
        """Answer a user question using the knowledge graph + live data tools."""
        return await answer_question(question, self.llm, self.graph, self.settings)

    # ------------------------------------------------------------------
    # Daily overview
    # ------------------------------------------------------------------
    async def daily_overview(self) -> DailyOverview:
        """Produce a full daily market overview."""
        articles = await self.fetch_all()
        if not articles:
            return DailyOverview(market_mood="No articles found for your interests.")

        prompt = self._build_overview_prompt(articles)
        raw = await self.llm.generate(prompt, max_tokens=2048)
        return self._parse_overview(raw, articles)

    def _build_overview_prompt(self, articles: list[Article]) -> str:
        article_block = "\n".join(
            f"- [{i+1}] {a.title} | {a.source_name} | {a.published_at or 'unknown time'}"
            + (f" | sentiment={a.raw_sentiment:.2f}" if a.raw_sentiment is not None else "")
            + (f"\n  {a.summary[:200]}" if a.summary else "")
            for i, a in enumerate(articles[:40])
        )

        interests_str = ", ".join(self.interests.topics)
        tickers_str = ", ".join(self.interests.tickers) if self.interests.tickers else "none"

        return f"""You are a financial market analyst AI. Analyze the following news articles and produce a structured daily market overview.

User interests: {interests_str}
Tracked tickers: {tickers_str}

Articles:
{article_block}

Instructions:
1. Group related articles into 3-8 topic clusters.
2. For each cluster provide:
   - headline: a concise headline
   - summary: 2-3 sentence summary of the news
   - urgency: one of "breaking", "high", "normal", "low"
   - impact: {{affected_assets: [...], direction: "bullish"/"bearish"/"neutral"/"mixed", confidence: "high"/"medium"/"low", reasoning: "..."}}
   - source_ids: list of article numbers [1], [2], etc.
3. Provide an overall market_mood (one sentence).

Respond ONLY with valid JSON in this exact format:
```json
{{
  "market_mood": "...",
  "digests": [
    {{
      "headline": "...",
      "summary": "...",
      "urgency": "normal",
      "impact": {{
        "affected_assets": ["SPY", "QQQ"],
        "direction": "bullish",
        "confidence": "medium",
        "reasoning": "..."
      }},
      "source_ids": [1, 3]
    }}
  ]
}}
```"""

    def _parse_overview(self, raw: str, articles: list[Article]) -> DailyOverview:
        data = extract_json(raw)
        if not isinstance(data, dict):
            # Fallback: return raw text as a single digest
            return DailyOverview(
                market_mood="Could not parse structured output.",
                digests=[
                    NewsDigest(
                        headline="Raw Analysis",
                        summary=raw[:1000],
                        source_articles=articles[:10],
                    )
                ],
            )

        digests: list[NewsDigest] = []
        for d in data.get("digests", []):
            source_ids = d.get("source_ids", [])
            linked = [articles[i - 1] for i in source_ids if 0 < i <= len(articles)]
            impact_raw = d.get("impact", {})
            digests.append(
                NewsDigest(
                    headline=d.get("headline", ""),
                    summary=d.get("summary", ""),
                    urgency=Urgency(d.get("urgency", "normal")),
                    impact=MarketImpact(
                        affected_assets=impact_raw.get("affected_assets", []),
                        direction=impact_raw.get("direction", ""),
                        confidence=impact_raw.get("confidence", ""),
                        reasoning=impact_raw.get("reasoning", ""),
                    ),
                    source_articles=linked,
                )
            )

        return DailyOverview(
            market_mood=data.get("market_mood", ""),
            digests=digests,
        )

    # ------------------------------------------------------------------
    # Breaking news detection
    # ------------------------------------------------------------------
    async def check_breaking(self) -> list[BreakingAlert]:
        """Fetch latest news and flag anything that looks urgent/breaking."""
        articles = await self.fetch_all()
        # Filter to articles not yet seen
        new_articles = [a for a in articles if a.url not in self._seen_urls]
        if not new_articles:
            return []

        # Mark as seen
        for a in new_articles:
            if a.url:
                self._seen_urls.add(a.url)

        prompt = self._build_breaking_prompt(new_articles)
        raw = await self.llm.generate(prompt, max_tokens=1024)
        return self._parse_breaking(raw, new_articles)

    def _build_breaking_prompt(self, articles: list[Article]) -> str:
        article_block = "\n".join(
            f"- [{i+1}] {a.title} | {a.source_name}"
            + (f"\n  {a.summary[:200]}" if a.summary else "")
            for i, a in enumerate(articles[:30])
        )

        interests_str = ", ".join(self.interests.topics)

        return f"""You are a financial market analyst AI monitoring for breaking news.

User interests: {interests_str}

New articles since last check:
{article_block}

Identify ONLY articles that require IMMEDIATE attention — major market-moving events such as:
- Unexpected central bank decisions
- Major earnings surprises (>10% miss/beat)
- Geopolitical shocks affecting markets
- Flash crashes or extreme volatility
- Major M&A announcements
- Regulatory actions against major companies

If nothing is truly breaking, respond with: {{"alerts": []}}

Otherwise respond with JSON:
```json
{{
  "alerts": [
    {{
      "headline": "...",
      "summary": "...",
      "reason": "Why this needs immediate attention",
      "impact": {{
        "affected_assets": [...],
        "direction": "bearish",
        "confidence": "high",
        "reasoning": "..."
      }},
      "source_ids": [1]
    }}
  ]
}}
```"""

    def _parse_breaking(self, raw: str, articles: list[Article]) -> list[BreakingAlert]:
        data = extract_json(raw)
        if not isinstance(data, dict):
            return []

        alerts: list[BreakingAlert] = []
        for a in data.get("alerts", []):
            source_ids = a.get("source_ids", [])
            linked = [articles[i - 1] for i in source_ids if 0 < i <= len(articles)]
            impact_raw = a.get("impact", {})
            alerts.append(
                BreakingAlert(
                    digest=NewsDigest(
                        headline=a.get("headline", ""),
                        summary=a.get("summary", ""),
                        urgency=Urgency.BREAKING,
                        impact=MarketImpact(
                            affected_assets=impact_raw.get("affected_assets", []),
                            direction=impact_raw.get("direction", ""),
                            confidence=impact_raw.get("confidence", ""),
                            reasoning=impact_raw.get("reasoning", ""),
                        ),
                        source_articles=linked,
                    ),
                    reason=a.get("reason", ""),
                )
            )
        return alerts
