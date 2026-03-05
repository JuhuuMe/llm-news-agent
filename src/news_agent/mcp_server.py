"""MCP server that exposes the financial news agent as tools.

Provides both raw data tools (fetch_news, stock_price, etc.) and LLM-powered
analysis tools (news_overview, ask_news, check_breaking) that use a local model
(e.g. Qwen 3.5 8B via vLLM) to produce structured market analysis.

Run with:
    news-agent-mcp                     # stdio transport (default)
    news-agent-mcp --transport sse     # SSE transport on port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from news_agent.agent import MarketNewsAgent
from news_agent.config import Settings, load_settings
from news_agent.extraction import extract_from_article
from news_agent.graph import GraphStore
from news_agent.llm import LLM
from news_agent.models import UserInterests
from news_agent.qa import answer_question
from news_agent.sources.alphavantage import AlphaVantageSource
from news_agent.sources.base import NewsSource
from news_agent.sources.finnhub import FinnhubSource
from news_agent.sources.newsapi import NewsAPISource
from news_agent.sources.reddit import RedditSource
from news_agent.sources.rss import RSSSource
from news_agent.tools import execute_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state — initialised lazily on first tool call
# ---------------------------------------------------------------------------

_settings: Settings | None = None
_graph: GraphStore | None = None
_sources: list[NewsSource] | None = None
_llm: LLM | None = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def _get_graph() -> GraphStore:
    global _graph
    if _graph is None:
        _graph = GraphStore(Path("news_graph.db"))
    return _graph


def _get_sources() -> list[NewsSource]:
    global _sources
    if _sources is None:
        s = _get_settings()
        _sources = []
        if s.enable_rss:
            _sources.append(RSSSource())
        if s.enable_reddit:
            _sources.append(RedditSource())
        if s.newsapi_key:
            _sources.append(NewsAPISource(s.newsapi_key))
        if s.finnhub_key:
            _sources.append(FinnhubSource(s.finnhub_key))
        if s.alphavantage_key:
            _sources.append(AlphaVantageSource(s.alphavantage_key))
    return _sources


def _get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM(_get_settings())
    return _llm


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------


async def _prewarm() -> None:
    """Prewarm: ping vLLM, fetch news, build knowledge graph."""
    logger.info("Prewarming MCP server…")

    # Init settings, graph, sources
    settings = _get_settings()
    graph = _get_graph()
    sources = _get_sources()

    # 1. Ping vLLM to trigger model loading
    llm_ready = False
    if settings.llm_backend == "vllm":
        llm = _get_llm()
        try:
            await llm.generate("Say OK.", max_tokens=4)
            logger.info("vLLM ping OK — model is warm.")
            llm_ready = True
        except Exception as exc:
            logger.warning("vLLM ping failed: %s", exc)
    elif settings.llm_backend == "api" and settings.hf_token:
        llm = _get_llm()
        try:
            await llm.generate("Say OK.", max_tokens=4)
            logger.info("HF API ping OK.")
            llm_ready = True
        except Exception as exc:
            logger.warning("HF API ping failed: %s", exc)
    elif settings.llm_backend == "local":
        llm_ready = True

    # 2. Fetch news and build knowledge graph
    if sources:
        logger.info("Fetching news for initial knowledge graph…")
        query = "stock market economy inflation"
        tasks = [source.fetch(query) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        articles = []
        seen_urls: set[str] = set()
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Source fetch failed during prewarm: %s", result)
                continue
            for article in result:
                url = article.url.strip()
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    articles.append(article)

        articles.sort(
            key=lambda a: a.published_at or datetime.min, reverse=True,
        )
        logger.info("Fetched %d articles from %d sources.", len(articles), len(sources))

        # 3. Run entity extraction + graph ingest (requires LLM)
        if llm_ready and articles:
            llm = _get_llm()
            graph.memory_cycle()
            ingested = 0
            for article in articles:
                try:
                    ext = await extract_from_article(llm, article)
                    graph.ingest_extraction(
                        article, ext.entities, ext.events, ext.relationships,
                    )
                    ingested += 1
                except Exception as exc:
                    logger.warning("Ingest failed: %s", exc)
            stats = graph.stats()
            logger.info(
                "Knowledge graph ready: %d articles ingested, "
                "%d entities, %d events, %d edges.",
                ingested, stats["entities"], stats["events"], stats["edges"],
            )
        elif not llm_ready:
            logger.warning(
                "LLM not available — skipping entity extraction. "
                "Graph will be built on first ingest_news call.",
            )

    logger.info("Prewarm complete.")


@asynccontextmanager
async def server_lifespan(server):
    """Run prewarm on startup, cleanup on shutdown."""
    await _prewarm()
    try:
        yield
    finally:
        if _graph is not None:
            _graph.close()


mcp = FastMCP(
    "Financial News Agent",
    instructions=(
        "Financial news agent with access to live market data, news feeds, "
        "and a knowledge graph.  Use fetch_news to get recent articles, "
        "search_graph to query the knowledge graph, and the individual market "
        "data tools (stock_price, crypto_price, etc.) for live quotes."
    ),
    lifespan=server_lifespan,
)


# ---------------------------------------------------------------------------
# News tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def fetch_news(
    topics: str = "stock market, inflation, oil",
    tickers: str = "",
    max_results: int = 30,
) -> str:
    """Fetch the latest financial news from multiple sources.

    Returns a list of recent articles with titles, summaries, sources, and timestamps.
    Articles are deduplicated across sources and sorted newest-first.

    Args:
        topics: Comma-separated topics to search for (e.g. "oil, inflation, Fed")
        tickers: Comma-separated stock tickers for ticker-specific news (e.g. "AAPL,TSLA")
        max_results: Maximum number of articles to return (default 30)
    """
    sources = _get_sources()
    if not sources:
        return "No news sources configured. Set API keys in .env or enable RSS/Reddit."

    query = topics.strip()
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    tasks = [source.fetch(query, max_results=max_results) for source in sources]

    # Ticker-specific fetches from Finnhub
    for source in sources:
        if isinstance(source, FinnhubSource) and ticker_list:
            for ticker in ticker_list[:5]:
                tasks.append(source.fetch_for_ticker(ticker, max_results=10))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles = []
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

    articles.sort(key=lambda a: a.published_at or datetime.min, reverse=True)
    articles = articles[:max_results]

    if not articles:
        return f"No articles found for topics='{topics}', tickers='{tickers}'."

    lines = [f"Found {len(articles)} articles:\n"]
    for i, a in enumerate(articles, 1):
        ts = a.published_at.strftime("%Y-%m-%d %H:%M") if a.published_at else "unknown"
        sentiment = f" [sentiment={a.raw_sentiment:.2f}]" if a.raw_sentiment is not None else ""
        lines.append(f"[{i}] {a.title}")
        lines.append(f"    Source: {a.source_name} | {ts}{sentiment}")
        if a.summary:
            lines.append(f"    {a.summary[:300]}")
        if a.url:
            lines.append(f"    URL: {a.url}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
async def ingest_news(
    topics: str = "stock market, inflation, oil",
    tickers: str = "",
) -> str:
    """Fetch news and ingest into the knowledge graph with entity/event extraction.

    This builds the knowledge graph by fetching articles, extracting entities
    (companies, commodities, institutions) and events (rate decisions, earnings),
    then storing them with relationships.  Run this before using search_graph.

    Args:
        topics: Comma-separated topics to search for
        tickers: Comma-separated stock tickers for ticker-specific news
    """
    sources = _get_sources()
    settings = _get_settings()
    graph = _get_graph()

    if not sources:
        return "No news sources configured."

    # Check if LLM is available (needed for entity extraction)
    if not settings.hf_token and settings.llm_backend == "api":
        return (
            "HF_TOKEN not set — entity extraction requires an LLM backend. "
            "Set HF_TOKEN in .env, or use LLM_BACKEND=vllm / LLM_BACKEND=local."
        )

    llm = _get_llm()
    query = topics.strip()
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    tasks = [source.fetch(query) for source in sources]
    for source in sources:
        if isinstance(source, FinnhubSource) and ticker_list:
            for ticker in ticker_list[:5]:
                tasks.append(source.fetch_for_ticker(ticker))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles = []
    seen_urls: set[str] = set()
    for result in results:
        if isinstance(result, Exception):
            continue
        for article in result:
            url = article.url.strip()
            if url and url in seen_urls:
                continue
            seen_urls.add(url)
            articles.append(article)

    articles.sort(key=lambda a: a.published_at or datetime.min, reverse=True)

    # Run memory cycle (decay + prune)
    graph.memory_cycle()

    ingested = 0
    for article in articles:
        try:
            result = await extract_from_article(llm, article)
            graph.ingest_extraction(article, result.entities, result.events, result.relationships)
            ingested += 1
        except Exception as exc:
            logger.warning("Ingest failed for '%s': %s", article.title[:60], exc)

    stats = graph.stats()
    return (
        f"Ingested {ingested}/{len(articles)} articles into the knowledge graph.\n"
        f"Graph now has: {stats['articles']} articles, {stats['entities']} entities, "
        f"{stats['events']} events, {stats['edges']} edges."
    )


@mcp.tool()
async def search_graph(entity: str) -> str:
    """Search the knowledge graph for articles, events, and relationships about an entity.

    Returns related articles, events that affect this entity, and connected entities
    from the knowledge graph.  Run ingest_news first to populate the graph.

    Args:
        entity: Entity name to search for (e.g. "AAPL", "Federal Reserve", "OIL")
    """
    graph = _get_graph()
    subgraph = graph.get_subgraph_for_entity(entity)
    sections: list[str] = []

    if subgraph["articles"]:
        lines = []
        for a in subgraph["articles"][:20]:
            sentiment = f" [sentiment={a['sentiment']:.2f}]" if a.get("sentiment") else ""
            lines.append(
                f"- {a['title']} ({a['source_name']}, {a['published_at'] or 'unknown'})"
                f"{sentiment} [relevance={a['relevance']:.2f}]"
            )
            if a.get("summary"):
                lines.append(f"  {a['summary'][:200]}")
        sections.append(f"RELATED ARTICLES ({len(subgraph['articles'])}):\n" + "\n".join(lines))

    if subgraph["events"]:
        lines = []
        seen: set[str] = set()
        for ev in subgraph["events"]:
            desc = ev["description"][:100]
            if desc not in seen:
                seen.add(desc)
                direction = f" → {ev['direction'].upper()}" if ev.get("direction") else ""
                lines.append(f"- [{ev['event_type']}] {ev['description']}{direction}")
        if lines:
            sections.append("RELATED EVENTS:\n" + "\n".join(lines[:15]))

    if subgraph["related_entities"]:
        lines = []
        seen_names: set[str] = set()
        for r in subgraph["related_entities"]:
            if r["name"] not in seen_names:
                seen_names.add(r["name"])
                direction = f" [{r['direction']}]" if r.get("direction") else ""
                lines.append(f"- {r['name']} ({r['entity_type']}) — {r['relation']}{direction}")
        if lines:
            sections.append("RELATED ENTITIES:\n" + "\n".join(lines[:15]))

    if not sections:
        return (
            f"No data found for '{entity}' in the knowledge graph. "
            "Try running ingest_news first."
        )

    return "\n\n".join(sections)


@mcp.tool()
async def graph_stats() -> str:
    """Get knowledge graph statistics: counts and memory relevance."""
    stats = _get_graph().stats()
    lines = [
        f"Articles:  {stats['articles']}",
        f"Entities:  {stats['entities']}",
        f"Events:    {stats['events']}",
        f"Edges:     {stats['edges']}",
    ]
    for table in ("articles", "entities", "events"):
        avg = stats.get(f"{table}_avg_relevance")
        if avg is not None:
            lines.append(f"  {table} avg relevance: {avg:.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live market data tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def stock_price(ticker: str) -> str:
    """Get the latest stock quote for a ticker symbol.

    Returns current price, open, high, low, previous close, and change percentage.
    Uses Finnhub (primary) with Alpha Vantage fallback.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL", "MSFT", "TSLA")
    """
    result = await execute_tool("stock_price", ticker, _get_settings())
    return result.output


@mcp.tool()
async def crypto_price(symbol: str) -> str:
    """Get the latest cryptocurrency price in USD.

    Uses Finnhub (Binance) with Alpha Vantage fallback.

    Args:
        symbol: Crypto symbol (e.g. "BTC", "ETH", "SOL")
    """
    result = await execute_tool("crypto_price", symbol, _get_settings())
    return result.output


@mcp.tool()
async def forex_rate(from_currency: str, to_currency: str) -> str:
    """Get the latest foreign exchange rate between two currencies.

    Uses Alpha Vantage (primary) with Finnhub fallback.

    Args:
        from_currency: Source currency code (e.g. "EUR", "GBP", "JPY")
        to_currency: Target currency code (e.g. "USD", "EUR")
    """
    result = await execute_tool("forex_rate", f"{from_currency},{to_currency}", _get_settings())
    return result.output


@mcp.tool()
async def economic_indicator(series_id: str) -> str:
    """Get the latest value of a FRED economic indicator.

    Returns the most recent 5 data points plus series metadata.
    Requires a FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html).

    Args:
        series_id: FRED series ID. Common ones:
            CPIAUCSL (CPI), FEDFUNDS (Fed Funds Rate), UNRATE (Unemployment),
            GDP (GDP), DGS10 (10Y Treasury), PCEPILFE (Core PCE), M2SL (M2 Money Supply)
    """
    result = await execute_tool("economic_indicator", series_id, _get_settings())
    return result.output


@mcp.tool()
async def market_index(symbol: str) -> str:
    """Get the latest market index level via ETF proxy.

    Supports common index names and ETF symbols.

    Args:
        symbol: Index name or ETF (e.g. "SPY", "QQQ", "S&P500", "NASDAQ")
    """
    result = await execute_tool("market_index", symbol, _get_settings())
    return result.output


@mcp.tool()
async def commodity_price(symbol: str) -> str:
    """Get commodity price via ETF proxy.

    Args:
        symbol: Commodity name or ETF symbol.
            Common: GLD/GOLD (gold), USO/OIL (oil), SLV/SILVER (silver),
            UNG/NATGAS (natural gas), CPER/COPPER, WEAT/WHEAT, CORN
    """
    result = await execute_tool("commodity_price", symbol, _get_settings())
    return result.output


@mcp.tool()
async def company_profile(ticker: str) -> str:
    """Get basic company information — sector, market cap, exchange, IPO date.

    Requires a Finnhub API key.

    Args:
        ticker: Stock ticker symbol (e.g. "NVDA", "AAPL", "GOOGL")
    """
    result = await execute_tool("company_profile", ticker, _get_settings())
    return result.output


# ---------------------------------------------------------------------------
# LLM-powered analysis tools (Qwen inside the MCP server)
# ---------------------------------------------------------------------------


def _build_agent(
    topics: str, tickers: str,
) -> MarketNewsAgent:
    """Build a MarketNewsAgent with the given interests."""
    settings = _get_settings()
    topic_list = [t.strip() for t in topics.split(",") if t.strip()]
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    interests = UserInterests(
        topics=topic_list or ["stock market"],
        tickers=ticker_list,
        regions=[],
    )
    return MarketNewsAgent(settings, interests)


@mcp.tool()
async def news_overview(
    topics: str = "stock market, inflation, oil",
    tickers: str = "",
) -> str:
    """Get a structured daily market overview analyzed by the local LLM.

    Fetches news, then uses the local LLM (e.g. Qwen 3.5 8B) to cluster
    articles into topics, assess market impact, and assign urgency levels.
    Returns a structured digest with market mood, topic clusters,
    affected assets, and directional calls.

    Requires LLM_BACKEND to be configured (vllm, api, or local).

    Args:
        topics: Comma-separated topics (e.g. "oil, Fed, tech earnings")
        tickers: Comma-separated stock tickers (e.g. "AAPL,TSLA")
    """
    settings = _get_settings()
    if not settings.hf_token and settings.llm_backend == "api":
        return (
            "HF_TOKEN not set. Set HF_TOKEN in .env, "
            "or use LLM_BACKEND=vllm / LLM_BACKEND=local."
        )

    agent = _build_agent(topics, tickers)
    try:
        overview = await agent.daily_overview()
    except Exception as exc:
        return f"Overview generation failed: {exc}"

    lines = [f"MARKET MOOD: {overview.market_mood}\n"]
    for i, d in enumerate(overview.digests, 1):
        lines.append(f"[{i}] {d.headline} [{d.urgency.value.upper()}]")
        lines.append(f"    {d.summary}")
        if d.impact.affected_assets:
            assets = ", ".join(d.impact.affected_assets)
            lines.append(
                f"    Impact: {assets} → {d.impact.direction} "
                f"({d.impact.confidence})"
            )
            if d.impact.reasoning:
                lines.append(f"    Reasoning: {d.impact.reasoning}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
async def ask_news(
    question: str,
    topics: str = "stock market, inflation",
    tickers: str = "",
) -> str:
    """Ask a question and get an answer using the knowledge graph + local LLM.

    Uses the full Q&A pipeline: extracts entities from your question,
    traverses the knowledge graph for context, then runs a ReAct tool
    loop (up to 4 rounds of live data lookups) using the local LLM
    (e.g. Qwen 3.5 8B) to produce a structured answer with consensus
    view, outlier perspectives, key drivers, and a watch list.

    Requires the knowledge graph to be populated (run ingest_news first)
    and an LLM backend to be configured.

    Args:
        question: Your market/financial question
        topics: Context topics for graph search
        tickers: Context tickers for graph search
    """
    settings = _get_settings()
    if not settings.hf_token and settings.llm_backend == "api":
        return (
            "HF_TOKEN not set. Set HF_TOKEN in .env, "
            "or use LLM_BACKEND=vllm / LLM_BACKEND=local."
        )

    llm = _get_llm()
    graph = _get_graph()

    try:
        return await answer_question(question, llm, graph, settings)
    except Exception as exc:
        return f"Q&A failed: {exc}"


@mcp.tool()
async def check_breaking(
    topics: str = "stock market, inflation, oil",
    tickers: str = "",
) -> str:
    """Check for breaking / urgent market-moving news using the local LLM.

    Fetches latest articles and uses the local LLM (e.g. Qwen 3.5 8B)
    to identify truly urgent events: unexpected rate decisions, major
    earnings surprises, geopolitical shocks, flash crashes, M&A, etc.

    Returns only genuinely market-moving alerts, not routine news.
    Requires an LLM backend to be configured.

    Args:
        topics: Comma-separated topics to monitor
        tickers: Comma-separated stock tickers to monitor
    """
    settings = _get_settings()
    if not settings.hf_token and settings.llm_backend == "api":
        return (
            "HF_TOKEN not set. Set HF_TOKEN in .env, "
            "or use LLM_BACKEND=vllm / LLM_BACKEND=local."
        )

    agent = _build_agent(topics, tickers)
    try:
        alerts = await agent.check_breaking()
    except Exception as exc:
        return f"Breaking news check failed: {exc}"

    if not alerts:
        return "No breaking news detected. Markets appear calm."

    lines = []
    for i, alert in enumerate(alerts, 1):
        d = alert.digest
        lines.append(f"[BREAKING {i}] {d.headline}")
        lines.append(f"  {d.summary}")
        lines.append(f"  Reason: {alert.reason}")
        if d.impact.affected_assets:
            assets = ", ".join(d.impact.affected_assets)
            lines.append(
                f"  Impact: {assets} → {d.impact.direction} "
                f"({d.impact.confidence})"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Financial News Agent MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
