#!/usr/bin/env python3
"""CLI entry point for the financial market news agent."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from news_agent.agent import MarketNewsAgent
from news_agent.config import load_settings
from news_agent.models import BreakingAlert, DailyOverview, Urgency, UserInterests

console = Console()

URGENCY_STYLES = {
    Urgency.BREAKING: "bold red",
    Urgency.HIGH: "bold yellow",
    Urgency.NORMAL: "white",
    Urgency.LOW: "dim",
}

DIRECTION_ICONS = {
    "bullish": "[green]▲ Bullish[/green]",
    "bearish": "[red]▼ Bearish[/red]",
    "neutral": "[dim]● Neutral[/dim]",
    "mixed": "[yellow]◆ Mixed[/yellow]",
}


def collect_interests() -> UserInterests:
    """Interactive prompt to collect user interests."""
    console.print(
        Panel(
            "[bold]Financial Market News Agent[/bold]\n"
            "Configure your interests to get personalized market analysis.",
            title="Setup",
        )
    )

    topics_raw = Prompt.ask(
        "Topics of interest (comma-separated)",
        default="stock market, crypto, commodities, central banks, tech earnings",
    )
    topics = [t.strip() for t in topics_raw.split(",") if t.strip()]

    tickers_raw = Prompt.ask(
        "Tickers to track (comma-separated, or leave empty)",
        default="",
    )
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    regions_raw = Prompt.ask(
        "Regions (comma-separated)",
        default="US, EU",
    )
    regions = [r.strip() for r in regions_raw.split(",") if r.strip()]

    return UserInterests(topics=topics, tickers=tickers, regions=regions)


def render_overview(overview: DailyOverview) -> None:
    """Pretty-print a daily overview to the terminal."""
    console.print()
    console.print(
        Panel(
            f"[bold]{overview.market_mood}[/bold]",
            title="Market Mood",
            border_style="blue",
        )
    )

    for i, digest in enumerate(overview.digests, 1):
        style = URGENCY_STYLES.get(digest.urgency, "white")
        direction = DIRECTION_ICONS.get(digest.impact.direction, digest.impact.direction)

        console.print()
        console.print(f"[bold {style}]{'─' * 60}[/bold {style}]")
        console.print(f"[bold {style}]{i}. {digest.headline}[/bold {style}]")
        console.print(f"   [dim]Urgency:[/dim] [{style}]{digest.urgency.value.upper()}[/{style}]")
        console.print()
        console.print(f"   {digest.summary}")

        if digest.impact.direction:
            console.print()
            console.print(f"   [bold]Market Impact:[/bold] {direction}")
            if digest.impact.affected_assets:
                console.print(
                    f"   [dim]Assets:[/dim] {', '.join(digest.impact.affected_assets)}"
                )
            console.print(f"   [dim]Confidence:[/dim] {digest.impact.confidence}")
            if digest.impact.reasoning:
                console.print(f"   [dim]Reasoning:[/dim] {digest.impact.reasoning}")

        if digest.source_articles:
            console.print()
            console.print("   [dim]Sources:[/dim]")
            for sa in digest.source_articles[:3]:
                console.print(f"     • {sa.title[:80]} [dim]({sa.source_name})[/dim]")
                if sa.url:
                    console.print(f"       [link]{sa.url}[/link]")

    console.print()
    console.print(f"[dim]Generated at {overview.generated_at:%Y-%m-%d %H:%M UTC}[/dim]")


def render_alert(alert: BreakingAlert) -> None:
    """Render a breaking news alert."""
    d = alert.digest
    direction = DIRECTION_ICONS.get(d.impact.direction, d.impact.direction)

    content = Text()
    content.append(f"{d.headline}\n\n", style="bold")
    content.append(f"{d.summary}\n\n")
    content.append(f"Why this matters: {alert.reason}\n\n", style="italic")
    if d.impact.direction:
        content.append(f"Impact: {d.impact.direction.upper()}")
        if d.impact.affected_assets:
            content.append(f" on {', '.join(d.impact.affected_assets)}")
        content.append(f"\n{d.impact.reasoning}")

    console.print()
    console.print(
        Panel(
            content,
            title="BREAKING",
            border_style="bold red",
        )
    )
    for sa in d.source_articles[:3]:
        console.print(f"  [dim]→ {sa.url}[/dim]")


def render_graph_stats(agent: MarketNewsAgent) -> None:
    """Show knowledge graph statistics."""
    stats = agent.graph.stats()
    console.print(
        f"[dim]Knowledge graph: {stats['articles']} articles, "
        f"{stats['entities']} entities, {stats['events']} events, "
        f"{stats['edges']} edges[/dim]"
    )


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------


async def cmd_overview(agent: MarketNewsAgent) -> None:
    """Run the daily overview command."""
    with console.status("[bold blue]Fetching news from all sources…"):
        articles = await agent.fetch_all()

    if not articles:
        console.print("[yellow]No articles found for your interests.[/yellow]")
        return

    console.print(f"[dim]Fetched {len(articles)} articles. Building knowledge graph…[/dim]")
    with console.status("[bold blue]Extracting entities & relationships…"):
        ingested = await agent.ingest_to_graph(articles)
    console.print(f"[dim]Ingested {ingested} articles into graph.[/dim]")
    render_graph_stats(agent)

    with console.status("[bold blue]Generating overview…"):
        overview = await agent.daily_overview()
    render_overview(overview)

    # Offer to ask follow-up questions
    console.print()
    console.print("[bold]You can now ask questions about today's news.[/bold]")
    console.print("[dim]Type 'quit' or 'q' to exit.[/dim]")
    await _qa_loop(agent)


async def cmd_watch(agent: MarketNewsAgent) -> None:
    """Continuous watch mode — poll for breaking news."""
    interval = agent.settings.watch_interval_seconds
    console.print(
        Panel(
            f"Monitoring for breaking news every {interval}s.\n"
            "Press Ctrl+C to stop.",
            title="Watch Mode",
            border_style="green",
        )
    )

    # Do an initial scan with graph ingest
    with console.status("[bold green]Initial scan…"):
        articles = await agent.fetch_all()
        if articles:
            await agent.ingest_to_graph(articles)
        alerts = await agent.check_breaking()
    render_graph_stats(agent)
    if alerts:
        for alert in alerts:
            render_alert(alert)
    else:
        console.print("[dim]No breaking news right now. Watching…[/dim]")

    while True:
        await asyncio.sleep(interval)
        try:
            with console.status("[dim]Checking for new articles…[/dim]"):
                articles = await agent.fetch_all()
                if articles:
                    await agent.ingest_to_graph(articles)
                alerts = await agent.check_breaking()
            if alerts:
                for alert in alerts:
                    render_alert(alert)
            else:
                console.print(
                    f"[dim]{asyncio.get_event_loop().time():.0f}s — no breaking news[/dim]"
                )
        except Exception as exc:
            console.print(f"[red]Error during poll: {exc}[/red]")


async def cmd_ask(agent: MarketNewsAgent) -> None:
    """Interactive Q&A mode — fetch news, build graph, then answer questions."""
    with console.status("[bold blue]Fetching news and building knowledge graph…"):
        articles = await agent.fetch_all()
        if articles:
            ingested = await agent.ingest_to_graph(articles)
            console.print(f"[dim]Ingested {ingested} articles into graph.[/dim]")
    render_graph_stats(agent)
    console.print()
    console.print(
        Panel(
            "[bold]Ask questions about financial markets[/bold]\n"
            "The agent uses a knowledge graph built from recent news to answer.\n"
            "Answers include consensus views and outlier perspectives.\n\n"
            'Examples: "Why did oil prices move today?"\n'
            '          "What is the outlook for tech stocks?"\n'
            '          "How are crypto markets reacting to Fed policy?"',
            title="Q&A Mode",
            border_style="cyan",
        )
    )
    await _qa_loop(agent)


async def _qa_loop(agent: MarketNewsAgent) -> None:
    """Shared interactive Q&A loop."""
    while True:
        console.print()
        try:
            question = Prompt.ask("[bold cyan]Question[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            break

        if question.strip().lower() in ("quit", "q", "exit", ""):
            break

        if question.strip().lower() == "refresh":
            with console.status("[bold blue]Refreshing news and graph…"):
                articles = await agent.fetch_all()
                if articles:
                    ingested = await agent.ingest_to_graph(articles)
                    console.print(f"[dim]Ingested {ingested} new articles.[/dim]")
            render_graph_stats(agent)
            continue

        if question.strip().lower() == "stats":
            render_graph_stats(agent)
            continue

        with console.status("[bold cyan]Thinking…"):
            answer = await agent.ask(question)

        console.print()
        console.print(Panel(Markdown(answer), title="Answer", border_style="cyan"))


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-powered financial market news agent",
    )
    parser.add_argument(
        "mode",
        choices=["overview", "watch", "ask"],
        nargs="?",
        default="overview",
        help=(
            "'overview' for daily summary (then Q&A), "
            "'watch' for continuous breaking-news alerts, "
            "'ask' for interactive Q&A"
        ),
    )
    parser.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Comma-separated topics (skip interactive prompt)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers (skip interactive prompt)",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use default interests without prompting",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = load_settings()
    if not settings.has_any_news_source:
        console.print(
            "[red]No news API keys configured.[/red]\n"
            "Copy .env.example to .env and add at least one API key.\n"
            "See README.md for details."
        )
        sys.exit(1)

    if not settings.hf_token and settings.llm_backend == "api":
        console.print(
            "[red]HF_TOKEN not set.[/red]\n"
            "Get a free token at https://huggingface.co/settings/tokens"
        )
        sys.exit(1)

    # Collect or build interests
    if args.topics or args.non_interactive:
        topics = (
            [t.strip() for t in args.topics.split(",")]
            if args.topics
            else ["stock market", "crypto", "commodities", "central banks"]
        )
        tickers = (
            [t.strip().upper() for t in args.tickers.split(",")]
            if args.tickers
            else []
        )
        interests = UserInterests(topics=topics, tickers=tickers)
    else:
        interests = collect_interests()

    console.print()
    console.print(f"[dim]Topics:  {', '.join(interests.topics)}[/dim]")
    console.print(f"[dim]Tickers: {', '.join(interests.tickers) or '(none)'}[/dim]")
    console.print(f"[dim]Model:   {settings.hf_model} ({settings.llm_backend})[/dim]")

    agent = MarketNewsAgent(settings, interests)
    console.print(f"[dim]Sources: {', '.join(s.name for s in agent.sources)}[/dim]")

    try:
        if args.mode == "watch":
            asyncio.run(cmd_watch(agent))
        elif args.mode == "ask":
            asyncio.run(cmd_ask(agent))
        else:
            asyncio.run(cmd_overview(agent))
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


if __name__ == "__main__":
    main()
