# LLM Financial Market News Agent

A Python CLI tool that fetches financial news from multiple sources, builds a **knowledge graph**, and uses a Hugging Face 3B language model with **live data tools** to produce market analysis.

- **Daily Overview** — clustered, summarized market news with expected impact analysis
- **Watch Mode** — continuous monitoring with breaking-news alerts
- **Interactive Q&A** — ask "Why did oil move today?" and get answers grounded in news + live data, with consensus and outlier views

## Three-Phase Pipeline

Every interaction follows three distinct phases:

```
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1: DATA                                                   │
│  Fetch news from 3 APIs → deduplicate → extract entities/events  │
│  → build knowledge graph (SQLite)                                │
│                                                                  │
│  PHASE 2: AGENTIC AI                                             │
│  Parse question → traverse graph for context → ReAct tool loop   │
│  (LLM can call stock_price, economic_indicator, etc. for live    │
│  data) → reason across news + data                               │
│                                                                  │
│  PHASE 3: REPORT                                                 │
│  Structured output: consensus view + outlier/contrarian views    │
│  + key drivers + watch list                                      │
└──────────────────────────────────────────────────────────────────┘
```

**Phase 1** runs once at startup (or on `refresh`). **Phase 2 + 3** run for each question.

## Quick Start

### 1. Install

```bash
git clone <repo-url> && cd llm-news-agent
pip install -e .

# Or with local model support (requires GPU):
pip install -e ".[local]"
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your keys
```

You need **at least one** news source key and a **Hugging Face token**:

| Service | Free Tier | What it provides | Sign Up |
|---------|-----------|-----------------|---------|
| NewsAPI.org | 100 req/day | Broad news from 80k+ sources | https://newsapi.org/register |
| Finnhub | 60 req/min | Financial news + stock/crypto quotes | https://finnhub.io/register |
| Alpha Vantage | 25 req/day | News sentiment + quotes + forex | https://www.alphavantage.co/support/#api-key |
| FRED | Unlimited | Economic indicators (CPI, GDP, rates) | https://fred.stlouisfed.org/docs/api/api_key.html |
| Hugging Face | Free inference API | LLM (Llama-3.2-3B) | https://huggingface.co/settings/tokens |

### 3. Run

```bash
# Interactive — prompts for your interests, then shows daily overview + Q&A
python main.py

# Daily overview with CLI args (non-interactive)
python main.py overview --topics "tech earnings, crypto, fed policy" --tickers "AAPL,NVDA,BTC"

# Watch mode — polls every 5 min, alerts on breaking news
python main.py watch --topics "stock market, central banks" --tickers "SPY,QQQ"

# Q&A mode — fetch news, build graph, then ask questions with live data tools
python main.py ask --topics "commodities, energy, central banks"

# Use defaults without prompting
python main.py overview --non-interactive
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NEWSAPI_KEY` | — | NewsAPI.org API key |
| `FINNHUB_KEY` | — | Finnhub API key |
| `ALPHAVANTAGE_KEY` | — | Alpha Vantage API key |
| `FRED_KEY` | — | FRED API key (for economic indicators) |
| `HF_TOKEN` | — | Hugging Face access token |
| `HF_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | Model ID on HF Hub |
| `LLM_BACKEND` | `api` | `api` (HF Inference API) or `local` (on-device) |
| `WATCH_INTERVAL_SECONDS` | `300` | Polling interval for watch mode |

## Live Data Tools

During Q&A, the LLM can autonomously call these tools to fetch real-time data:

| Tool | Example | What it returns |
|------|---------|----------------|
| `stock_price` | `stock_price(AAPL)` | Price, open, high, low, change% |
| `crypto_price` | `crypto_price(BTC)` | Price in USD, change% |
| `forex_rate` | `forex_rate(EUR, USD)` | Exchange rate |
| `economic_indicator` | `economic_indicator(CPIAUCSL)` | Latest values + series info |
| `market_index` | `market_index(SPY)` | Index level, change% |
| `commodity_price` | `commodity_price(GOLD)` | Commodity price via ETF proxy |
| `company_profile` | `company_profile(NVDA)` | Sector, market cap, exchange |

Common FRED series: `CPIAUCSL` (CPI), `FEDFUNDS` (Fed rate), `UNRATE` (unemployment), `GDP`, `DGS10` (10Y Treasury), `PCEPILFE` (Core PCE).

## Modes

### Daily Overview

Fetches news, builds the knowledge graph, then clusters and summarizes:

```
Phase 1/3 — DATA   Fetching news & building knowledge graph
  Fetched 47 articles from 3 sources
  Ingested 47 articles into knowledge graph

┌─────────────────────────────────────────────────────────┐
│ Market Mood                                             │
│ Risk-on sentiment dominates as tech earnings beat       │
│ expectations; Fed signals holding rates steady.         │
└─────────────────────────────────────────────────────────┘

1. Tech Earnings Beat Expectations                [NORMAL]
   Market Impact: ▲ Bullish | Assets: QQQ, NVDA, MSFT

2. Fed Holds Rates, Signals Patience              [HIGH]
   …

You can now ask questions about today's news.
```

### Watch Mode

Polls news sources, ingests into the graph, and alerts on breaking events.

### Q&A Mode

Three-phase interactive Q&A with tool use:

```
Phase 1/3 — DATA   Fetching news & building knowledge graph
  Fetched 47 articles from 3 sources

Question: What is the current inflation rate and how is it affecting markets?

Phase 2/3 — AGENT  Reasoning & fetching live data…
  → TOOL: economic_indicator(CPIAUCSL)    ← agent decides it needs CPI data
  → TOOL: stock_price(SPY)                ← then checks market level
Phase 3/3 — REPORT

┌──────────────────────────────────────────────────────────┐
│ Answer                                                   │
│                                                          │
│ Consensus View: CPI came in at 3.1% year-over-year,     │
│ slightly below the 3.2% consensus. Markets interpreted   │
│ this as confirmation that the Fed can begin cutting…     │
│                                                          │
│ Outlier Views: BofA argues the decline is driven by      │
│ base effects and core services inflation remains sticky…  │
│                                                          │
│ Key Drivers: …                                           │
│ Watch List: …                                            │
└──────────────────────────────────────────────────────────┘
```

Commands in Q&A: `refresh` (re-fetch news), `stats` (graph info), `quit` (exit).

## Knowledge Graph

The agent builds a graph from news articles:

- **Nodes**: Articles, Entities (tickers, commodities, institutions, indices, currencies), Events
- **Edges**: MENTIONS, REPORTS_ON, AFFECTS (with bullish/bearish), RELATED_TO, CAUSED_BY

This enables causal reasoning: "OPEC cut → oil supply down → oil price up → energy stocks up" rather than just keyword matching.

Stored in SQLite (`news_graph.db`), auto-prunes data older than 7 days.

## Using a Local Model

```bash
pip install -e ".[local]"
# In .env:
LLM_BACKEND=local
HF_MODEL=meta-llama/Llama-3.2-3B-Instruct
```

Other compatible 3B models: `microsoft/Phi-3.5-mini-instruct`, `Qwen/Qwen2.5-3B-Instruct`.

## Project Structure

```
llm-news-agent/
├── main.py                         # CLI entry point (overview, watch, ask)
├── src/news_agent/
│   ├── agent.py                    # Core orchestrator (three-phase pipeline)
│   ├── llm.py                      # HF Inference API / local model wrapper
│   ├── models.py                   # Pydantic data models
│   ├── config.py                   # Settings from .env
│   ├── graph.py                    # SQLite-backed knowledge graph
│   ├── extraction.py               # LLM entity/relationship extraction
│   ├── qa.py                       # Q&A pipeline + ReAct tool loop
│   ├── tools.py                    # Live data lookup tools
│   └── sources/
│       ├── base.py                 # Abstract news source interface
│       ├── newsapi.py              # NewsAPI.org adapter
│       ├── finnhub.py              # Finnhub adapter
│       └── alphavantage.py         # Alpha Vantage adapter
├── news_graph.db                   # Auto-created knowledge graph
├── pyproject.toml
├── .env.example
├── CLAUDE.md                       # Technical architecture docs
└── README.md
```

## Next Steps (Beyond POC)

- **Webhook output** — Slack, Telegram, email alerts for breaking news
- **Larger models** — 7B/13B or API models (Claude, GPT-4) for deeper analysis
- **Historical backtesting** — compare predicted vs actual market impact
- **Web dashboard** — FastAPI + HTMX browser UI
- **Vector hybrid** — embeddings alongside graph traversal for better recall
- **Scheduled runs** — cron/systemd for automated daily reports
- **More sources** — RSS feeds, Reddit sentiment, SEC EDGAR filings
