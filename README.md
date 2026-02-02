# LLM Financial Market News Agent

A Python CLI tool that fetches financial news from multiple sources, builds a **knowledge graph**, then uses a Hugging Face 3B language model to:

- **Daily Overview** — clustered, summarized market news with expected impact analysis
- **Watch Mode** — continuous monitoring with breaking-news alerts when immediate attention is needed
- **Interactive Q&A** — ask questions like "Why did oil move today?" and get answers with consensus + outlier views, powered by Graph RAG

## Architecture

```
User Interests ──► News Fetcher (3 APIs) ──► Dedup ──► Entity Extraction ──► Knowledge Graph (SQLite)
                     │                                       │                        │
                     ├─ NewsAPI.org                          │                        ├─ Daily Overview
                     ├─ Finnhub                         LLM (3B)                     ├─ Breaking Alerts
                     └─ Alpha Vantage                        │                        └─ Q&A (Graph RAG)
                                                             ▼
                                                      Entities, Events,
                                                      Relationships
```

### Graph RAG for Q&A

The agent builds a knowledge graph from news articles with:

- **Nodes**: Articles, Entities (tickers, commodities, institutions, indices, currencies), Events (rate decisions, earnings, geopolitical)
- **Edges**: MENTIONS, REPORTS_ON, AFFECTS (with bullish/bearish direction), RELATED_TO, CAUSED_BY

When you ask a question, the pipeline:
1. Extracts entities from your question
2. Traverses the graph to find related articles, events, and causal chains
3. Assembles a compact context from the subgraph
4. Generates an answer with **consensus opinion** + **outlier/contrarian views**

The graph is stored in SQLite (`news_graph.db`) and automatically prunes data older than 7 days.

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

| Service | Free Tier | Sign Up |
|---------|-----------|---------|
| NewsAPI.org | 100 req/day | https://newsapi.org/register |
| Finnhub | 60 req/min | https://finnhub.io/register |
| Alpha Vantage | 25 req/day | https://www.alphavantage.co/support/#api-key |
| Hugging Face | Free inference API | https://huggingface.co/settings/tokens |

### 3. Run

```bash
# Interactive — prompts for your interests, then shows daily overview + Q&A
python main.py

# Daily overview with CLI args (non-interactive)
python main.py overview --topics "tech earnings, crypto, fed policy" --tickers "AAPL,NVDA,BTC"

# Watch mode — polls every 5 min, alerts on breaking news
python main.py watch --topics "stock market, central banks" --tickers "SPY,QQQ"

# Q&A mode — fetch news, build graph, then ask questions interactively
python main.py ask --topics "commodities, energy, central banks"

# Use defaults without prompting
python main.py overview --non-interactive
```

## Configuration

All config is via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `NEWSAPI_KEY` | — | NewsAPI.org API key |
| `FINNHUB_KEY` | — | Finnhub API key |
| `ALPHAVANTAGE_KEY` | — | Alpha Vantage API key |
| `HF_TOKEN` | — | Hugging Face access token |
| `HF_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | Model ID on HF Hub |
| `LLM_BACKEND` | `api` | `api` (HF Inference API) or `local` (on-device) |
| `WATCH_INTERVAL_SECONDS` | `300` | Polling interval for watch mode |

## Modes

### Daily Overview

Fetches news, builds the knowledge graph, then clusters and summarizes:

```
┌─────────────────────────────────────────────────────────┐
│ Market Mood                                             │
│ Risk-on sentiment dominates as tech earnings beat       │
│ expectations; Fed signals holding rates steady.         │
└─────────────────────────────────────────────────────────┘

1. Tech Earnings Beat Expectations                [NORMAL]
   NVIDIA and Microsoft reported Q4 earnings above
   consensus, driving Nasdaq futures higher…

   Market Impact: ▲ Bullish
   Assets: QQQ, NVDA, MSFT
   Confidence: high

2. Fed Holds Rates, Signals Patience              [HIGH]
   …
```

After the overview, you can ask follow-up questions interactively.

### Watch Mode

Continuously polls news sources, ingests into the graph, and alerts on breaking events:

```
┌──────────────────────────────────────────────────────────┐
│ BREAKING                                                 │
│ Fed Emergency Rate Cut — 50bps                           │
│                                                          │
│ The Federal Reserve announced an emergency 50 basis      │
│ point rate cut citing deteriorating economic conditions…  │
│                                                          │
│ Why: Unscheduled rate cut signals severe economic concern │
│ Impact: BEARISH on USD, BULLISH on equities/gold         │
└──────────────────────────────────────────────────────────┘
```

### Q&A Mode

Interactive question answering backed by the knowledge graph:

```
┌──────────────────────────────────────────────────────────┐
│ Q&A Mode                                                 │
│ Ask questions about financial markets                    │
└──────────────────────────────────────────────────────────┘

Question: Why did oil prices move today?

┌──────────────────────────────────────────────────────────┐
│ Answer                                                   │
│                                                          │
│ Consensus View: Oil prices rose 3.2% driven by OPEC+    │
│ announcing deeper production cuts and escalating         │
│ geopolitical tensions in the Middle East.                │
│                                                          │
│ Outlier Views: Goldman Sachs maintains that the rally    │
│ is temporary, citing weakening Chinese demand data that  │
│ most analysts are overlooking.                           │
│                                                          │
│ Key Drivers:                                             │
│ • OPEC+ production cut of 1.2M bbl/day                  │
│ • Middle East shipping route disruptions                 │
│ • USD weakening after dovish Fed comments                │
│                                                          │
│ Watch List:                                              │
│ • EIA inventory report (Wednesday)                       │
│ • OPEC monthly report (Thursday)                         │
└──────────────────────────────────────────────────────────┘
```

Special commands in Q&A: `refresh` (re-fetch news), `stats` (graph stats), `quit` (exit).

## Using a Local Model

If you have a GPU and want to run the model locally instead of using the HF Inference API:

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
├── main.py                         # CLI entry point (overview, watch, ask modes)
├── src/news_agent/
│   ├── agent.py                    # Core orchestrator (fetch → graph → LLM → output)
│   ├── llm.py                      # HF Inference API / local model wrapper
│   ├── models.py                   # Pydantic data models (articles, entities, events)
│   ├── config.py                   # Settings from .env
│   ├── graph.py                    # SQLite-backed knowledge graph store
│   ├── extraction.py               # LLM-based entity/relationship extraction
│   ├── qa.py                       # Q&A pipeline (Graph RAG)
│   └── sources/
│       ├── base.py                 # Abstract news source interface
│       ├── newsapi.py              # NewsAPI.org adapter
│       ├── finnhub.py              # Finnhub adapter
│       └── alphavantage.py         # Alpha Vantage adapter
├── news_graph.db                   # Auto-created SQLite knowledge graph
├── pyproject.toml
├── .env.example
└── README.md
```

## Next Steps (Beyond POC)

- **Webhook/notification output** — Slack, Telegram, email alerts for breaking news
- **Larger models** — swap in 7B/13B models or API-based models (Claude, GPT-4) for deeper analysis quality
- **Historical backtesting** — compare predicted impact vs actual market moves
- **Web dashboard** — FastAPI + HTMX for a lightweight browser UI
- **Vector hybrid** — add embedding-based search alongside graph traversal for better recall
- **Scheduled runs** — cron/systemd integration for automated daily reports
