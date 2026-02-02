# LLM Financial Market News Agent

A Python CLI tool that fetches financial news from multiple sources, then uses a Hugging Face 3B language model to produce:

- **Daily Overview** — clustered, summarized market news with expected impact analysis
- **Watch Mode** — continuous monitoring with breaking-news alerts when immediate attention is needed

## Architecture

```
User Interests ──► News Fetcher (3 APIs) ──► Dedup & Sort ──► LLM Agent ──► Formatted Output
                     │                                          │
                     ├─ NewsAPI.org (80k+ sources)              ├─ Daily Overview
                     ├─ Finnhub (financial-specific)            └─ Breaking Alerts
                     └─ Alpha Vantage (sentiment scores)
```

The LLM (default: `meta-llama/Llama-3.2-3B-Instruct`) clusters related articles, writes summaries, and assesses market impact (direction, affected assets, confidence level).

## Quick Start

### 1. Install

```bash
# Clone and install
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
# Interactive — prompts for your interests, then shows daily overview
python main.py

# Daily overview with CLI args (non-interactive)
python main.py overview --topics "tech earnings, crypto, fed policy" --tickers "AAPL,NVDA,BTC"

# Watch mode — polls every 5 min, alerts on breaking news
python main.py watch --topics "stock market, central banks" --tickers "SPY,QQQ"

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

Fetches all recent articles, groups them into topic clusters, and produces a structured analysis:

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

### Watch Mode

Continuously polls news sources and uses the LLM to identify truly breaking events:

```
┌──────────────────────────────────────────────────────────┐
│ 🚨 BREAKING                                              │
│ Fed Emergency Rate Cut — 50bps                           │
│                                                          │
│ The Federal Reserve announced an emergency 50 basis      │
│ point rate cut citing deteriorating economic conditions…  │
│                                                          │
│ Why: Unscheduled rate cut signals severe economic concern │
│ Impact: BEARISH on USD, BULLISH on equities/gold         │
└──────────────────────────────────────────────────────────┘
```

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
├── main.py                         # CLI entry point
├── src/news_agent/
│   ├── agent.py                    # Core orchestrator (fetch → LLM → output)
│   ├── llm.py                      # HF Inference API / local model wrapper
│   ├── models.py                   # Pydantic data models
│   ├── config.py                   # Settings from .env
│   └── sources/
│       ├── base.py                 # Abstract news source interface
│       ├── newsapi.py              # NewsAPI.org adapter
│       ├── finnhub.py              # Finnhub adapter
│       └── alphavantage.py         # Alpha Vantage adapter
├── pyproject.toml
├── .env.example
└── README.md
```

## Next Steps (Beyond POC)

- **Persistent storage** — SQLite/DuckDB to track articles over time and avoid re-processing
- **Webhook/notification output** — Slack, Telegram, email alerts for breaking news
- **Larger models** — swap in 7B/13B models or API-based models (Claude, GPT-4) for better analysis
- **Historical backtesting** — compare predicted impact vs actual market moves
- **Web dashboard** — FastAPI + HTMX for a lightweight browser UI
