# Technical Architecture — LLM Financial News Agent

## Three-Phase Pipeline

Every interaction follows a strict three-phase pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: DATA                                                          │
│  Fetch → Deduplicate → Extract entities/events → Ingest into graph      │
├─────────────────────────────────────────────────────────────────────────┤
│  PHASE 2: AGENTIC AI                                                    │
│  Parse question → Traverse graph → ReAct tool loop (live data) → Reason │
├─────────────────────────────────────────────────────────────────────────┤
│  PHASE 3: REPORT                                                        │
│  Structure answer → Consensus + Outliers → Key drivers → Watch list     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Data Collection & Knowledge Graph

**Entry point:** `agent.py → fetch_all()` + `ingest_to_graph()`

1. **Parallel fetch** from all configured news APIs (NewsAPI, Finnhub, Alpha Vantage)
2. **Deduplication** by URL across sources
3. **Entity extraction** (LLM call per article) → entities (tickers, commodities, institutions), events (rate decisions, earnings), relationships (AFFECTS, CAUSED_BY)
4. **Graph ingest** into SQLite — articles, entities, events stored as nodes; relationships as typed edges
5. **Pruning** — data older than 7 days automatically removed

**Graph schema:**
```sql
articles  (id, url, title, summary, source_name, published_at, tickers, sentiment, created_at)
entities  (id, name, entity_type, aliases, created_at)  -- UNIQUE(name, entity_type)
events    (id, description, event_type, timestamp, created_at)
edges     (id, source_type, source_id, target_type, target_id, relation, direction, weight, created_at)
```

**Edge types:**
- `MENTIONS` — article → entity (auto-created during ingest)
- `REPORTS_ON` — article → event (auto-created during ingest)
- `AFFECTS` — event → entity (with direction: bullish/bearish/neutral)
- `RELATED_TO` — entity → entity (structural relationships)
- `CAUSED_BY` — event → event (causal chains)

### Phase 2: Agentic AI (ReAct Loop)

**Entry point:** `qa.py → answer_question()` → `_react_loop()`

The Q&A pipeline:

1. **Entity extraction from question** — LLM identifies mentioned assets/topics
2. **Graph traversal** — finds related articles, events, causal chains via SQL joins
3. **Context assembly** — compacts subgraph into text for the LLM prompt
4. **ReAct tool loop** (up to 4 rounds):
   - LLM reads graph context + tool descriptions
   - LLM decides if it needs live data → outputs `TOOL: tool_name(args)`
   - Tool executes (HTTP call to Finnhub/Alpha Vantage/FRED)
   - Result appended to prompt → LLM continues reasoning
   - Repeat until LLM outputs `ANSWER:` or max rounds reached

**Available tools** (defined in `tools.py`):

| Tool | Data Source | What it returns |
|------|-----------|----------------|
| `stock_price(TICKER)` | Finnhub → Alpha Vantage | Price, open, high, low, change% |
| `crypto_price(SYMBOL)` | Finnhub (Binance) → Alpha Vantage | Price in USD, change% |
| `forex_rate(FROM, TO)` | Alpha Vantage → Finnhub | Exchange rate |
| `economic_indicator(SERIES)` | FRED API | Latest 5 data points + series metadata |
| `market_index(SYMBOL)` | Finnhub (via ETF proxy) | Index level, change% |
| `commodity_price(SYMBOL)` | Finnhub (via ETF proxy) | Commodity price, change% |
| `company_profile(TICKER)` | Finnhub | Sector, market cap, exchange |

**Tool call parsing:** Regex `TOOL:\s*(\w+)\(([^)]*)\)` — simple and reliable with 3B models.

**Fallback chain:** Each tool tries Finnhub first, then Alpha Vantage. Economic indicators require FRED key.

### Phase 3: Report Generation

The LLM prompt enforces a structured output format:

1. **Consensus View** — what majority of sources/data agree on, with citations
2. **Outlier / Contrarian Views** — minority opinions, contradictory data points
3. **Key Drivers** — 2-4 causal factors
4. **Watch List** — upcoming events/data that could change the picture

For daily overview mode, the report uses JSON output parsed into `DailyOverview` → `NewsDigest` objects with urgency levels and market impact assessments.

## Module Dependency Graph

```
main.py
  ├── agent.py (MarketNewsAgent)
  │     ├── config.py (Settings)
  │     ├── llm.py (LLM — HF Inference API / local transformers)
  │     ├── graph.py (GraphStore — SQLite)
  │     ├── extraction.py (entity/relationship extraction)
  │     ├── qa.py (Q&A pipeline + ReAct loop)
  │     │     └── tools.py (live data lookup)
  │     └── sources/
  │           ├── newsapi.py
  │           ├── finnhub.py
  │           └── alphavantage.py
  └── models.py (Pydantic data models)
```

## LLM Considerations for 3B Models

3B parameter models (Llama-3.2-3B, Phi-3.5-mini) have specific constraints:

1. **JSON output** — sometimes malformed. `llm.py:extract_json()` does best-effort parsing with depth-tracking brace matching and markdown fence stripping.
2. **Context window** — typically 4-8K tokens. Prompts are kept compact: max 40 articles in overview, max 25 articles in Q&A context, max 4 tool rounds.
3. **Tool calling** — no native function calling. Uses text-based ReAct pattern (`TOOL: name(args)` / `TOOL RESULT:` / `ANSWER:`).
4. **Instruction following** — simpler prompts with explicit JSON templates work better than complex multi-step instructions.

## Data Freshness Strategy

- **News articles**: Fetched on-demand from APIs (real-time)
- **Knowledge graph**: Persisted in SQLite, auto-pruned at 7 days
- **Live data tools**: Called during Q&A (real-time via API)
- **Watch mode**: Polls every 5 minutes (configurable)
- **Graph rebuild**: `refresh` command in Q&A mode re-fetches and re-ingests

## Adding a New News Source

1. Create `src/news_agent/sources/newsource.py`
2. Implement `NewsSource` ABC (one method: `async fetch(query, max_results) -> list[Article]`)
3. Add API key to `config.py` Settings class
4. Wire up in `agent.py → _build_sources()`
5. Add key to `.env.example`

## Adding a New Tool

1. Add implementation function in `tools.py` (async, returns string)
2. Add to `execute_tool()` dispatch
3. Add description to `_TOOL_DESCRIPTIONS` string
4. Tools are automatically available to the ReAct loop — no other changes needed
