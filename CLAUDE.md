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
5. **Memory cycle** — LSTM-style relevance management (see below) replaces blunt TTL pruning

**Graph schema:**
```sql
articles  (id, url, title, summary, source_name, published_at, tickers, sentiment,
           created_at, relevance, last_referenced, reference_count)
entities  (id, name, entity_type, aliases, created_at,
           relevance, last_referenced, reference_count)  -- UNIQUE(name, entity_type)
events    (id, description, event_type, timestamp, created_at,
           relevance, last_referenced, reference_count)
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
  │     ├── graph.py (GraphStore — SQLite + LSTM memory)
  │     ├── extraction.py (entity/relationship extraction)
  │     ├── qa.py (Q&A pipeline + ReAct loop)
  │     │     └── tools.py (live data lookup)
  │     └── sources/
  │           ├── rss.py (Reuters, AP, CNBC, BBC, NPR, Guardian)
  │           ├── reddit.py (r/economics, r/wallstreetbets, etc.)
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

## LSTM-Style Memory Management

The knowledge graph uses a memory model inspired by LSTM gates to decide what to keep
and what to forget. Every node (article, entity, event) carries a `relevance` score
between 0.0 and 1.0. The system runs a **memory cycle** on each ingest:

```
┌─────────────────────────────────────────────────────────┐
│  FORGET GATE (Decay)                                     │
│  relevance *= 0.85 per cycle — everything slowly fades   │
├─────────────────────────────────────────────────────────┤
│  INPUT GATE (Boost on re-reference)                      │
│  relevance += 0.3 when article/entity seen again         │
│  relevance += 0.1 when user queries about it             │
├─────────────────────────────────────────────────────────┤
│  STRUCTURAL GATE (Connectivity bonus)                    │
│  relevance += 0.02 * log(1 + edge_count)                 │
│  Highly-connected nodes (e.g. "Fed") resist decay        │
├─────────────────────────────────────────────────────────┤
│  OUTPUT GATE (Prune)                                     │
│  Remove nodes with relevance < 0.05                      │
│  Also removes all orphaned edges                         │
└─────────────────────────────────────────────────────────┘
```

**Hyperparameters** (in `graph.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DECAY_FACTOR` | 0.85 | Per-cycle multiplicative decay |
| `BOOST_ON_REFERENCE` | 0.3 | Relevance added when node is re-referenced by new articles |
| `CONNECTIVITY_BONUS` | 0.02 | Per-edge bonus (log-scaled, diminishing returns) |
| `PRUNE_THRESHOLD` | 0.05 | Nodes below this are removed |
| `INITIAL_RELEVANCE` | 1.0 | New nodes start at full relevance |

**What this means in practice:**

- A one-off article about a minor earnings beat decays to zero in ~18 cycles and gets pruned
- "The Fed" entity keeps getting re-referenced by new articles → stays near 1.0
- A rate decision event persists as long as downstream articles keep linking to it
- Old articles about resolved topics (e.g. a passed election) quietly disappear
- When a user asks about "oil", all oil-related entities get a +0.1 boost — user interest signals relevance
- All graph queries filter by `relevance >= 0.05` and sort by `relevance DESC` — more relevant data surfaces first

**Migration:** If upgrading from the old schema, `_migrate_add_memory_columns()` adds `relevance`, `last_referenced`, and `reference_count` to all node tables. The old `prune_older_than_days()` method still exists but now delegates to `memory_cycle()`.

## Data Freshness Strategy

- **News articles**: Fetched on-demand from APIs (real-time)
- **Knowledge graph**: Persisted in SQLite, LSTM-style memory management (no fixed TTL)
- **Live data tools**: Called during Q&A (real-time via API)
- **Watch mode**: Polls every 5 minutes (configurable)
- **Graph rebuild**: `refresh` command in Q&A mode re-fetches and re-ingests
- **Memory cycle**: Runs automatically during each ingest — decay, boost, prune

## News Sources

| Source | Auth | Implementation |
|--------|------|---------------|
| **RSS** (Reuters, AP, CNBC, BBC, NPR, Guardian) | None | `sources/rss.py` — stdlib `xml.etree` parser, no dependencies |
| **Reddit** (r/economics, r/wallstreetbets, r/geopolitics) | None | `sources/reddit.py` — public JSON API (`/r/{sub}/hot.json`) |
| **NewsAPI.org** | API key | `sources/newsapi.py` — `/v2/everything` endpoint |
| **Finnhub** | API key | `sources/finnhub.py` — `/news` + `/company-news` |
| **Alpha Vantage** | API key | `sources/alphavantage.py` — `NEWS_SENTIMENT` function |

RSS and Reddit are enabled by default (`ENABLE_RSS=true`, `ENABLE_REDDIT=true`).
They provide full macro/political/financial coverage with no setup.

RSS feeds are topic-mapped: user topics like "oil", "politics", "inflation" resolve to
specific feed categories (commodities_energy, politics_elections, central_banks_inflation).

Reddit has a minimum quality filter (score >= 10 upvotes) to cut noise.

## Adding a New News Source

1. Create `src/news_agent/sources/newsource.py`
2. Implement `NewsSource` ABC (one method: `async fetch(query, max_results) -> list[Article]`)
3. For keyed sources: add API key to `config.py` Settings class + `.env.example`
4. Wire up in `agent.py → _build_sources()`

## Adding a New Tool

1. Add implementation function in `tools.py` (async, returns string)
2. Add to `execute_tool()` dispatch
3. Add description to `_TOOL_DESCRIPTIONS` string
4. Tools are automatically available to the ReAct loop — no other changes needed

## Roadmap / Key Differentiators

These features define why this tool exists vs. a generic ChatGPT session:

1. **Portfolio tracking** — User adds their stocks and interests. The agent tracks those positions across news cycles, flags relevant moves, and maintains a persistent portfolio context.
2. **Source quality rating** — Rate and weight sources by accuracy/reliability over time. Sources that consistently provide actionable, accurate news rank higher in the analysis.
3. **LSTM-gated trend tracking** — The memory system (above) ensures long-running trends persist while noise decays. Unlike a chat history that resets, the knowledge graph carries forward structural understanding of the market.
4. **Automated delivery + alerts** — Daily briefing delivery (scheduled, not on-demand) plus real-time breaking alerts. The user doesn't need to ask — the tool pushes relevant information when it matters.
