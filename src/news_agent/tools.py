"""Data-lookup tools the LLM can invoke during Q&A.

Each tool is a simple async function that fetches live data from free APIs
(Finnhub, Alpha Vantage, FRED) and returns a formatted string.

The ReAct loop in qa.py parses tool calls from LLM output and executes them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import httpx

from news_agent.config import Settings

logger = logging.getLogger(__name__)

# Timeout for all HTTP calls
_TIMEOUT = 15


@dataclass
class ToolResult:
    name: str
    args: str
    output: str
    success: bool


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_TOOL_DESCRIPTIONS = """\
Available tools (use EXACTLY this syntax to call one):

TOOL: stock_price(TICKER)
  Get the latest stock quote. Example: TOOL: stock_price(AAPL)

TOOL: crypto_price(SYMBOL)
  Get the latest crypto price in USD. Example: TOOL: crypto_price(BTC)

TOOL: forex_rate(FROM, TO)
  Get the latest exchange rate. Example: TOOL: forex_rate(EUR, USD)

TOOL: economic_indicator(SERIES_ID)
  Get the latest value of a FRED economic indicator.
  Common series: CPI (CPIAUCSL), Fed Funds Rate (FEDFUNDS), Unemployment (UNRATE),
  GDP (GDP), 10Y Treasury (DGS10), Core PCE (PCEPILFE), M2 Money Supply (M2SL).
  Example: TOOL: economic_indicator(CPIAUCSL)

TOOL: market_index(SYMBOL)
  Get the latest index/ETF price. Example: TOOL: market_index(SPY)

TOOL: commodity_price(SYMBOL)
  Get commodity price via ETF proxy. Common: GLD (gold), USO (oil), SLV (silver), UNG (nat gas).
  Example: TOOL: commodity_price(GLD)

TOOL: company_profile(TICKER)
  Get basic company info (sector, market cap, 52-week range). Example: TOOL: company_profile(NVDA)"""


def get_tool_descriptions() -> str:
    return _TOOL_DESCRIPTIONS


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def execute_tool(name: str, args: str, settings: Settings) -> ToolResult:
    """Dispatch and execute a tool call. Returns a ToolResult."""
    args = args.strip()
    try:
        if name == "stock_price":
            output = await _stock_price(args, settings)
        elif name == "crypto_price":
            output = await _crypto_price(args, settings)
        elif name == "forex_rate":
            parts = [a.strip() for a in args.split(",")]
            if len(parts) != 2:
                return ToolResult(name, args, "Error: forex_rate requires two arguments: FROM, TO", False)
            output = await _forex_rate(parts[0], parts[1], settings)
        elif name == "economic_indicator":
            output = await _economic_indicator(args, settings)
        elif name == "market_index":
            output = await _market_index(args, settings)
        elif name == "commodity_price":
            output = await _commodity_price(args, settings)
        elif name == "company_profile":
            output = await _company_profile(args, settings)
        else:
            return ToolResult(name, args, f"Error: Unknown tool '{name}'", False)
        return ToolResult(name, args, output, True)
    except Exception as exc:
        logger.warning("Tool %s(%s) failed: %s", name, args, exc)
        return ToolResult(name, args, f"Error: {exc}", False)


# ---------------------------------------------------------------------------
# Stock price — via Finnhub (primary) or Alpha Vantage (fallback)
# ---------------------------------------------------------------------------

async def _stock_price(ticker: str, settings: Settings) -> str:
    ticker = ticker.upper().strip()
    # Try Finnhub first
    if settings.finnhub_key:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": ticker, "token": settings.finnhub_key},
            )
            resp.raise_for_status()
            d = resp.json()
        if d.get("c") and d["c"] > 0:
            return (
                f"{ticker}: ${d['c']:.2f} (open=${d['o']:.2f}, high=${d['h']:.2f}, "
                f"low=${d['l']:.2f}, prev_close=${d['pc']:.2f}, "
                f"change={d['dp']:.2f}%)"
            )

    # Fallback to Alpha Vantage
    if settings.alphavantage_key:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "GLOBAL_QUOTE",
                    "symbol": ticker,
                    "apikey": settings.alphavantage_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        gq = data.get("Global Quote", {})
        if gq.get("05. price"):
            price = float(gq["05. price"])
            change_pct = gq.get("10. change percent", "N/A")
            return f"{ticker}: ${price:.2f} (change: {change_pct})"

    return f"No stock data available for {ticker} (no Finnhub or Alpha Vantage key configured)"


# ---------------------------------------------------------------------------
# Crypto price — via Finnhub
# ---------------------------------------------------------------------------

async def _crypto_price(symbol: str, settings: Settings) -> str:
    symbol = symbol.upper().strip()
    # Finnhub uses BINANCE:BTCUSDT format
    exchange_symbol = f"BINANCE:{symbol}USDT"

    if settings.finnhub_key:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": exchange_symbol, "token": settings.finnhub_key},
            )
            resp.raise_for_status()
            d = resp.json()
        if d.get("c") and d["c"] > 0:
            return (
                f"{symbol}/USD: ${d['c']:,.2f} (open=${d['o']:,.2f}, high=${d['h']:,.2f}, "
                f"low=${d['l']:,.2f}, change={d['dp']:.2f}%)"
            )

    # Fallback: Alpha Vantage crypto
    if settings.alphavantage_key:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "CURRENCY_EXCHANGE_RATE",
                    "from_currency": symbol,
                    "to_currency": "USD",
                    "apikey": settings.alphavantage_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        rate_data = data.get("Realtime Currency Exchange Rate", {})
        if rate_data.get("5. Exchange Rate"):
            price = float(rate_data["5. Exchange Rate"])
            return f"{symbol}/USD: ${price:,.2f}"

    return f"No crypto data available for {symbol}"


# ---------------------------------------------------------------------------
# Forex rate — via Alpha Vantage
# ---------------------------------------------------------------------------

async def _forex_rate(from_currency: str, to_currency: str, settings: Settings) -> str:
    from_currency = from_currency.upper().strip()
    to_currency = to_currency.upper().strip()

    if settings.alphavantage_key:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "CURRENCY_EXCHANGE_RATE",
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "apikey": settings.alphavantage_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        rate_data = data.get("Realtime Currency Exchange Rate", {})
        if rate_data.get("5. Exchange Rate"):
            rate = float(rate_data["5. Exchange Rate"])
            return f"{from_currency}/{to_currency}: {rate:.4f}"

    if settings.finnhub_key:
        # Finnhub forex uses OANDA:EUR_USD format
        symbol = f"OANDA:{from_currency}_{to_currency}"
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": symbol, "token": settings.finnhub_key},
            )
            resp.raise_for_status()
            d = resp.json()
        if d.get("c") and d["c"] > 0:
            return f"{from_currency}/{to_currency}: {d['c']:.4f} (change: {d['dp']:.2f}%)"

    return f"No forex data available for {from_currency}/{to_currency}"


# ---------------------------------------------------------------------------
# Economic indicator — via FRED API
# ---------------------------------------------------------------------------

async def _economic_indicator(series_id: str, settings: Settings) -> str:
    series_id = series_id.upper().strip()

    if not settings.fred_key:
        return (
            f"FRED API key not configured. Cannot look up {series_id}. "
            "Set FRED_KEY in .env (free at https://fred.stlouisfed.org/docs/api/api_key.html)"
        )

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": settings.fred_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    observations = data.get("observations", [])
    if not observations:
        return f"No data found for FRED series {series_id}"

    # Also get series info for the title
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            "https://api.stlouisfed.org/fred/series",
            params={
                "series_id": series_id,
                "api_key": settings.fred_key,
                "file_type": "json",
            },
        )
        resp.raise_for_status()
        series_info = resp.json()

    title = series_id
    serieses = series_info.get("serieses", [])
    if serieses:
        title = serieses[0].get("title", series_id)
        units = serieses[0].get("units", "")
        frequency = serieses[0].get("frequency", "")
    else:
        units = ""
        frequency = ""

    lines = [f"{title} ({series_id})"]
    if units:
        lines[0] += f" — {units}"
    if frequency:
        lines[0] += f" ({frequency})"

    for obs in observations[:5]:
        date = obs.get("date", "?")
        value = obs.get("value", "N/A")
        lines.append(f"  {date}: {value}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Market index — reuse stock_price with ETF symbols
# ---------------------------------------------------------------------------

async def _market_index(symbol: str, settings: Settings) -> str:
    symbol = symbol.upper().strip()
    # Map common index names to ETF tickers
    index_map = {
        "S&P500": "SPY", "S&P 500": "SPY", "SP500": "SPY",
        "NASDAQ": "QQQ", "NASDAQ100": "QQQ",
        "DOW": "DIA", "DJIA": "DIA", "DOW JONES": "DIA",
        "RUSSELL": "IWM", "RUSSELL2000": "IWM",
        "VIX": "VIXY",
    }
    mapped = index_map.get(symbol, symbol)
    result = await _stock_price(mapped, settings)
    if mapped != symbol:
        return f"{symbol} (via {mapped}): " + result.split(": ", 1)[-1]
    return result


# ---------------------------------------------------------------------------
# Commodity price — via ETF proxy
# ---------------------------------------------------------------------------

async def _commodity_price(symbol: str, settings: Settings) -> str:
    symbol = symbol.upper().strip()
    # Map commodity names to ETF tickers
    commodity_map = {
        "GOLD": "GLD", "OIL": "USO", "CRUDE": "USO",
        "SILVER": "SLV", "NATGAS": "UNG", "NATURAL GAS": "UNG",
        "COPPER": "CPER", "WHEAT": "WEAT", "CORN": "CORN",
    }
    mapped = commodity_map.get(symbol, symbol)
    result = await _stock_price(mapped, settings)
    if mapped != symbol:
        return f"{symbol} (via {mapped} ETF): " + result.split(": ", 1)[-1]
    return result


# ---------------------------------------------------------------------------
# Company profile — via Finnhub
# ---------------------------------------------------------------------------

async def _company_profile(ticker: str, settings: Settings) -> str:
    ticker = ticker.upper().strip()

    if not settings.finnhub_key:
        return f"Finnhub API key required for company profiles. Cannot look up {ticker}."

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            "https://finnhub.io/api/v1/stock/profile2",
            params={"symbol": ticker, "token": settings.finnhub_key},
        )
        resp.raise_for_status()
        d = resp.json()

    if not d.get("name"):
        return f"No company profile found for {ticker}"

    lines = [
        f"{d.get('name', ticker)} ({ticker})",
        f"  Sector: {d.get('finnhubIndustry', 'N/A')}",
        f"  Market Cap: ${d.get('marketCapitalization', 0):,.0f}M",
        f"  Exchange: {d.get('exchange', 'N/A')}",
        f"  IPO Date: {d.get('ipo', 'N/A')}",
    ]
    if d.get("weburl"):
        lines.append(f"  Website: {d['weburl']}")

    return "\n".join(lines)
