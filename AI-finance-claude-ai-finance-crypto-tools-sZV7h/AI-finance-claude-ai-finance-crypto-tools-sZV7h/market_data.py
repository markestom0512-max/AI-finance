"""
Market Data Fetcher
Fetches OHLCV candle data from crypto exchanges via CCXT.
"""

import ccxt
import pandas as pd
from datetime import datetime, timezone
from typing import Optional


SUPPORTED_EXCHANGES = ["binance", "kraken", "coinbase", "bybit", "okx"]

TIMEFRAME_LABELS = {
    "1m": "1 minute",
    "5m": "5 minutes",
    "15m": "15 minutes",
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day",
    "1w": "1 week",
}


def get_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    if exchange_id not in SUPPORTED_EXCHANGES:
        raise ValueError(f"Exchange '{exchange_id}' not supported. Choose from: {SUPPORTED_EXCHANGES}")
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True})


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 200,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data for a given symbol.

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT'
        timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w)
        limit: Number of candles to fetch (max ~500 depending on exchange)
        exchange_id: Exchange to use

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    exchange = get_exchange(exchange_id)

    if timeframe not in exchange.timeframes:
        available = list(exchange.timeframes.keys())
        raise ValueError(f"Timeframe '{timeframe}' not available. Options: {available}")

    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)

    return df


def fetch_ticker(symbol: str, exchange_id: str = "binance") -> dict:
    """Fetch current ticker data (price, 24h change, volume)."""
    exchange = get_exchange(exchange_id)
    ticker = exchange.fetch_ticker(symbol)
    return {
        "symbol": symbol,
        "price": ticker.get("last"),
        "bid": ticker.get("bid"),
        "ask": ticker.get("ask"),
        "change_24h_pct": ticker.get("percentage"),
        "high_24h": ticker.get("high"),
        "low_24h": ticker.get("low"),
        "volume_24h": ticker.get("baseVolume"),
        "quote_volume_24h": ticker.get("quoteVolume"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def fetch_order_book_summary(symbol: str, exchange_id: str = "binance", depth: int = 20) -> dict:
    """
    Fetch order book and compute bid/ask imbalance.
    Useful signal: high bid pressure = bullish, high ask pressure = bearish.
    """
    exchange = get_exchange(exchange_id)
    ob = exchange.fetch_order_book(symbol, limit=depth)

    bid_volume = sum(qty for _, qty in ob["bids"])
    ask_volume = sum(qty for _, qty in ob["asks"])
    total = bid_volume + ask_volume

    imbalance = (bid_volume - ask_volume) / total if total > 0 else 0
    spread = (ob["asks"][0][0] - ob["bids"][0][0]) if ob["asks"] and ob["bids"] else 0
    spread_pct = spread / ob["bids"][0][0] * 100 if ob["bids"] else 0

    return {
        "bid_volume": round(bid_volume, 4),
        "ask_volume": round(ask_volume, 4),
        "imbalance": round(imbalance, 4),   # positive = more buyers, negative = more sellers
        "spread": round(spread, 6),
        "spread_pct": round(spread_pct, 4),
        "best_bid": ob["bids"][0][0] if ob["bids"] else None,
        "best_ask": ob["asks"][0][0] if ob["asks"] else None,
    }


def get_market_summary(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 200,
    exchange_id: str = "binance",
) -> dict:
    """Fetch all market data needed for analysis."""
    ohlcv = fetch_ohlcv(symbol, timeframe, limit, exchange_id)
    ticker = fetch_ticker(symbol, exchange_id)
    ob_summary = fetch_order_book_summary(symbol, exchange_id)

    return {
        "ohlcv": ohlcv,
        "ticker": ticker,
        "order_book": ob_summary,
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange": exchange_id,
        "candles_fetched": len(ohlcv),
    }
