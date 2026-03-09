#!/usr/bin/env python3
"""
AI Finance — Market Analysis Tool
Analyzes crypto market movements and generates AI-powered insights & predictions.

Usage:
    python main.py                              # Interactive mode
    python main.py --symbol BTC/USDT            # Analyze BTC
    python main.py --symbol ETH/USDT --tf 4h   # ETH on 4h timeframe
    python main.py --symbol SOL/USDT --tf 15m --exchange bybit
"""

import argparse
import sys
import os
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

import market_data as md
import technical_analysis as ta
import ai_analyzer

load_dotenv()

console = Console()

BANNER = """
╔═══════════════════════════════════════════════╗
║        AI FINANCE — Market Analyzer           ║
║   Powered by Claude Opus 4.6 + CCXT           ║
╚═══════════════════════════════════════════════╝
"""

DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_EXCHANGE = "binance"
DEFAULT_LIMIT = 200


def check_env():
    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print(
            "[red]Error:[/red] ANTHROPIC_API_KEY not set.\n"
            "Copy [bold].env.example[/bold] to [bold].env[/bold] and add your key.",
            style="bold red"
        )
        sys.exit(1)


def print_indicators_table(ind: dict, trend: dict):
    table = Table(title="Technical Indicators", box=box.ROUNDED, style="cyan")
    table.add_column("Indicator", style="bold white")
    table.add_column("Value", justify="right")
    table.add_column("Signal", justify="center")

    # RSI
    rsi = ind["rsi_14"]
    rsi_signal = "[red]Overbought[/red]" if rsi > 70 else "[green]Oversold[/green]" if rsi < 30 else "[yellow]Neutral[/yellow]"
    table.add_row("RSI (14)", f"{rsi:.2f}", rsi_signal)

    # MACD
    macd_signal = "[green]Bullish[/green]" if ind["macd"] > ind["macd_signal"] else "[red]Bearish[/red]"
    table.add_row("MACD", f"{ind['macd']:+.6f}", macd_signal)
    table.add_row("MACD Signal", f"{ind['macd_signal']:+.6f}", "")
    table.add_row("MACD Histogram", f"{ind['macd_histogram']:+.6f}", "[green]▲[/green]" if ind["macd_histogram"] > 0 else "[red]▼[/red]")

    # Stochastic
    stoch_signal = "[red]Overbought[/red]" if ind["stoch_k"] > 80 else "[green]Oversold[/green]" if ind["stoch_k"] < 20 else "[yellow]Neutral[/yellow]"
    table.add_row("Stoch K/D", f"{ind['stoch_k']:.1f} / {ind['stoch_d']:.1f}", stoch_signal)

    # BB
    pct_b = ind["bb_pct_b"]
    bb_signal = "[red]Near Upper[/red]" if pct_b > 0.8 else "[green]Near Lower[/green]" if pct_b < 0.2 else "[yellow]Mid-Band[/yellow]"
    table.add_row("BB %B", f"{pct_b:.3f}", bb_signal)
    table.add_row("BB Width", f"{ind['bb_bandwidth']:.4f}", "")

    # EMAs
    price = ind["price"]
    table.add_row("EMA 9/21/50/200",
                  f"{ind['ema_9']:,.2f} / {ind['ema_21']:,.2f} / {ind['ema_50']:,.2f} / {ind['ema_200']:,.2f}", "")

    # VWAP
    vwap_diff = (price - ind["vwap"]) / ind["vwap"] * 100
    vwap_signal = "[green]Above[/green]" if vwap_diff > 0 else "[red]Below[/red]"
    table.add_row("VWAP", f"{ind['vwap']:,.6f} ({vwap_diff:+.2f}%)", vwap_signal)

    # ATR
    atr_pct = ind["atr_14"] / price * 100
    table.add_row("ATR (14)", f"{ind['atr_14']:,.6f} ({atr_pct:.2f}%)", "")

    # Volume
    vol_signal = "[green]High[/green]" if ind["volume_vs_avg"] > 1.5 else "[yellow]Normal[/yellow]" if ind["volume_vs_avg"] > 0.7 else "[red]Low[/red]"
    table.add_row("Volume vs Avg", f"{ind['volume_vs_avg']:.2f}x", vol_signal)
    table.add_row("OBV Trend", ind["obv_trend"].upper(), "[green]▲[/green]" if ind["obv_trend"] == "rising" else "[red]▼[/red]")

    # Trend
    trend_color = "green" if trend["direction"] == "uptrend" else "red" if trend["direction"] == "downtrend" else "yellow"
    table.add_row("Trend", f"{trend['direction'].upper()} ({trend['strength']})", f"[{trend_color}]{trend['slope_10c_pct']:+.2f}%[/{trend_color}]")

    console.print(table)


def print_levels_table(sr: dict, vol_profile: dict, price: float):
    table = Table(title="Key Price Levels", box=box.ROUNDED, style="magenta")
    table.add_column("Type", style="bold white")
    table.add_column("Price", justify="right")
    table.add_column("Distance", justify="right")

    table.add_row("POC (Volume)", f"${vol_profile['point_of_control']:,.6f}",
                  f"{(vol_profile['point_of_control'] - price) / price * 100:+.2f}%")

    for r in sr["resistance_levels"]:
        dist = (r - price) / price * 100
        table.add_row("[red]Resistance[/red]", f"${r:,.6f}", f"[red]{dist:+.2f}%[/red]")

    for s in sr["support_levels"]:
        dist = (s - price) / price * 100
        table.add_row("[green]Support[/green]", f"${s:,.6f}", f"[green]{dist:+.2f}%[/green]")

    console.print(table)


def print_order_book(ob: dict):
    imbalance = ob["imbalance"]
    bar_width = 30
    buy_ratio = (imbalance + 1) / 2
    buy_blocks = int(buy_ratio * bar_width)
    sell_blocks = bar_width - buy_blocks
    bar = f"[green]{'█' * buy_blocks}[/green][red]{'█' * sell_blocks}[/red]"

    direction = "[green]BUY PRESSURE[/green]" if imbalance > 0.05 else "[red]SELL PRESSURE[/red]" if imbalance < -0.05 else "[yellow]BALANCED[/yellow]"

    console.print(Panel(
        f"Spread: {ob['spread_pct']:.4f}%  |  Imbalance: {imbalance:+.4f}  |  {direction}\n"
        f"Buy Vol: {ob['bid_volume']:.4f}  {bar}  Sell Vol: {ob['ask_volume']:.4f}",
        title="Order Book",
        border_style="blue",
    ))


def run_analysis(symbol: str, timeframe: str, exchange_id: str, limit: int):
    console.print(BANNER, style="bold cyan")

    check_env()

    # Step 1 — Fetch market data
    with console.status(f"[bold cyan]Fetching {symbol} data from {exchange_id}...[/bold cyan]"):
        try:
            market = md.get_market_summary(symbol, timeframe, limit, exchange_id)
        except Exception as e:
            console.print(f"[red]Failed to fetch market data:[/red] {e}")
            sys.exit(1)

    ticker = market["ticker"]
    price = ticker["price"]

    console.print(Panel(
        f"[bold white]{symbol}[/bold white]  |  "
        f"[bold {'green' if ticker['change_24h_pct'] >= 0 else 'red'}]"
        f"${price:,.6f}  ({ticker['change_24h_pct']:+.2f}%)[/bold {'green' if ticker['change_24h_pct'] >= 0 else 'red'}]\n"
        f"Timeframe: [cyan]{timeframe}[/cyan]  |  Exchange: [cyan]{exchange_id.capitalize()}[/cyan]  |  "
        f"Candles: [cyan]{market['candles_fetched']}[/cyan]",
        title="Market Overview",
        border_style="cyan",
    ))

    # Step 2 — Technical analysis
    with console.status("[bold cyan]Computing technical indicators...[/bold cyan]"):
        try:
            analysis = ta.full_analysis(market["ohlcv"])
        except Exception as e:
            console.print(f"[red]Technical analysis failed:[/red] {e}")
            sys.exit(1)

    # Print indicators
    print_indicators_table(analysis["latest_indicators"], analysis["trend"])
    print_levels_table(analysis["support_resistance"], analysis["volume_profile"], price)
    print_order_book(market["order_book"])

    # Print patterns
    if analysis["candlestick_patterns"]:
        console.print("\n[bold magenta]Candlestick Patterns Detected:[/bold magenta]")
        for p in analysis["candlestick_patterns"][-5:]:
            signal_color = "green" if "bullish" in p["signal"] or p["signal"] == "strong_bullish" else \
                           "red" if "bearish" in p["signal"] or p["signal"] == "strong_bearish" else "yellow"
            console.print(f"  • [{signal_color}]{p['pattern']}[/{signal_color}] — {p['signal'].replace('_', ' ').title()}")

    # Step 3 — AI Analysis
    console.print("\n")
    console.print(Panel(
        "[bold]Claude Opus 4.6 is analyzing the market data with adaptive thinking...[/bold]\n"
        "[dim]This may take 15-30 seconds for deep analysis.[/dim]",
        title="AI Analysis",
        border_style="yellow",
    ))
    console.print()

    try:
        _ = ai_analyzer.analyze_market(market, analysis, stream=True)
    except anthropic.AuthenticationError:
        console.print("[red]Invalid ANTHROPIC_API_KEY. Check your .env file.[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]AI analysis failed:[/red] {e}")
        sys.exit(1)

    console.print("\n")
    console.print(Panel(
        f"Analysis completed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        "[dim]This is not financial advice. Always do your own research.[/dim]",
        border_style="dim",
    ))


def interactive_mode():
    console.print(BANNER, style="bold cyan")
    console.print("[bold]Interactive Mode[/bold] — Configure your analysis\n")

    symbol = console.input(f"[cyan]Symbol[/cyan] (default: {DEFAULT_SYMBOL}): ").strip().upper() or DEFAULT_SYMBOL
    if "/" not in symbol:
        symbol = symbol + "/USDT"

    console.print(f"\nTimeframes: 1m, 5m, 15m, 1h, 4h, 1d, 1w")
    timeframe = console.input(f"[cyan]Timeframe[/cyan] (default: {DEFAULT_TIMEFRAME}): ").strip() or DEFAULT_TIMEFRAME

    console.print(f"\nExchanges: {', '.join(md.SUPPORTED_EXCHANGES)}")
    exchange_id = console.input(f"[cyan]Exchange[/cyan] (default: {DEFAULT_EXCHANGE}): ").strip().lower() or DEFAULT_EXCHANGE

    limit = console.input(f"[cyan]Candles to fetch[/cyan] (default: {DEFAULT_LIMIT}): ").strip()
    limit = int(limit) if limit.isdigit() else DEFAULT_LIMIT

    console.print()
    run_analysis(symbol, timeframe, exchange_id, limit)


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered crypto market analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --symbol BTC/USDT
  python main.py --symbol ETH/USDT --tf 4h
  python main.py --symbol SOL/USDT --tf 15m --exchange bybit --limit 300
        """
    )
    parser.add_argument("--symbol", "-s", default=None, help="Trading pair (e.g. BTC/USDT)")
    parser.add_argument("--tf", "--timeframe", "-t", default=DEFAULT_TIMEFRAME, help="Timeframe (1m/5m/15m/1h/4h/1d)")
    parser.add_argument("--exchange", "-e", default=DEFAULT_EXCHANGE, help=f"Exchange ({'/'.join(md.SUPPORTED_EXCHANGES)})")
    parser.add_argument("--limit", "-l", type=int, default=DEFAULT_LIMIT, help="Number of candles to fetch")

    args = parser.parse_args()

    if args.symbol is None:
        interactive_mode()
    else:
        symbol = args.symbol.upper()
        if "/" not in symbol:
            symbol = symbol + "/USDT"
        run_analysis(symbol, args.tf, args.exchange, args.limit)


if __name__ == "__main__":
    main()
