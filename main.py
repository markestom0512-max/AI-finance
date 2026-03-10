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
from notifier import TelegramNotifier

load_dotenv()

console = Console()

BANNER = """
╔═══════════════════════════════════════════════╗
║        AI FINANCE — Market Analyzer           ║
║   Powered by Claude Haiku 4.5 + CCXT          ║
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


def print_key_signals(ind: dict, trend: dict, ob: dict):
    """Print a compact table with only the most important signals."""
    table = Table(title="Indicateurs Clés", box=box.SIMPLE_HEAVY, style="cyan", show_header=True)
    table.add_column("Indicateur", style="bold white", min_width=18)
    table.add_column("Valeur", justify="right", min_width=14)
    table.add_column("Signal", justify="center", min_width=12)

    # RSI
    rsi = ind["rsi_14"]
    rsi_signal = "[red]Suracheté[/red]" if rsi > 70 else "[green]Survendu[/green]" if rsi < 30 else "[yellow]Neutre[/yellow]"
    table.add_row("RSI (14)", f"{rsi:.1f}", rsi_signal)

    # MACD
    macd_signal_str = "[green]Haussier[/green]" if ind["macd_histogram"] > 0 else "[red]Baissier[/red]"
    table.add_row("MACD Histo", f"{ind['macd_histogram']:+.5f}", macd_signal_str)

    # Trend
    trend_color = "green" if trend["direction"] == "uptrend" else "red" if trend["direction"] == "downtrend" else "yellow"
    trend_label = "Haussier" if trend["direction"] == "uptrend" else "Baissier" if trend["direction"] == "downtrend" else "Latéral"
    table.add_row("Tendance", f"{trend_label} ({trend['strength']})", f"[{trend_color}]{trend['slope_10c_pct']:+.2f}%[/{trend_color}]")

    # BB %B
    pct_b = ind["bb_pct_b"]
    bb_signal = "[red]Bande haute[/red]" if pct_b > 0.8 else "[green]Bande basse[/green]" if pct_b < 0.2 else "[yellow]Milieu[/yellow]"
    table.add_row("BB %B", f"{pct_b:.3f}", bb_signal)

    # VWAP
    vwap_diff = (ind["price"] - ind["vwap"]) / ind["vwap"] * 100
    vwap_signal = "[green]Au-dessus[/green]" if vwap_diff > 0 else "[red]En-dessous[/red]"
    table.add_row("vs VWAP", f"{vwap_diff:+.2f}%", vwap_signal)

    # Volume
    vol_signal = "[green]Élevé[/green]" if ind["volume_vs_avg"] > 1.5 else "[yellow]Normal[/yellow]" if ind["volume_vs_avg"] > 0.7 else "[red]Faible[/red]"
    table.add_row("Volume vs Moy", f"{ind['volume_vs_avg']:.2f}x", vol_signal)

    # Order book pressure
    imb = ob["imbalance"]
    ob_signal = "[green]Acheteurs[/green]" if imb > 0.05 else "[red]Vendeurs[/red]" if imb < -0.05 else "[yellow]Équilibré[/yellow]"
    table.add_row("Carnet ordres", f"{imb:+.3f}", ob_signal)

    console.print(table)


def print_levels(sr: dict, vol_profile: dict, price: float):
    """Print the 2 most relevant support/resistance levels + POC."""
    lines = []
    lines.append(f"  POC (Volume)  ${vol_profile['point_of_control']:,.4f}  ({(vol_profile['point_of_control']-price)/price*100:+.2f}%)")

    for r in sr["resistance_levels"][:2]:
        lines.append(f"  [red]Résistance[/red]    ${r:,.4f}  ({(r-price)/price*100:+.2f}%)")

    for s in sr["support_levels"][:2]:
        lines.append(f"  [green]Support[/green]       ${s:,.4f}  ({(s-price)/price*100:+.2f}%)")

    console.print(Panel("\n".join(lines), title="Niveaux Clés", border_style="magenta"))


def run_analysis(symbol: str, timeframe: str, exchange_id: str, limit: int):
    console.print(BANNER, style="bold cyan")

    check_env()

    # Step 1 — Fetch market data
    with console.status(f"[bold cyan]Récupération {symbol} depuis {exchange_id}...[/bold cyan]"):
        try:
            market = md.get_market_summary(symbol, timeframe, limit, exchange_id)
        except Exception as e:
            console.print(f"[red]Échec récupération données:[/red] {e}")
            sys.exit(1)

    ticker = market["ticker"]
    price = ticker["price"]

    console.print(Panel(
        f"[bold white]{symbol}[/bold white]  |  "
        f"[bold {'green' if ticker['change_24h_pct'] >= 0 else 'red'}]"
        f"${price:,.4f}  ({ticker['change_24h_pct']:+.2f}%)[/bold {'green' if ticker['change_24h_pct'] >= 0 else 'red'}]\n"
        f"Timeframe: [cyan]{timeframe}[/cyan]  |  Exchange: [cyan]{exchange_id.capitalize()}[/cyan]",
        title="Aperçu Marché",
        border_style="cyan",
    ))

    # Step 2 — Technical analysis
    with console.status("[bold cyan]Calcul des indicateurs...[/bold cyan]"):
        try:
            analysis = ta.full_analysis(market["ohlcv"])
        except Exception as e:
            console.print(f"[red]Analyse technique échouée:[/red] {e}")
            sys.exit(1)

    print_key_signals(analysis["latest_indicators"], analysis["trend"], market["order_book"])
    print_levels(analysis["support_resistance"], analysis["volume_profile"], price)

    # Patterns (only if notable)
    notable_patterns = [p for p in analysis["candlestick_patterns"][-3:] if p["signal"] != "neutral"]
    if notable_patterns:
        pat_lines = []
        for p in notable_patterns:
            color = "green" if "bullish" in p["signal"] else "red" if "bearish" in p["signal"] else "yellow"
            pat_lines.append(f"  [{color}]{p['pattern']}[/{color}] — {p['signal'].replace('_', ' ').title()}")
        console.print(Panel("\n".join(pat_lines), title="Patterns Chandelier", border_style="yellow"))

    # Step 3 — AI Analysis
    console.print("\n")
    console.print(Panel(
        "[bold]Analyse IA en cours (Haiku 4.5)...[/bold]\n"
        "[dim]Génération des scénarios et conseils...[/dim]",
        title="Analyse IA",
        border_style="yellow",
    ))
    console.print()

    try:
        analysis_text = ai_analyzer.analyze_market(market, analysis, stream=True)
    except anthropic.AuthenticationError:
        console.print("[red]ANTHROPIC_API_KEY invalide. Vérifiez votre fichier .env.[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Analyse IA échouée:[/red] {e}")
        sys.exit(1)

    # Send to Telegram if configured
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if tg_token and tg_chat:
        notifier = TelegramNotifier(tg_token, tg_chat)
        change = ticker["change_24h_pct"]
        arrow = "📈" if change >= 0 else "📉"
        header = (
            f"{arrow} *{symbol}* — `${price:,.4f}`  ({change:+.2f}%)\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  TF: {timeframe}\n"
            f"{'─' * 32}\n\n"
        )
        notifier.send_long(header + analysis_text)
        console.print("[green]Analyse envoyée sur Telegram.[/green]")
    else:
        console.print("[dim]Telegram non configuré (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID manquants dans .env).[/dim]")

    console.print("\n")
    console.print(Panel(
        f"Analyse terminée à {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        "[dim]Ceci n'est pas un conseil financier. Faites toujours vos propres recherches.[/dim]",
        border_style="dim",
    ))


def interactive_mode():
    console.print(BANNER, style="bold cyan")
    console.print("[bold]Mode Interactif[/bold] — Configurez votre analyse\n")

    symbol = console.input(f"[cyan]Symbole[/cyan] (défaut: {DEFAULT_SYMBOL}): ").strip().upper() or DEFAULT_SYMBOL
    if "/" not in symbol:
        symbol = symbol + "/USDT"

    console.print(f"\nTimeframes: 1m, 5m, 15m, 1h, 4h, 1d, 1w")
    timeframe = console.input(f"[cyan]Timeframe[/cyan] (défaut: {DEFAULT_TIMEFRAME}): ").strip() or DEFAULT_TIMEFRAME

    console.print(f"\nExchanges: {', '.join(md.SUPPORTED_EXCHANGES)}")
    exchange_id = console.input(f"[cyan]Exchange[/cyan] (défaut: {DEFAULT_EXCHANGE}): ").strip().lower() or DEFAULT_EXCHANGE

    limit = console.input(f"[cyan]Bougies à récupérer[/cyan] (défaut: {DEFAULT_LIMIT}): ").strip()
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
