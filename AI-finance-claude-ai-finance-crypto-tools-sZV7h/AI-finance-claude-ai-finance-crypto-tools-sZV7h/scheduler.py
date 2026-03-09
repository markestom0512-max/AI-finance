#!/usr/bin/env python3
"""
AI Finance — Automated Scheduler
Runs market analyses on a configurable schedule and sends results to Telegram.

Usage:
    python scheduler.py                     # Run with config.yaml
    python scheduler.py --config my.yaml    # Custom config file
    python scheduler.py --dry-run           # Test without sending Telegram
    python scheduler.py --test-telegram     # Verify Telegram bot connection
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

import market_data as md
import technical_analysis as ta
import ai_analyzer
from notifier import TelegramNotifier

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# How often the main loop wakes up to check the schedule (seconds)
TICK_SECONDS = 60


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        log.error(f"Config file not found: {path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Schedule logic ────────────────────────────────────────────────────────────

def get_current_interval(schedule: list) -> int | None:
    """
    Return interval_minutes for the current day/time, or None if off.

    Walks through the schedule looking for the first slot that matches
    today's day name and current HH:MM time.
    """
    now = datetime.now()
    day_name = DAY_NAMES[now.weekday()]
    current_time = now.strftime("%H:%M")

    for entry in schedule:
        if day_name not in entry.get("days", []):
            continue
        for slot in entry.get("slots", []):
            if slot["from"] <= current_time < slot["to"]:
                return slot["interval_minutes"]
        # Day found but no slot matched → off for this day at this hour
        return None

    return None  # Day not listed → off


def describe_schedule(schedule: list) -> str:
    """Return a human-readable summary of the schedule."""
    lines = []
    for entry in schedule:
        days = ", ".join(d.capitalize() for d in entry.get("days", []))
        slots = entry.get("slots", [])
        if not slots:
            lines.append(f"  {days}: OFF")
        else:
            for slot in slots:
                lines.append(
                    f"  {days}  {slot['from']}–{slot['to']} "
                    f"→ every {slot['interval_minutes']} min"
                )
    return "\n".join(lines) if lines else "  (empty — all days OFF)"


# ── Analysis & formatting ─────────────────────────────────────────────────────

def run_analysis_for_asset(asset: dict) -> tuple[str, dict]:
    """
    Fetch market data, run technical analysis, and call Claude.

    Returns:
        (analysis_text, market_data_dict)
    """
    symbol = asset["symbol"]
    timeframe = asset.get("timeframe", "1h")
    exchange_id = asset.get("exchange", "binance")
    limit = asset.get("limit", 200)

    log.info(f"Fetching {symbol} data from {exchange_id}...")
    market = md.get_market_summary(symbol, timeframe, limit, exchange_id)

    log.info(f"Computing technical indicators for {symbol}...")
    analysis = ta.full_analysis(market["ohlcv"])

    log.info(f"Requesting AI analysis for {symbol} from Claude...")
    text = ai_analyzer.analyze_market(market, analysis, stream=False)

    return text, market


def format_telegram_header(symbol: str, market: dict, timeframe: str) -> str:
    """Build the header block that precedes the AI analysis in Telegram."""
    ticker = market["ticker"]
    price = ticker["price"]
    change = ticker["change_24h_pct"]
    arrow = "📈" if change >= 0 else "📉"
    color = "+" if change >= 0 else ""

    ob = market["order_book"]
    imbalance = ob["imbalance"]
    ob_label = "Buy pressure" if imbalance > 0.05 else "Sell pressure" if imbalance < -0.05 else "Balanced"

    return (
        f"{arrow} *{symbol}* — `${price:,.4f}`  ({color}{change:.2f}%)\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  TF: {timeframe}\n"
        f"📊 Order book: {ob_label} ({imbalance:+.3f})\n"
        f"{'─' * 32}\n\n"
    )


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_scheduler(config: dict, notifier: TelegramNotifier | None, dry_run: bool):
    assets = config["assets"]
    schedule = config["schedule"]

    log.info(f"Monitoring {len(assets)} asset(s):")
    for a in assets:
        log.info(f"  • {a['symbol']} on {a.get('exchange', 'binance')} ({a.get('timeframe', '1h')})")

    log.info("Schedule:\n" + describe_schedule(schedule))

    # Track last successful run time per asset
    last_run: dict[str, datetime | None] = {a["symbol"]: None for a in assets}

    if notifier:
        asset_list = ", ".join(a["symbol"] for a in assets)
        notifier.send(
            f"🤖 *AI Finance Scheduler started*\n"
            f"Assets: `{asset_list}`\n\n"
            f"Schedule:\n```\n{describe_schedule(schedule)}\n```"
        )

    while True:
        now = datetime.now()
        interval = get_current_interval(schedule)

        if interval is None:
            log.info("Time slot: OFF — sleeping.")
        else:
            log.info(f"Time slot: active — interval {interval} min")

            for asset in assets:
                symbol = asset["symbol"]
                last = last_run[symbol]
                elapsed_min = (now - last).total_seconds() / 60 if last else float("inf")

                if elapsed_min < interval:
                    remaining = int(interval - elapsed_min)
                    log.info(f"  {symbol}: next analysis in ~{remaining} min")
                    continue

                log.info(f"  {symbol}: running analysis...")

                if dry_run:
                    log.info(f"  [DRY RUN] Skipping actual analysis for {symbol}")
                    last_run[symbol] = now
                    continue

                try:
                    analysis_text, market = run_analysis_for_asset(asset)
                    header = format_telegram_header(symbol, market, asset.get("timeframe", "1h"))
                    full_message = header + analysis_text

                    if notifier:
                        notifier.send_long(full_message)
                        log.info(f"  {symbol}: analysis sent to Telegram.")
                    else:
                        log.info(f"  {symbol}: analysis done (no notifier).")

                    last_run[symbol] = now

                except Exception as e:
                    log.error(f"  {symbol}: analysis failed — {e}")
                    if notifier:
                        notifier.send(f"❌ *{symbol}* analysis failed:\n`{e}`")

        time.sleep(TICK_SECONDS)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Finance — Automated scheduler with Telegram notifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scheduler.py                      # Run with config.yaml
  python scheduler.py --dry-run            # Test schedule without analysis
  python scheduler.py --test-telegram      # Verify Telegram bot
  python scheduler.py --config my.yaml     # Custom config file
        """,
    )
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Run without calling Claude or sending Telegram messages")
    parser.add_argument("--test-telegram", action="store_true", help="Test Telegram connection and exit")
    args = parser.parse_args()

    # Validate environment
    if not os.getenv("ANTHROPIC_API_KEY") and not args.dry_run and not args.test_telegram:
        log.error("ANTHROPIC_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    # Build notifier
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    notifier = None

    if token and chat_id:
        notifier = TelegramNotifier(token, chat_id)
    elif not args.dry_run:
        log.warning(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — "
            "analyses will run but NOT be sent anywhere.\n"
            "  → Add them to .env, or use --dry-run to test the schedule."
        )

    # --test-telegram
    if args.test_telegram:
        if not notifier:
            log.error("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env first.")
            sys.exit(1)
        success = notifier.test_connection()
        sys.exit(0 if success else 1)

    # Load config & start
    config = load_config(args.config)

    try:
        run_scheduler(config, notifier, dry_run=args.dry_run)
    except KeyboardInterrupt:
        log.info("Scheduler stopped by user.")
        if notifier:
            notifier.send("🛑 *AI Finance Scheduler stopped.*")


if __name__ == "__main__":
    main()
