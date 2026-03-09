# AI Finance — Market Analyzer

An AI-powered tool that analyzes crypto market movements in real-time and generates precise insights & predictions using Claude Opus 4.6.

## Architecture

```
Market Data (CCXT)
    ↓
Technical Analysis Engine
  • RSI, MACD, Bollinger Bands
  • Stochastic, ATR, EMA (9/21/50/200)
  • VWAP, OBV, Volume Profile
  • Candlestick Pattern Detection
  • Support/Resistance Levels
    ↓
Order Book Analysis
  • Bid/Ask Imbalance
  • Spread monitoring
    ↓
Claude Opus 4.6 (Adaptive Thinking)
  • Market context & trend
  • Signal confluence
  • Price predictions
  • Risk assessment
    ↓
Telegram (automated scheduler)
  • Configurable time slots per day
  • Intense / light / off modes
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your keys in .env
```

## Usage — Manual analysis

```bash
# Interactive mode
python main.py

# Analyze BTC/USDT on 1h (default)
python main.py --symbol BTC/USDT

# ETH on 4h timeframe
python main.py --symbol ETH/USDT --tf 4h

# SOL on 15m via Bybit, 300 candles
python main.py --symbol SOL/USDT --tf 15m --exchange bybit --limit 300
```

## Usage — Automated scheduler with Telegram

The scheduler runs 24/7 in the background and sends analyses to Telegram on a
fully configurable schedule (intense / light / off slots per day).

### 1 — Create a Telegram bot

1. Open Telegram and message `@BotFather` → `/newbot`
2. Copy the bot token you receive
3. Send any message to your new bot, then open:
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
   Find `"id"` under `"chat"` — that is your `TELEGRAM_CHAT_ID`

### 2 — Configure `.env`

```env
ANTHROPIC_API_KEY=...
TELEGRAM_BOT_TOKEN=123456:ABCdef...
TELEGRAM_CHAT_ID=987654321
```

### 3 — Edit `config.yaml`

```yaml
assets:
  - symbol: BTC/USDT
    timeframe: 1h
    exchange: binance

schedule:
  - days: [monday, tuesday, wednesday, thursday, friday]
    slots:
      - from: "08:00"
        to: "12:00"
        interval_minutes: 30    # intense — every 30 min
      - from: "18:00"
        to: "22:00"
        interval_minutes: 60    # moderate — every hour
  - days: [saturday]
    slots:
      - from: "10:00"
        to: "14:00"
        interval_minutes: 120   # light — every 2h
  # sunday: not listed = OFF
```

### 4 — Run

```bash
# Verify Telegram connection
python scheduler.py --test-telegram

# Dry run (test schedule logic without calling Claude)
python scheduler.py --dry-run

# Start the scheduler
python scheduler.py
```

## Supported Exchanges
- Binance (default)
- Kraken
- Coinbase
- Bybit
- OKX

## AI Analysis Sections

Claude produces 6 structured sections per analysis:

| Section | Content |
|---------|---------|
| A — Market Context | Overall structure, trend phase |
| B — Signal Confluence | Top 3-5 signals, strength rating |
| C — Critical Levels | Key S/R, VWAP, POC implications |
| D — Short-Term Prediction | Primary + alternative scenarios with probabilities |
| E — Risk Assessment | Stop-loss zones, R/R ratio |
| F — Actionable Summary | Clear directional stance |

## Disclaimer

This tool is for educational and research purposes only. It does not constitute financial advice. Always do your own research before making any trading decisions.
